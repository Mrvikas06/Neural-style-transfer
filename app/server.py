"""High-performance ASGI server for Neural Style Transfer Studio.

Exposes REST APIs, real-time Server-Sent Events (SSE) for iterative optimization streaming,
and serves the flagship modern web studio frontend.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PIL import Image
import torch
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from nst.config import HardwareConfig, ImageConfig, LossConfig, NSTConfig, OptimizationConfig
from nst.engine import run_style_transfer
from nst.image_utils import image_to_tensor, load_image, tensor_to_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nst_server")

static_dir = root_dir / "app" / "static"
assets_dir = root_dir / "assets"
outputs_dir = root_dir / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

PRESETS_DATA = {
    "picasso_ballerina": {
        "id": "picasso_ballerina",
        "title": "Cubist Ballerina",
        "artist": "Pablo Picasso",
        "movement": "Cubism (1909)",
        "content_url": "/assets/content/dancing.jpg",
        "style_url": "/assets/style/picasso.jpg",
        "content_path": assets_dir / "content" / "dancing.jpg",
        "style_path": assets_dir / "style" / "picasso.jpg",
        "default_style_weight": 1e5,
        "default_content_weight": 1.0,
        "note": "Translates kinetic ballet posture into geometric facets and earthy cubist planes.",
    },
    "monet_bridge": {
        "id": "monet_bridge",
        "title": "Impressionist Bridge",
        "artist": "Claude Monet",
        "movement": "Impressionism (1899)",
        "content_url": "/assets/content/bridge.jpg",
        "style_url": "/assets/style/monet.jpg",
        "content_path": assets_dir / "content" / "bridge.jpg",
        "style_path": assets_dir / "style" / "monet.jpg",
        "default_style_weight": 2e5,
        "default_content_weight": 1.0,
        "note": "Infuses suspension bridge cables with oil-on-canvas impasto and water lily reflections.",
    },
    "starry_ballerina": {
        "id": "starry_ballerina",
        "title": "Starry Night Nocturne",
        "artist": "Vincent van Gogh",
        "movement": "Post-Impressionism (1889)",
        "content_url": "/assets/content/dancing.jpg",
        "style_url": "/assets/style/starry_night.png",
        "content_path": assets_dir / "content" / "dancing.jpg",
        "style_path": assets_dir / "style" / "starry_night.png",
        "default_style_weight": 1.2e5,
        "default_content_weight": 1.0,
        "note": "Applies swirling cobalt brushstrokes and energetic celestial vortices.",
    },
}


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
async def index_view(request: Request) -> FileResponse:
    """Serve the flagship modern web studio index."""
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return JSONResponse({"status": "error", "message": "Static assets not yet built"}, status_code=404)
    return FileResponse(index_file)


async def api_status(request: Request) -> JSONResponse:
    """Return hardware status, VRAM telemetry, and device availability."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "Universal CPU"
    vram_allocated_mb = 0.0
    vram_total_mb = 0

    if cuda_available:
        vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_total_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

    return JSONResponse(
        {
            "status": "online",
            "version": "0.2.0",
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram_allocated_mb": round(vram_allocated_mb, 2),
            "vram_total_mb": vram_total_mb,
            "torch_version": torch.__version__,
        }
    )


async def api_presets(request: Request) -> JSONResponse:
    """Return curated exhibition presets."""
    serialized = {}
    for key, val in PRESETS_DATA.items():
        serialized[key] = {
            "id": val["id"],
            "title": val["title"],
            "artist": val["artist"],
            "movement": val["movement"],
            "content_url": val["content_url"],
            "style_url": val["style_url"],
            "default_style_weight": val["default_style_weight"],
            "default_content_weight": val["default_content_weight"],
            "note": val["note"],
        }
    return JSONResponse({"presets": serialized})


async def api_stylize_stream(request: Request) -> StreamingResponse:
    """Execute style transfer with real-time SSE stream for preview frames and loss curves."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    preset_id = body.get("preset_id")
    resolution = int(body.get("resolution", 256))
    steps = int(body.get("steps", 100))
    style_weight = float(body.get("style_weight", 1e5))
    content_weight = float(body.get("content_weight", 1.0))
    tv_weight = float(body.get("tv_weight", 1e-4))
    optimizer_type = body.get("optimizer", "lbfgs")

    content_data = body.get("content_base64")
    style_data = body.get("style_base64")

    # Resolve input images
    if preset_id and preset_id in PRESETS_DATA:
        p = PRESETS_DATA[preset_id]
        content_img = Image.open(p["content_path"]).convert("RGB")
        style_img = Image.open(p["style_path"]).convert("RGB")
    elif content_data and style_data:
        try:
            content_bytes = base64.b64decode(content_data.split(",")[-1])
            style_bytes = base64.b64decode(style_data.split(",")[-1])
            content_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
            style_img = Image.open(io.BytesIO(style_bytes)).convert("RGB")
        except Exception as e:
            return JSONResponse({"error": f"Failed to decode base64 images: {e}"}, status_code=400)
    else:
        # Fallback to default preset
        p = PRESETS_DATA["picasso_ballerina"]
        content_img = Image.open(p["content_path"]).convert("RGB")
        style_img = Image.open(p["style_path"]).convert("RGB")

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    config = NSTConfig(
        hardware=HardwareConfig(device=device),
        image=ImageConfig(default_size=resolution, preserve_aspect_ratio=True),
        loss=LossConfig(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            loss_normalization="mean",
        ),
        optimization=OptimizationConfig(
            optimizer=optimizer_type,
            max_iter=steps,
            lr=1.0 if optimizer_type == "lbfgs" else 0.03,
        ),
    )

    event_queue: queue.Queue = queue.Queue()

    def run_worker() -> None:
        try:
            target_size = config.image.default_size
            content_tensor = image_to_tensor(
                content_img,
                target_size=target_size,
                preserve_aspect_ratio=True,
                device=device,
            )

            _, _, h, w = content_tensor.shape
            style_resized = style_img.resize((w, h), Image.Resampling.LANCZOS)
            style_tensor = image_to_tensor(
                style_resized,
                target_size=max(h, w),
                preserve_aspect_ratio=False,
                device=device,
            )

            last_preview_time = time.perf_counter()

            def step_cb(s: int, current_tensor: torch.Tensor, current_loss: float, parts: Dict[str, float]) -> None:
                nonlocal last_preview_time
                now = time.perf_counter()
                # Yield updates every 5 steps or on last step
                if s % 5 == 0 or s == steps or (now - last_preview_time) > 0.4:
                    last_preview_time = now
                    pil_img = tensor_to_image(current_tensor)
                    buffer = io.BytesIO()
                    # JPEG for fast streaming preview
                    pil_img.save(buffer, format="JPEG", quality=80)
                    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    event_queue.put(
                        {
                            "type": "progress",
                            "step": s,
                            "total": steps,
                            "loss": round(current_loss, 4),
                            "content_loss": round(parts["content"], 4),
                            "style_loss": round(parts["style"], 4),
                            "tv_loss": round(parts["tv"], 4),
                            "preview": f"data:image/jpeg;base64,{b64_str}",
                        }
                    )

            start_t = time.perf_counter()
            output_tensor = run_style_transfer(
                content_img=content_tensor,
                style_img=style_tensor,
                config=config,
                callback=step_cb,
            )
            elapsed_t = time.perf_counter() - start_t

            final_img = tensor_to_image(output_tensor)
            out_filename = f"stylized_{int(time.time())}.png"
            out_path = outputs_dir / out_filename
            final_img.save(out_path, format="PNG")

            event_queue.put(
                {
                    "type": "complete",
                    "result_url": f"/outputs/{out_filename}",
                    "elapsed": round(elapsed_t, 2),
                    "steps": steps,
                    "resolution": f"{final_img.width}x{final_img.height}",
                    "throughput": round(steps / elapsed_t, 1),
                    "device": torch.cuda.get_device_name(0) if cuda_available else "CPU",
                }
            )

        except Exception as exc:
            logger.error("Error during style transfer optimization: %s", exc, exc_info=True)
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put({"type": "eof"})

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()

    async def sse_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        while True:
            # Poll from thread queue without blocking the async event loop
            try:
                event = await loop.run_in_executor(None, event_queue.get, True, 0.5)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if event.get("type") == "eof":
                break

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# App Router & Static File Mounts
# ---------------------------------------------------------------------------
routes = [
    Route("/", endpoint=index_view),
    Route("/api/status", endpoint=api_status, methods=["GET"]),
    Route("/api/presets", endpoint=api_presets, methods=["GET"]),
    Route("/api/stylize", endpoint=api_stylize_stream, methods=["POST"]),
    Mount("/static", app=StaticFiles(directory=static_dir), name="static"),
    Mount("/assets", app=StaticFiles(directory=assets_dir), name="assets"),
    Mount("/outputs", app=StaticFiles(directory=outputs_dir), name="outputs"),
]

app = Starlette(routes=routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the Uvicorn ASGI server."""
    logger.info("Starting NST Pro Studio server on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
