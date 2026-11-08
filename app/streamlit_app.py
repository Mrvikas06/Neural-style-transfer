"""Neural Style Transfer — Modern Studio Interface.

A bespoke, dark-mode creative suite powered by PyTorch, VGG-19, and CUDA acceleration.
Designed with modern typography, glassmorphic surfaces, and curated artistic presets.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure src/ is on sys.path for standalone Streamlit execution
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
import torch
from PIL import Image

from nst.config import HardwareConfig, ImageConfig, LossConfig, NSTConfig, OptimizationConfig
from nst.engine import run_style_transfer
from nst.image_utils import image_to_tensor, load_image, tensor_to_image

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NST Atelier — Neural Art Studio",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Precision Studio Design System (Linear / Teenage Engineering / Leica inspired)
# ---------------------------------------------------------------------------
STUDIO_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset and base typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #ECECED;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit default chrome & decoration */
header[data-testid="stHeader"] {
    background: transparent !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}
footer {
    display: none !important;
}

/* Canvas background: Matte Charcoal Slate */
.stApp {
    background-color: #0A0A0C !important;
    background-image: 
        radial-gradient(ellipse 60% 40% at 50% -20%, rgba(255, 255, 255, 0.03), transparent),
        linear-gradient(180deg, #0A0A0C 0%, #0E0F12 100%) !important;
    background-attachment: fixed;
}

/* Adjust top padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* Sidebar styling: Minimalist darkroom control panel */
section[data-testid="stSidebar"] {
    background-color: #0E0F12 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.07) !important;
}

/* Top Studio Navigation Header */
.atelier-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.atelier-brand {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
}

.atelier-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}

.atelier-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #71717A;
    background: #18191E;
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

/* Status indicator badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #141518;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #A1A1AA;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #22C55E;
}

.status-dot-cpu {
    background-color: #EAB308;
}

/* Precision Cards */
.studio-panel {
    background: #121316;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.panel-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #71717A;
    margin-bottom: 0.75rem;
}

/* Segmented Control Tabs styling */
div[data-baseweb="tab-list"] {
    background-color: #121316 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}

div[data-baseweb="tab"] {
    border-radius: 6px !important;
    border: none !important;
    padding: 8px 18px !important;
    color: #8E8E93 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    background: transparent !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #202227 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}

div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Retro Dieter Rams Tactile Primary Action Button */
div.stButton > button:first-child {
    background: #FF5500 !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.75rem !important;
    border-radius: 6px !important;
    border: 1px solid #FF7733 !important;
    box-shadow: 0 4px 0 #C43D00, 0 6px 16px rgba(255, 85, 0, 0.3) !important;
    transition: all 0.1s ease !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

div.stButton > button:first-child:hover {
    background: #FF661A !important;
    border-color: #FF884D !important;
    transform: translateY(1px) !important;
    box-shadow: 0 3px 0 #C43D00, 0 8px 20px rgba(255, 85, 0, 0.4) !important;
}

div.stButton > button:first-child:active {
    transform: translateY(4px) !important;
    box-shadow: 0 0 0 #C43D00 !important;
}

/* Secondary & Download Buttons */
div[data-testid="stDownloadButton"] > button {
    background: #18191E !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.15s ease !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: #22242B !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* Telemetry Grid */
.telemetry-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.telemetry-cell {
    background: #141519;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

.cell-caption {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #71717A;
}

.cell-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FAFAFA;
    margin-top: 0.25rem;
}

/* Image Frames */
div[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background-color: #0E0F12 !important;
}

/* File Uploader Restyling */
div[data-testid="stFileUploader"] section {
    background-color: #121316 !important;
    border: 1px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div {
    background-color: #FFFFFF !important;
    border-radius: 4px !important;
}

div[data-testid="stProgress"] > div > div {
    background-color: #18191E !important;
    border-radius: 4px !important;
    height: 6px !important;
}
</style>
"""

st.markdown(STUDIO_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Exhibition Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "picasso_ballerina": {
        "title": "Cubist Ballerina",
        "artist": "Pablo Picasso",
        "movement": "Cubism (1909)",
        "content_path": Path("assets/content/dancing.jpg"),
        "style_path": Path("assets/style/picasso.jpg"),
        "recommended_style_weight": 1e5,
        "recommended_content_weight": 1.0,
        "note": "Translates figurative human motion into geometric facets and earth-tone planes.",
    },
    "monet_bridge": {
        "title": "Impressionist Bridge",
        "artist": "Claude Monet",
        "movement": "Impressionism (1899)",
        "content_path": Path("assets/content/bridge.jpg"),
        "style_path": Path("assets/style/monet.jpg"),
        "recommended_style_weight": 2e5,
        "recommended_content_weight": 1.0,
        "note": "Infuses suspension bridge engineering with organic oil dab textures and dappled light.",
    },
    "starry_ballerina": {
        "title": "Starry Night Nocturne",
        "artist": "Vincent van Gogh",
        "movement": "Post-Impressionism (1889)",
        "content_path": Path("assets/content/dancing.jpg"),
        "style_path": Path("assets/style/starry_night.png"),
        "recommended_style_weight": 1e5,
        "recommended_content_weight": 1.0,
        "note": "Applies vibrant impasto swirling brushstrokes across the portrait background.",
    },
}


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def initialize_app_state() -> None:
    if "stylized_image" not in st.session_state:
        st.session_state.stylized_image = None
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


# ---------------------------------------------------------------------------
# Sidebar Parameter Panel
# ---------------------------------------------------------------------------
def render_sidebar(cuda_available: bool, gpu_name: str) -> NSTConfig:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 0.5rem">
                <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #71717A;">
                    Engine Settings
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Style Strength
        intensity_options = {
            "Subtle (2×10⁴)": 2e4,
            "Balanced (1×10⁵)": 1e5,
            "Vivid (3×10⁵)": 3e5,
            "Heavy (8×10⁵)": 8e5,
        }
        selected_intensity_label = st.select_slider(
            "Style Intensity",
            options=list(intensity_options.keys()),
            value="Balanced (1×10⁵)",
            help="Governs how aggressively Gram-matrix textures supersede photo content.",
        )
        style_weight = intensity_options[selected_intensity_label]

        # Resolution
        resolution = st.select_slider(
            "Working Resolution",
            options=[192, 256, 384, 512],
            value=256,
            format_func=lambda r: f"{r}px (4.2s)" if r == 256 else f"{r}px",
            help="256px delivers high fidelity with conservative VRAM footprint.",
        )

        # Iterations
        steps = st.slider(
            "Optimization Iterations",
            min_value=30,
            max_value=200,
            value=100,
            step=10,
        )

        # Advanced Settings
        with st.expander("Advanced Hyperparameters", expanded=False):
            optimizer = st.selectbox(
                "Optimizer",
                options=["lbfgs", "adam"],
                index=0,
                help="L-BFGS uses second-order curvature approximation. Adam uses first-order momentum.",
            )
            content_weight = st.slider(
                "Content Weight (α)",
                min_value=0.2,
                max_value=3.0,
                value=1.0,
                step=0.2,
            )
            tv_weight = st.selectbox(
                "Total Variation (Smoothness)",
                options=[0.0, 1e-5, 1e-4, 5e-4],
                index=2,
                format_func=lambda w: "None" if w == 0.0 else f"{w:.1e}",
            )

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #52525B; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1rem;">
                <div>VGG-19 Layer Extractor: <b>Pruned relu5_1</b></div>
                <div>Gram Normalization: <b>Resolution-Invariant MSE</b></div>
                <div>Runtime Platform: <b>{gpu_name if cuda_available else 'CPU Fallback'}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return NSTConfig(
        hardware=HardwareConfig(device="cuda" if cuda_available else "cpu"),
        image=ImageConfig(default_size=resolution, preserve_aspect_ratio=True),
        loss=LossConfig(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            loss_normalization="mean",
        ),
        optimization=OptimizationConfig(
            optimizer=optimizer,
            max_iter=steps,
            lr=1.0 if optimizer == "lbfgs" else 0.03,
        ),
    )


# ---------------------------------------------------------------------------
# Main Application View
# ---------------------------------------------------------------------------
def main() -> None:
    initialize_app_state()

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "Universal CPU"

    # Navigation Header
    st.markdown(
        f"""
        <div class="atelier-nav">
            <div class="atelier-brand">
                <span class="atelier-logo">NST Atelier</span>
                <span class="atelier-version">v0.2.0</span>
            </div>
            <div class="status-pill">
                <span class="status-dot {'status-dot-cpu' if not cuda_available else ''}"></span>
                <span>{gpu_name} • {'CUDA 12.4' if cuda_available else 'CPU Mode'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = render_sidebar(cuda_available=cuda_available, gpu_name=gpu_name)

    # Workflow Tabs
    tab_curated, tab_custom = st.tabs(["Curated Exhibition", "Custom Upload"])

    content_image: Optional[Image.Image] = None
    style_image: Optional[Image.Image] = None

    with tab_curated:
        selected_key = st.radio(
            "Select Preset",
            options=list(PRESETS.keys()),
            format_func=lambda k: f"{PRESETS[k]['title']} — {PRESETS[k]['artist']}",
            horizontal=True,
            label_visibility="collapsed",
        )

        preset = PRESETS[selected_key]

        c_col, s_col = st.columns(2)
        with c_col:
            st.markdown(
                f"""
                <div class="panel-header">Source Content · Photograph</div>
                """,
                unsafe_allow_html=True,
            )
            if preset["content_path"].exists():
                content_image = Image.open(preset["content_path"])
                st.image(content_image, width="stretch")

        with s_col:
            st.markdown(
                f"""
                <div class="panel-header">Artistic Motif · {preset['artist']} ({preset['movement']})</div>
                """,
                unsafe_allow_html=True,
            )
            if preset["style_path"].exists():
                style_image = Image.open(preset["style_path"])
                st.image(style_image, width="stretch")

        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #71717A; margin-top: 0.5rem;">
                {preset['note']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_custom:
        cust_c_col, cust_s_col = st.columns(2)
        with cust_c_col:
            st.markdown(
                """<div class="panel-header">Upload Base Photograph</div>""",
                unsafe_allow_html=True,
            )
            c_file = st.file_uploader(
                "Upload photograph",
                type=["jpg", "jpeg", "png", "webp"],
                key="custom_c_file",
                label_visibility="collapsed",
            )
            if c_file is not None:
                content_image = load_image(c_file)
                st.image(content_image, width="stretch")

        with cust_s_col:
            st.markdown(
                """<div class="panel-header">Upload Style Reference</div>""",
                unsafe_allow_html=True,
            )
            s_file = st.file_uploader(
                "Upload painting",
                type=["jpg", "jpeg", "png", "webp"],
                key="custom_s_file",
                label_visibility="collapsed",
            )
            if s_file is not None:
                style_image = load_image(s_file)
                st.image(style_image, width="stretch")

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # Synthesis Trigger Strip
    if content_image is None or style_image is None:
        st.info("Select or upload both a content photograph and a style painting to begin.")
        return

    action_col_info, action_col_btn = st.columns([3, 1])
    with action_col_info:
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #8E8E93; padding-top: 0.6rem;">
                Ready · <b>{config.image.default_size}×{config.image.default_size}px</b> canvas · 
                <b>{config.optimization.max_iter}</b> steps of <b>{config.optimization.optimizer.upper()}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col_btn:
        run_requested = st.button("Synthesize Artwork", use_container_width=True)

    # Execution State
    if run_requested:
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="panel-header">Optimization in Progress</div>""", unsafe_allow_html=True)

        progress_bar = st.progress(0.0)
        telemetry_slot = st.empty()
        preview_slot = st.empty()

        device = config.get_device()
        target_size = config.image.default_size

        content_tensor = image_to_tensor(
            content_image,
            target_size=target_size,
            preserve_aspect_ratio=True,
            device=device,
        )

        _, _, h, w = content_tensor.shape
        style_resized = style_image.resize((w, h), Image.Resampling.LANCZOS)
        style_tensor = image_to_tensor(
            style_resized,
            target_size=max(h, w),
            preserve_aspect_ratio=False,
            device=device,
        )

        total_steps = config.optimization.max_iter

        def on_step_callback(
            step: int,
            current_tensor: torch.Tensor,
            loss_val: float,
            parts: Dict[str, float],
        ) -> None:
            pct = min(1.0, float(step) / float(total_steps))
            progress_bar.progress(pct)

            telemetry_slot.markdown(
                f"""
                <div class="telemetry-row">
                    <div class="telemetry-cell">
                        <div class="cell-caption">Step</div>
                        <div class="cell-number">{step} / {total_steps}</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="cell-caption">Total Loss</div>
                        <div class="cell-number">{loss_val:.3f}</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="cell-caption">Content Loss</div>
                        <div class="cell-number">{parts['content']:.3f}</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="cell-caption">Style Loss</div>
                        <div class="cell-number">{parts['style']:.3f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            current_pil = tensor_to_image(current_tensor)
            preview_slot.image(current_pil, width="stretch")

        start_time = time.perf_counter()
        try:
            with st.spinner("Executing L-BFGS gradient descent..."):
                output_tensor = run_style_transfer(
                    content_img=content_tensor,
                    style_img=style_tensor,
                    config=config,
                    callback=on_step_callback,
                )

            elapsed = time.perf_counter() - start_time
            out_img = tensor_to_image(output_tensor)

            # Clear transient optimization overlay/counters to maintain clean canvas
            progress_bar.empty()
            telemetry_slot.empty()

            st.session_state.stylized_image = out_img
            st.session_state.last_metrics = {
                "runtime": elapsed,
                "width": out_img.width,
                "height": out_img.height,
                "steps": total_steps,
                "device": gpu_name if cuda_available else "CPU",
                "optimizer": config.optimization.optimizer.upper(),
            }

        except Exception as exc:
            st.error(f"Error during optimization: {exc}")
            return

    # Masterpiece Result Display
    if st.session_state.stylized_image is not None:
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="panel-header">Synthesized Masterpiece</div>""", unsafe_allow_html=True)

        res_col_img, res_col_details = st.columns([3, 2])

        with res_col_img:
            st.image(st.session_state.stylized_image, width="stretch")

        with res_col_details:
            m = st.session_state.last_metrics or {}
            st.markdown(
                f"""
                <div class="studio-panel">
                    <div class="panel-header">Telemetry & Specification</div>
                    <div class="telemetry-row" style="grid-template-columns: repeat(2, 1fr);">
                        <div class="telemetry-cell">
                            <div class="cell-caption">Compute Duration</div>
                            <div class="cell-number">{m.get('runtime', 0.0):.2f}s</div>
                        </div>
                        <div class="telemetry-cell">
                            <div class="cell-caption">Resolution</div>
                            <div class="cell-number">{m.get('width', 256)}×{m.get('height', 256)}</div>
                        </div>
                        <div class="telemetry-cell">
                            <div class="cell-caption">Hardware Device</div>
                            <div class="cell-number" style="font-size: 0.85rem;">{m.get('device', 'CPU')}</div>
                        </div>
                        <div class="telemetry-cell">
                            <div class="cell-caption">Optimizer</div>
                            <div class="cell-number">{m.get('optimizer', 'LBFGS')}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            buf = io.BytesIO()
            st.session_state.stylized_image.save(buf, format="PNG")
            img_data = buf.getvalue()

            st.download_button(
                label="Export Artwork (PNG)",
                data=img_data,
                file_name="nst_atelier_artwork.png",
                mime="image/png",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
