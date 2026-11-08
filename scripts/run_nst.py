"""Command-line interface (CLI) for running Neural Style Transfer headlessly."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

# Ensure src/ is on sys.path for standalone script execution
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nst.config import NSTConfig
from nst.engine import run_style_transfer
from nst.image_utils import image_to_tensor, load_image, save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nst-cli")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer — AI Art Generator (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Path to content image file (JPEG, PNG, WebP).",
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Path to style reference image file (JPEG, PNG, WebP).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/stylized.png",
        help="Destination path for synthesized artwork.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base YAML configuration file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Target image dimension in pixels (e.g. 256 for 4GB RTX 3050).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum optimization steps (e.g. 300 for L-BFGS).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["lbfgs", "adam"],
        default=None,
        help="Optimization algorithm.",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=None,
        help="Weight alpha for content preservation.",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=None,
        help="Weight beta for artistic texture synthesis.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=None,
        help="Weight gamma for total variation smoothness.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Hardware execution device.",
    )
    return parser.parse_args(args)


def main() -> None:
    """Main execution function for CLI runner."""
    args = parse_args()

    # Load base configuration
    config_path = Path(args.config)
    if config_path.is_file():
        logger.info(f"Loading configuration from '{config_path}'")
        config = NSTConfig.from_yaml(config_path)
    else:
        logger.warning(
            f"Config file '{config_path}' not found. Initializing default configuration."
        )
        config = NSTConfig()

    # Apply CLI overrides if specified
    if args.image_size is not None:
        config.image.default_size = args.image_size
    if args.steps is not None:
        config.optimization.max_iter = args.steps
    if args.optimizer is not None:
        config.optimization.optimizer = args.optimizer
    if args.content_weight is not None:
        config.loss.content_weight = args.content_weight
    if args.style_weight is not None:
        config.loss.style_weight = args.style_weight
    if args.tv_weight is not None:
        config.loss.tv_weight = args.tv_weight
    if args.device is not None:
        config.hardware.device = args.device

    device = config.get_device()
    logger.info(f"Target execution device resolved to: {device}")

    # Load and preprocess input images
    logger.info(f"Loading content image: '{args.content}'")
    content_pil = load_image(args.content)

    logger.info(f"Loading style image: '{args.style}'")
    style_pil = load_image(args.style)

    target_size = config.image.default_size
    preserve_aspect = config.image.preserve_aspect_ratio

    content_tensor = image_to_tensor(
        content_pil,
        target_size=target_size,
        preserve_aspect_ratio=preserve_aspect,
        device=device,
    )

    # Style image is resized to match content dimensions for aligned feature processing
    _, _, h, w = content_tensor.shape
    style_resized_pil = style_pil.resize((w, h))
    style_tensor = image_to_tensor(
        style_resized_pil,
        target_size=max(h, w),
        preserve_aspect_ratio=False,
        device=device,
    )

    logger.info(
        f"Input tensors prepared: Shape={content_tensor.shape}, Range=[{content_tensor.min():.2f}, {content_tensor.max():.2f}]"
    )

    # Progress bar tracking
    pbar = tqdm(total=config.optimization.max_iter, desc="Stylizing Artwork", unit="step")
    last_reported_step = 0

    def progress_callback(step: int, img: torch.Tensor, total_loss: float, components: dict) -> None:
        nonlocal last_reported_step
        delta = step - last_reported_step
        if delta > 0:
            pbar.update(delta)
            last_reported_step = step
            pbar.set_postfix(
                loss=f"{total_loss:.3e}",
                content=f"{components['content']:.2e}",
                style=f"{components['style']:.2e}",
                tv=f"{components['tv']:.2e}",
            )

    start_time = time.perf_counter()

    try:
        output_tensor = run_style_transfer(
            content_img=content_tensor,
            style_img=style_tensor,
            config=config,
            callback=progress_callback,
        )
    finally:
        pbar.close()

    elapsed = time.perf_counter() - start_time
    logger.info(f"Optimization finished in {elapsed:.2f} seconds.")

    out_path = Path(args.output)
    save_image(output_tensor, out_path)
    logger.info(f"Stylized artwork successfully saved to: '{out_path}'")


if __name__ == "__main__":
    main()
