"""Empirical hardware benchmarking script for Neural Style Transfer.

Measures iteration throughput, latency, and peak memory consumption across CUDA and CPU targets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch

from nst.config import HardwareConfig, ImageConfig, LossConfig, NSTConfig, OptimizationConfig
from nst.engine import run_style_transfer
from nst.model import build_vgg19_feature_extractor


def benchmark_device(
    device_name: str,
    steps: int = 50,
    size: int = 256,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """Execute an empirical benchmark run on the specified hardware device.

    Parameters
    ----------
    device_name : str
        Target device ('cuda', 'cuda:0', or 'cpu').
    steps : int, optional
        Number of optimization iterations to execute, by default 50.
    size : int, optional
        Spatial canvas edge length in pixels, by default 256.
    pretrained : bool, optional
        Whether to load ImageNet pretrained weights, by default True. Set False for offline tests.

    Returns
    -------
    Dict[str, Any]
        Dictionary of measured telemetry metrics.
    """
    device = torch.device(device_name)
    gpu_label = (
        torch.cuda.get_device_name(0)
        if "cuda" in device_name and torch.cuda.is_available()
        else "Universal CPU"
    )
    print(f"\n{'=' * 70}")
    print(f" BENCHMARK: {device_name.upper()} ({gpu_label}) | Canvas: {size}x{size} | Steps: {steps}")
    print(f"{'=' * 70}")

    config = NSTConfig(
        hardware=HardwareConfig(device=device_name),
        image=ImageConfig(default_size=size),
        loss=LossConfig(),
        optimization=OptimizationConfig(max_iter=steps, optimizer="lbfgs"),
    )

    content = torch.rand(1, 3, size, size, device=device)
    style = torch.rand(1, 3, size, size, device=device)

    # Warmup feature extractor
    print("Warming up VGG-19 feature extractor...")
    extractor = build_vgg19_feature_extractor(pretrained=pretrained, device=device)
    with torch.no_grad():
        _ = extractor(content)

    if "cuda" in device_name and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    print(f"Running {steps} L-BFGS optimization steps...")
    start_time = time.perf_counter()

    _ = run_style_transfer(
        content_img=content,
        style_img=style,
        config=config,
        feature_model=extractor,
    )

    if "cuda" in device_name and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    throughput = steps / elapsed
    ms_per_step = (elapsed / steps) * 1000

    peak_vram_mb = 0.0
    if "cuda" in device_name and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print(f"\n[RESULTS] {device_name.upper()}:")
    print(f"  Wall-Clock Time : {elapsed:.2f} s")
    print(f"  Throughput      : {throughput:.2f} it/s")
    print(f"  Latency/Step    : {ms_per_step:.2f} ms/step")
    if peak_vram_mb > 0:
        print(f"  Peak VRAM       : {peak_vram_mb:.1f} MB")

    return {
        "device": device_name,
        "elapsed": elapsed,
        "throughput": throughput,
        "ms_per_step": ms_per_step,
        "peak_vram_mb": peak_vram_mb,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI benchmark options."""
    parser = argparse.ArgumentParser(description="NST Empirical Performance Benchmark")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=50, help="Optimization steps to benchmark")
    parser.add_argument("--size", type=int, default=256, help="Square canvas dimension")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            benchmark_device("cuda", steps=args.steps, size=args.size)
        benchmark_device("cpu", steps=min(args.steps, 10), size=args.size)
    else:
        benchmark_device(args.device, steps=args.steps, size=args.size)
