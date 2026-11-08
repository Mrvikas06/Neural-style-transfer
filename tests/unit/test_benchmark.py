"""Unit tests for the hardware benchmark script."""

from __future__ import annotations

import torch

from scripts.benchmark import benchmark_device


def test_benchmark_device_cpu_smoke() -> None:
    """Verify that benchmark_device executes on CPU with valid telemetry output."""
    results = benchmark_device("cpu", steps=2, size=64, pretrained=False)
    assert results["device"] == "cpu"
    assert results["elapsed"] > 0.0
    assert results["throughput"] > 0.0
    assert results["ms_per_step"] > 0.0
    assert "peak_vram_mb" in results
