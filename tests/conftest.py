"""Global pytest fixtures and test data generators for Neural Style Transfer tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Tuple

# Ensure repository root and src/ are on sys.path in all CI/CD runners
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
for p in (str(repo_root), str(src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def dummy_rgb_image() -> Image.Image:
    """Generate a simple synthetic 128x128 RGB PIL image for testing."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :64] = [255, 0, 0]      # Red quadrant
    arr[:64, 64:] = [0, 255, 0]      # Green quadrant
    arr[64:, :64] = [0, 0, 255]      # Blue quadrant
    arr[64:, 64:] = [255, 255, 255]  # White quadrant
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_rgba_image() -> Image.Image:
    """Generate a synthetic 128x128 RGBA PIL image with an alpha channel."""
    arr = np.full((128, 128, 4), 200, dtype=np.uint8)
    arr[:, :, 3] = 128  # 50% opacity
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def dummy_grayscale_image() -> Image.Image:
    """Generate a synthetic 128x128 Grayscale PIL image."""
    arr = np.linspace(0, 255, 128 * 128, dtype=np.uint8).reshape((128, 128))
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def dummy_image_buffer(dummy_rgb_image: Image.Image) -> io.BytesIO:
    """Provide an in-memory JPEG byte buffer."""
    buf = io.BytesIO()
    dummy_rgb_image.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def dummy_tensor_256() -> torch.Tensor:
    """Generate a deterministic synthetic image tensor of shape (1, 3, 256, 256)."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)


@pytest.fixture
def dummy_tensor_64() -> torch.Tensor:
    """Generate a fast synthetic image tensor of shape (1, 3, 64, 64) for quick unit tests."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)
