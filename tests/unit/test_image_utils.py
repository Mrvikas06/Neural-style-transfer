"""Unit tests for image loading, transformations, and tensor conversion."""

import io
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from nst.image_utils import (
    calculate_aspect_ratio_size,
    image_to_tensor,
    load_image,
    save_image,
    tensor_to_image,
)


def test_load_image_from_buffer(dummy_image_buffer: io.BytesIO):
    """Verify loading image from in-memory binary stream."""
    img = load_image(dummy_image_buffer)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (128, 128)


def test_load_image_from_file(tmp_path: Path, dummy_rgb_image: Image.Image):
    """Verify loading image from filesystem."""
    file_path = tmp_path / "test.png"
    dummy_rgb_image.save(file_path)

    loaded = load_image(file_path)
    assert loaded.mode == "RGB"
    assert loaded.size == (128, 128)


def test_load_image_nonexistent_file():
    """Verify error raised for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_path_to_image.jpg")


def test_calculate_aspect_ratio_size():
    """Verify aspect-ratio preserving dimension calculations."""
    # Landscape
    w, h = calculate_aspect_ratio_size(1000, 500, max_dimension=256)
    assert w == 256
    assert h == 128

    # Portrait
    w, h = calculate_aspect_ratio_size(400, 800, max_dimension=512)
    assert w == 256
    assert h == 512

    # Smaller than max
    w, h = calculate_aspect_ratio_size(120, 90, max_dimension=256)
    assert (w, h) == (120, 90)


def test_image_to_tensor_conversions(dummy_rgba_image: Image.Image):
    """Verify PIL to Tensor conversions, shapes, and value ranges."""
    tensor = image_to_tensor(dummy_rgba_image, target_size=64, preserve_aspect_ratio=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_tensor_to_image_and_roundtrip(dummy_tensor_64: torch.Tensor):
    """Verify converting tensor to PIL Image and back preserves fidelity."""
    pil_img = tensor_to_image(dummy_tensor_64)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (64, 64)

    # Convert back to tensor
    reconstructed = image_to_tensor(pil_img, target_size=64, preserve_aspect_ratio=False)
    # Check max deviation is within uint8 quantization error: 1/255 ~= 0.0039
    assert torch.max(torch.abs(dummy_tensor_64 - reconstructed)).item() < 0.01


def test_save_image(tmp_path: Path, dummy_tensor_64: torch.Tensor):
    """Verify saving image tensor to filesystem."""
    out_file = tmp_path / "subdir" / "output.png"
    result_path = save_image(dummy_tensor_64, out_file)

    assert result_path.is_file()
    assert result_path.stat().st_size > 0
