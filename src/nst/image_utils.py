"""Image loading, format conversion, resizing, and PyTorch tensor transformations."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
import torch
from PIL import Image


def load_image(source: Union[str, Path, BinaryIO]) -> Image.Image:
    """Load an image from a filesystem path or in-memory binary stream and convert to RGB.

    Parameters
    ----------
    source : Union[str, Path, BinaryIO]
        Filesystem path or file-like binary stream containing image data.

    Returns
    -------
    Image.Image
        A PIL Image in standard 3-channel 'RGB' mode.

    Raises
    ------
    FileNotFoundError
        If a file path is provided that does not exist.
    ValueError
        If the file cannot be decoded as an image.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image at '{path}': {e}") from e
    else:
        try:
            with Image.open(source) as img:
                return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image from binary stream: {e}") from e


def calculate_aspect_ratio_size(
    width: int, height: int, max_dimension: int
) -> Tuple[int, int]:
    """Calculate new (width, height) preserving aspect ratio capped by max_dimension.

    Parameters
    ----------
    width : int
        Original pixel width.
    height : int
        Original pixel height.
    max_dimension : int
        Upper bound on the longest dimension.

    Returns
    -------
    Tuple[int, int]
        New (width, height) satisfying max(width, height) == max_dimension.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive integers, got ({width}, {height})")

    if max(width, height) <= max_dimension:
        return (width, height)

    ratio = max_dimension / float(max(width, height))
    new_width = max(32, int(round(width * ratio)))
    new_height = max(32, int(round(height * ratio)))
    return (new_width, new_height)


def image_to_tensor(
    image: Image.Image,
    target_size: int = 256,
    preserve_aspect_ratio: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert a PIL Image to a 4D float tensor in [0.0, 1.0].

    Parameters
    ----------
    image : Image.Image
        PIL Image in RGB mode.
    target_size : int, optional
        Target dimension for resizing, by default 256.
    preserve_aspect_ratio : bool, optional
        If True, scales preserving aspect ratio so max(H, W) == target_size.
        If False, resizes squarely to (target_size, target_size).
    device : torch.device, optional
        Destination PyTorch device, by default torch.device('cpu').

    Returns
    -------
    torch.Tensor
        4D Float tensor with shape (1, 3, H, W) and values in [0.0, 1.0].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size

    if preserve_aspect_ratio:
        new_w, new_h = calculate_aspect_ratio_size(orig_w, orig_h, target_size)
    else:
        new_w, new_h = (target_size, target_size)

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Convert PIL Image to float32 numpy array in [0.0, 1.0]
    np_img = np.array(resized_img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)

    # Reorder channels from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)

    # Add batch dimension: (1, 3, H, W)
    tensor = tensor.unsqueeze(0).contiguous()

    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a 4D or 3D image tensor to a PIL Image in RGB format.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape (1, 3, H, W) or (3, H, W) in [0.0, 1.0].

    Returns
    -------
    Image.Image
        PIL Image in RGB mode with uint8 pixel values [0, 255].
    """
    # Clone and detach to prevent modifying the computational graph
    img_tensor = tensor.detach().cpu().clone()

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # Shape: (3, H, W)

    if img_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 color channels, got {img_tensor.shape[0]}")

    # Clamp values to valid [0.0, 1.0] range
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    # Convert to (H, W, 3) and uint8 in [0, 255]
    np_img = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    return Image.fromarray(np_img, mode="RGB")


def save_image(
    image: Union[Image.Image, torch.Tensor], destination: Union[str, Path]
) -> Path:
    """Save a PIL Image or image tensor to disk.

    Parameters
    ----------
    image : Union[Image.Image, torch.Tensor]
        Image object or tensor to save.
    destination : Union[str, Path]
        Target file path.

    Returns
    -------
    Path
        Resolved destination path.
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, torch.Tensor):
        pil_img = tensor_to_image(image)
    else:
        pil_img = image

    pil_img.save(dest_path)
    return dest_path
