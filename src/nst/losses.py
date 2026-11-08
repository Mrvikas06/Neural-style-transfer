"""Loss functions and Gram matrix computation for Neural Style Transfer."""

from __future__ import annotations

import torch
import torch.nn as nn


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of cross-channel feature correlations.

    Parameters
    ----------
    features : torch.Tensor
        Feature map tensor of shape (B, C, H, W) or (C, H, W).
        B must be 1 for style transfer.

    Returns
    -------
    torch.Tensor
        Symmetric Gram matrix of shape (C, C).
    """
    if features.dim() == 4:
        # Expected shape: (1, C, H, W)
        _, c, h, w = features.shape
        features_flat = features.view(c, h * w)
    elif features.dim() == 3:
        # Shape: (C, H, W)
        c, h, w = features.shape
        features_flat = features.view(c, h * w)
    else:
        raise ValueError(
            f"Expected feature tensor with 3 or 4 dimensions, got shape {features.shape}"
        )

    # Compute inner product of vectorized feature map activations
    return torch.mm(features_flat, features_flat.t())
