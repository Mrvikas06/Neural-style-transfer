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


class ContentLoss(nn.Module):
    """Content loss measuring semantic deviation from the content target representation.

    Parameters
    ----------
    target_feature : torch.Tensor
        Precomputed target feature map of shape (1, C, H, W).
    weight : float, optional
        Loss weighting coefficient (alpha), by default 1.0.
    normalization : str, optional
        Loss normalization ('mean' for MSE or 'gatys' for unnormalized sum), by default 'mean'.
    """

    def __init__(
        self,
        target_feature: torch.Tensor,
        weight: float = 1.0,
        normalization: str = "mean",
    ) -> None:
        super().__init__()
        # Detach target representation so its computational graph is not retained
        self.register_buffer("target", target_feature.detach().clone())
        self.weight = float(weight)
        self.normalization = normalization.lower()

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """Compute weighted mean squared error between generated and target features.

        Parameters
        ----------
        input_feature : torch.Tensor
            Generated feature map of shape (1, C, H, W).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        diff = input_feature - self.target
        if self.normalization == "gatys":
            loss = 0.5 * torch.sum(diff * diff)
        else:
            loss = torch.mean(diff * diff)
        return self.weight * loss


class StyleLoss(nn.Module):
    """Style loss measuring texture divergence via Gram matrix difference.

    Parameters
    ----------
    target_gram : torch.Tensor
        Precomputed target Gram matrix of shape (C, C).
    weight : float, optional
        Layer-specific weighting coefficient (w_l), by default 0.2.
    normalization : str, optional
        Normalization convention ('mean' for resolution-invariant MSE, or 'gatys'
        for 1 / (4 * N^2 * M^2)), by default 'mean'.
    """

    def __init__(
        self,
        target_gram: torch.Tensor,
        weight: float = 0.2,
        normalization: str = "mean",
    ) -> None:
        super().__init__()
        # Detach target Gram matrix
        self.register_buffer("target_gram", target_gram.detach().clone())
        self.weight = float(weight)
        self.normalization = normalization.lower()

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """Compute normalized style error.

        Parameters
        ----------
        input_feature : torch.Tensor
            Generated feature map of shape (1, C, H, W).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        g_generated = gram_matrix(input_feature)
        diff = g_generated - self.target_gram
        _, c, h, w = input_feature.shape

        if self.normalization == "gatys":
            sq_diff = torch.sum(diff * diff)
            denom = 4.0 * (c ** 2) * ((h * w) ** 2)
            loss = sq_diff / denom
        else:
            # Resolution-invariant normalized Gram MSE
            diff_norm = diff / float(h * w)
            loss = torch.mean(diff_norm * diff_norm)

        return self.weight * loss


class TotalVariationLoss(nn.Module):
    """Isotropic Total Variation regularization loss to encourage spatial smoothness.

    Parameters
    ----------
    weight : float, optional
        Loss weighting coefficient (gamma), by default 1.0e-4.
    """

    def __init__(self, weight: float = 1.0e-4) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate spatial total variation loss.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape (B, 3, H, W) in [0.0, 1.0].

        Returns
        -------
        torch.Tensor
            Scalar regularizer loss.
        """
        if self.weight <= 0.0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Finite forward differences in vertical and horizontal directions
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2)
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2)

        # Align dimensions to compute isotropic variation
        aligned_h = diff_h[:, :, :, :-1]
        aligned_w = diff_w[:, :, :-1, :]

        # Add 1e-8 for numerical stability (prevent derivative of sqrt(0))
        # Subtract sqrt(1e-8) = 1e-4 so flat images have zero base loss
        tv = torch.sum(torch.sqrt(aligned_h + aligned_w + 1e-8) - 1e-4)
        return self.weight * tv
