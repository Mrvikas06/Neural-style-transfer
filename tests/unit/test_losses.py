"""Unit tests for Gram matrix and loss functions (Content, Style, TV)."""

import pytest
import torch

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss, gram_matrix


def test_gram_matrix_properties():
    """Verify Gram matrix shape, symmetry, and diagonal properties."""
    feat = torch.randn(1, 16, 32, 32)
    gram = gram_matrix(feat)

    assert gram.shape == (16, 16)
    # Check symmetry: G == G^T
    assert torch.allclose(gram, gram.t(), atol=1e-5)
    # Diagonal elements must be non-negative
    diag = torch.diag(gram)
    assert (diag >= 0).all()


def test_gram_matrix_zero():
    """Verify zero tensor yields zero Gram matrix."""
    feat = torch.zeros(1, 8, 16, 16)
    gram = gram_matrix(feat)
    assert torch.all(gram == 0)


def test_content_loss():
    """Verify content loss computation, zero on identical input, and gradient stability."""
    target = torch.randn(1, 64, 16, 16)
    loss_module = ContentLoss(target_feature=target, weight=2.0)

    # Identical input must yield 0 loss
    zero_loss = loss_module(target)
    assert pytest.approx(zero_loss.item(), abs=1e-6) == 0.0

    # Distinct input with gradient tracking
    input_feat = torch.randn(1, 64, 16, 16, requires_grad=True)
    loss = loss_module(input_feat)
    assert loss.item() > 0.0

    loss.backward()
    assert input_feat.grad is not None
    assert torch.isfinite(input_feat.grad).all()


def test_style_loss_gatys_normalization():
    """Verify Gatys normalized style loss and gradient existence."""
    target_feat = torch.randn(1, 32, 16, 16)
    target_gram = gram_matrix(target_feat)
    style_module = StyleLoss(target_gram=target_gram, weight=0.5, normalization="gatys")

    # Identical features must yield 0 style loss
    zero_loss = style_module(target_feat)
    assert pytest.approx(zero_loss.item(), abs=1e-6) == 0.0

    # Distinct features
    input_feat = torch.randn(1, 32, 16, 16, requires_grad=True)
    loss = style_module(input_feat)
    assert loss.item() > 0.0

    loss.backward()
    assert input_feat.grad is not None
    assert torch.isfinite(input_feat.grad).all()


def test_total_variation_loss():
    """Verify TV loss on flat image vs high frequency noise."""
    tv_module = TotalVariationLoss(weight=1.0)

    # Flat image (zero variation)
    flat_img = torch.ones(1, 3, 32, 32, requires_grad=True)
    flat_loss = tv_module(flat_img)
    # Flat image differences are 0, with eps=1e-8 tv_loss ~= 0
    assert flat_loss.item() < 1e-3

    # High frequency noise image
    noise_img = torch.rand(1, 3, 32, 32, requires_grad=True)
    noise_loss = tv_module(noise_img)
    assert noise_loss.item() > 50.0

    # Check gradient flow
    noise_loss.backward()
    assert noise_img.grad is not None
    assert torch.isfinite(noise_img.grad).all()


def test_style_loss_mean_normalization():
    """Verify resolution-invariant mean normalization in StyleLoss."""
    target_feat = torch.randn(1, 16, 16, 16)
    target_gram = gram_matrix(target_feat)
    style_module = StyleLoss(target_gram=target_gram, weight=1.0, normalization="mean")

    zero_loss = style_module(target_feat)
    assert pytest.approx(zero_loss.item(), abs=1e-6) == 0.0

    input_feat = torch.randn(1, 16, 16, 16, requires_grad=True)
    loss = style_module(input_feat)
    assert loss.item() > 0.0
    loss.backward()
    assert input_feat.grad is not None


def test_content_loss_mean_normalization():
    """Verify mean MSE normalization in ContentLoss."""
    target_feat = torch.randn(1, 32, 16, 16)
    content_module = ContentLoss(target_feature=target_feat, weight=1.0, normalization="mean")

    zero_loss = content_module(target_feat)
    assert pytest.approx(zero_loss.item(), abs=1e-6) == 0.0

    input_feat = torch.randn(1, 32, 16, 16, requires_grad=True)
    loss = content_module(input_feat)
    assert loss.item() > 0.0
    loss.backward()
    assert input_feat.grad is not None
