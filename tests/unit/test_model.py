"""Unit tests for VGG-19 feature extraction, normalization, and pruning."""

import pytest
import torch
import torch.nn as nn

from nst.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FeatureExtractor,
    Normalization,
    build_vgg19_feature_extractor,
)


def test_normalization_layer():
    """Verify input normalization values and gradient flow."""
    norm = Normalization()
    # Test flat RGB input
    ones_tensor = torch.ones(1, 3, 10, 10, requires_grad=True)
    out = norm(ones_tensor)

    assert out.shape == (1, 3, 10, 10)
    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert pytest.approx(out[0, 0, 0, 0].item(), rel=1e-4) == expected_r

    # Test gradient pass-through
    out.sum().backward()
    assert ones_tensor.grad is not None
    assert torch.isfinite(ones_tensor.grad).all()


def test_build_vgg19_feature_extractor_structure():
    """Verify layer pruning, parameter freezing, and AvgPool substitution."""
    model = build_vgg19_feature_extractor(pretrained=False, use_avg_pool=True)

    # 1 Normalization layer + 30 VGG layers (0 through 29) = 31 layers total
    assert len(model) == 31
    assert isinstance(model[0], Normalization)

    # Verify no MaxPool2d layers remain
    for layer in model:
        assert not isinstance(layer, nn.MaxPool2d), "Found un-replaced MaxPool2d layer"

    # Verify all parameters are frozen
    for param in model.parameters():
        assert not param.requires_grad, "Model parameter was not frozen"


def test_feature_extractor_capture(dummy_tensor_64: torch.Tensor):
    """Verify FeatureExtractor captures correct feature map names and shapes."""
    model = build_vgg19_feature_extractor(pretrained=False, use_avg_pool=True)
    target_layers = ["conv_1_1", "conv_4_2"]
    extractor = FeatureExtractor(model, target_layers=target_layers)

    features = extractor.extract_features(dummy_tensor_64)

    assert "conv_1_1" in features
    assert "conv_4_2" in features

    # conv_1_1 has 64 filters and spatial size 64x64
    assert features["conv_1_1"].shape == (1, 64, 64, 64)
    # conv_4_2 has 512 filters and 3 downsampling stages (64 -> 32 -> 16 -> 8)
    assert features["conv_4_2"].shape == (1, 512, 8, 8)
