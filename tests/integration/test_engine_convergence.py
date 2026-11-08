"""Integration tests verifying end-to-end optimization convergence on synthetic tensors."""

import pytest
import torch

from nst.config import ImageConfig, LoggingConfig, NSTConfig, OptimizationConfig
from nst.engine import run_style_transfer
from nst.model import build_vgg19_feature_extractor


def test_engine_convergence_lbfgs():
    """Verify L-BFGS optimization loop executes, converges, and triggers callbacks."""
    torch.manual_seed(42)
    # Fast synthetic test on 32x32 images
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)

    # Use fast uninitialized model to avoid downloading weights
    feature_model = build_vgg19_feature_extractor(pretrained=False, device=torch.device("cpu"))

    config = NSTConfig(
        optimization=OptimizationConfig(optimizer="lbfgs", max_iter=5, lr=1.0),
        logging=LoggingConfig(log_frequency=1),
    )

    losses_recorded = []

    def callback(step, img_tensor, total_loss, components):
        losses_recorded.append(total_loss)
        assert img_tensor.shape == (1, 3, 32, 32)
        assert "content" in components
        assert "style" in components

    out = run_style_transfer(
        content_img=content,
        style_img=style,
        config=config,
        callback=callback,
        feature_model=feature_model,
    )

    assert out.shape == (1, 3, 32, 32)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert len(losses_recorded) > 0
    # Final loss should be lower than or equal to initial loss
    assert losses_recorded[-1] <= losses_recorded[0]


def test_engine_convergence_adam():
    """Verify Adam optimization loop executes, converges, and respects clamping."""
    torch.manual_seed(42)
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)

    feature_model = build_vgg19_feature_extractor(pretrained=False, device=torch.device("cpu"))

    config = NSTConfig(
        image=ImageConfig(initial_image="random"),
        optimization=OptimizationConfig(optimizer="adam", max_iter=15, lr=0.02),
        logging=LoggingConfig(log_frequency=1),
    )

    losses_recorded = []

    def callback(step, img_tensor, total_loss, components):
        losses_recorded.append(total_loss)

    out = run_style_transfer(
        content_img=content,
        style_img=style,
        config=config,
        callback=callback,
        feature_model=feature_model,
    )

    assert out.shape == (1, 3, 32, 32)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert len(losses_recorded) == 15
    assert losses_recorded[-1] < losses_recorded[0]
