"""Iterative optimization engine for Neural Style Transfer."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from nst.config import NSTConfig
from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss, gram_matrix
from nst.model import FeatureExtractor, build_vgg19_feature_extractor

logger = logging.getLogger(__name__)

CallbackType = Callable[[int, torch.Tensor, float, Dict[str, float]], None]


def initialize_target_image(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    init_mode: str = "content",
) -> torch.Tensor:
    """Initialize the optimization target image tensor.

    Parameters
    ----------
    content_img : torch.Tensor
        Content image tensor (1, 3, H, W).
    style_img : torch.Tensor
        Style image tensor (1, 3, H, W).
    init_mode : str, optional
        Initialization strategy ('content', 'style', 'random'), by default 'content'.

    Returns
    -------
    torch.Tensor
        Target image tensor with requires_grad=True.
    """
    mode = init_mode.lower()
    if mode == "content":
        target = content_img.clone().detach()
    elif mode == "style":
        target = style_img.clone().detach()
    elif mode == "random":
        target = torch.rand_like(content_img)
    else:
        raise ValueError(
            f"Unsupported initial_image mode: '{init_mode}'. Choose from 'content', 'style', 'random'."
        )

    # Set requires_grad to enable gradient descent on pixel values
    return target.requires_grad_(True)


def run_style_transfer(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    config: NSTConfig,
    callback: Optional[CallbackType] = None,
    feature_model: Optional[nn.Sequential] = None,
) -> torch.Tensor:
    """Execute iterative style transfer optimization.

    Parameters
    ----------
    content_img : torch.Tensor
        Content image tensor of shape (1, 3, H, W).
    style_img : torch.Tensor
        Style image tensor of shape (1, 3, H, W).
    config : NSTConfig
        Configuration options specifying loss weights, optimizer, and hardware settings.
    callback : Optional[CallbackType], optional
        Callback invoked at step intervals: callback(step, current_image, total_loss, components).
    feature_model : Optional[nn.Sequential], optional
        Pre-instantiated feature extractor. If None, builds default VGG-19 model.

    Returns
    -------
    torch.Tensor
        Stylized image tensor of shape (1, 3, H, W) clamped to [0.0, 1.0].
    """
    device = config.get_device()
    content_tensor = content_img.to(device)
    style_tensor = style_img.to(device)

    # Build or move feature extractor
    if feature_model is None:
        logger.info("Building VGG-19 feature extractor...")
        model = build_vgg19_feature_extractor(pretrained=True, device=device)
    else:
        model = feature_model.to(device)

    # Identify all required layers
    content_layer_names = config.model.layers.content
    style_layer_names = config.model.layers.style
    all_target_layers = list(set(content_layer_names + style_layer_names))

    extractor = FeatureExtractor(model, target_layers=all_target_layers)

    # Precompute and detach content targets P^l
    with torch.no_grad():
        content_features = extractor.extract_features(content_tensor)
        content_loss_modules: Dict[str, ContentLoss] = {}
        for name in content_layer_names:
            if name in content_features:
                content_loss_modules[name] = ContentLoss(
                    target_feature=content_features[name],
                    weight=config.loss.content_weight,
                    normalization=config.loss.loss_normalization,
                ).to(device)

    # Precompute and detach style target Gram matrices A^l
    with torch.no_grad():
        style_features = extractor.extract_features(style_tensor)
        style_loss_modules: Dict[str, StyleLoss] = {}
        for name in style_layer_names:
            if name in style_features:
                g_target = gram_matrix(style_features[name])
                layer_weight = config.model.layers.style_weights.get(name, 0.2)
                style_loss_modules[name] = StyleLoss(
                    target_gram=g_target,
                    weight=layer_weight * config.loss.style_weight,
                    normalization=config.loss.loss_normalization,
                ).to(device)

    # Total Variation Loss module
    tv_module = TotalVariationLoss(weight=config.loss.tv_weight).to(device)

    # Initialize optimization target tensor X
    input_img = initialize_target_image(
        content_tensor, style_tensor, init_mode=config.image.initial_image
    )

    # Setup Optimizer
    opt_name = config.optimization.optimizer.lower()
    if opt_name == "lbfgs":
        optimizer = optim.LBFGS(
            [input_img],
            lr=config.optimization.lr,
            max_iter=1,  # We control the outer loop
            tolerance_grad=config.optimization.tolerance_grad,
            tolerance_change=config.optimization.tolerance_change,
            history_size=config.optimization.history_size,
        )
    elif opt_name == "adam":
        optimizer = optim.Adam([input_img], lr=config.optimization.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    step = 0
    max_steps = config.optimization.max_iter

    try:
        if opt_name == "lbfgs":
            while step < max_steps:

                def closure() -> torch.Tensor:
                    nonlocal step
                    optimizer.zero_grad()

                    features = extractor.extract_features(input_img)

                    # Compute losses
                    c_loss = torch.tensor(0.0, device=device)
                    for name, mod in content_loss_modules.items():
                        c_loss = c_loss + mod(features[name])

                    s_loss = torch.tensor(0.0, device=device)
                    for name, mod in style_loss_modules.items():
                        s_loss = s_loss + mod(features[name])

                    tv_loss = tv_module(input_img)

                    total_loss = c_loss + s_loss + tv_loss
                    total_loss.backward()

                    step += 1

                    # Trigger periodic callback and logging
                    if callback is not None and (
                        step % config.logging.log_frequency == 0 or step == max_steps
                    ):
                        loss_components = {
                            "content": c_loss.item(),
                            "style": s_loss.item(),
                            "tv": tv_loss.item(),
                        }
                        callback(step, input_img.detach(), total_loss.item(), loss_components)

                    # Defensive GPU cache cleanup
                    if (
                        device.type == "cuda"
                        and config.hardware.empty_cache_frequency > 0
                        and step % config.hardware.empty_cache_frequency == 0
                    ):
                        torch.cuda.empty_cache()

                    return total_loss

                optimizer.step(closure)
                with torch.no_grad():
                    input_img.clamp_(0.0, 1.0)

        elif opt_name == "adam":
            for step in range(1, max_steps + 1):
                optimizer.zero_grad()

                features = extractor.extract_features(input_img)

                c_loss = torch.tensor(0.0, device=device)
                for name, mod in content_loss_modules.items():
                    c_loss = c_loss + mod(features[name])

                s_loss = torch.tensor(0.0, device=device)
                for name, mod in style_loss_modules.items():
                    s_loss = s_loss + mod(features[name])

                tv_loss = tv_module(input_img)

                total_loss = c_loss + s_loss + tv_loss
                total_loss.backward()

                optimizer.step()
                with torch.no_grad():
                    input_img.clamp_(0.0, 1.0)

                if callback is not None and (
                    step % config.logging.log_frequency == 0 or step == max_steps
                ):
                    loss_components = {
                        "content": c_loss.item(),
                        "style": s_loss.item(),
                        "tv": tv_loss.item(),
                    }
                    callback(step, input_img.detach(), total_loss.item(), loss_components)

                if (
                    device.type == "cuda"
                    and config.hardware.empty_cache_frequency > 0
                    and step % config.hardware.empty_cache_frequency == 0
                ):
                    torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as oom_err:
        logger.error("CUDA Out of Memory encountered during style transfer optimization.")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA Out of Memory! Consider lowering image size (e.g. 256) or switching to CPU fallback."
        ) from oom_err

    # Final clamping to strictly enforce [0.0, 1.0] domain
    with torch.no_grad():
        input_img.clamp_(0.0, 1.0)

    return input_img.detach()
