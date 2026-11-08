"""VGG-19 feature extractor construction, layer pruning, and differentiable normalization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Standard ImageNet channel mean and standard deviation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard mapping of VGG-19 feature layer indices to human-readable names
VGG19_LAYER_NAMES: Dict[int, str] = {
    0: "conv_1_1",
    1: "relu_1_1",
    2: "conv_1_2",
    3: "relu_1_2",
    4: "pool_1",
    5: "conv_2_1",
    6: "relu_2_1",
    7: "conv_2_2",
    8: "relu_2_2",
    9: "pool_2",
    10: "conv_3_1",
    11: "relu_3_1",
    12: "conv_3_2",
    13: "relu_3_2",
    14: "conv_3_3",
    15: "relu_3_3",
    16: "conv_3_4",
    17: "relu_3_4",
    18: "pool_3",
    19: "conv_4_1",
    20: "relu_4_1",
    21: "conv_4_2",
    22: "relu_4_2",
    23: "conv_4_3",
    24: "relu_4_3",
    25: "conv_4_4",
    26: "relu_4_4",
    27: "pool_4",
    28: "conv_5_1",
    29: "relu_5_1",
}


class Normalization(nn.Module):
    """Differentiable input normalization module prepended to the feature extractor.

    Normalizes an input image tensor in [0.0, 1.0] to match ImageNet's empirical distribution.
    Registered as persistent buffers so device placement (`.to(device)`) is automatic.

    Parameters
    ----------
    mean : Tuple[float, float, float]
        Per-channel RGB mean.
    std : Tuple[float, float, float]
        Per-channel RGB standard deviation.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        # Reshape to (1, 3, 1, 1) for broad-casting across (B, C, H, W)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply channel normalization.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (B, 3, H, W) with values in [0.0, 1.0].

        Returns
        -------
        torch.Tensor
            Normalized tensor with ImageNet mean 0 and unit variance.
        """
        return (img - self.mean) / self.std


def build_vgg19_feature_extractor(
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    max_layer_index: int = 29,
    use_avg_pool: bool = True,
) -> nn.Sequential:
    """Build a pruned, parameter-frozen VGG-19 feature extraction pipeline.

    Parameters
    ----------
    pretrained : bool, optional
        Whether to load ImageNet pretrained weights, by default True.
        Set to False in tests to avoid downloading weights.
    device : Optional[torch.device], optional
        Target device, by default None (CPU).
    max_layer_index : int, optional
        Highest layer index to retain in vgg19.features, by default 29 (relu_5_1).
    use_avg_pool : bool, optional
        Whether to substitute MaxPool2d with AvgPool2d for smoother gradient propagation,
        by default True.

    Returns
    -------
    nn.Sequential
        A Sequential module comprising a Normalization layer followed by
        truncated VGG-19 feature layers up to `max_layer_index`.
    """
    if device is None:
        device = torch.device("cpu")

    weights = VGG19_Weights.DEFAULT if pretrained else None
    vgg19 = models.vgg19(weights=weights)
    features = vgg19.features

    # Replace MaxPool2d with AvgPool2d and disable inplace ReLUs for gradient fidelity
    for i, layer in enumerate(features):
        if use_avg_pool and isinstance(layer, nn.MaxPool2d):
            features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.ReLU):
            features[i] = nn.ReLU(inplace=False)

    # Slice feature extractor up to max_layer_index (inclusive)
    # Layer index 29 corresponds to relu_5_1
    pruned_features = features[: max_layer_index + 1]

    # Freeze all feature extractor parameters
    for param in pruned_features.parameters():
        param.requires_grad = False

    # Prepend Normalization layer
    norm_layer = Normalization()
    model = nn.Sequential(norm_layer, *pruned_features)

    model.eval()
    return model.to(device)


class FeatureExtractor:
    """Wrapper to run a forward pass and capture feature maps at specified named layers."""

    def __init__(
        self,
        model: nn.Sequential,
        target_layers: List[str],
    ) -> None:
        self.model = model
        self.target_layers = set(target_layers)

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations for all configured target layers in a single forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from layer name (e.g. 'conv_4_2') to feature tensor (B, C, H, W).
        """
        features: Dict[str, torch.Tensor] = {}

        # The first layer is Normalization, subsequent layers correspond to VGG19_LAYER_NAMES
        current_tensor = x
        for i, layer in enumerate(self.model):
            current_tensor = layer(current_tensor)
            if i == 0:
                # Normalization layer
                continue

            vgg_idx = i - 1
            layer_name = VGG19_LAYER_NAMES.get(vgg_idx)
            if layer_name and layer_name in self.target_layers:
                features[layer_name] = current_tensor

        return features
