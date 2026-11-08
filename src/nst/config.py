"""Configuration management and schema validation for Neural Style Transfer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


@dataclass
class HardwareConfig:
    """Hardware and execution device configuration."""
    device: str = "auto"
    cuda_device_id: int = 0
    allow_cpu_fallback: bool = True
    empty_cache_frequency: int = 50


@dataclass
class ImageConfig:
    """Image processing dimensions and initialization."""
    default_size: int = 256
    max_size: int = 512
    preserve_aspect_ratio: bool = True
    initial_image: str = "content"  # "content", "random", or "style"


@dataclass
class LayerConfig:
    """Selected VGG-19 layers and their respective weights."""
    content: List[str] = field(default_factory=lambda: ["relu_4_2"])
    style: List[str] = field(
        default_factory=lambda: [
            "relu_1_1",
            "relu_2_1",
            "relu_3_1",
            "relu_4_1",
            "relu_5_1",
        ]
    )
    style_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "relu_1_1": 1.0,
            "relu_2_1": 0.8,
            "relu_3_1": 0.5,
            "relu_4_1": 0.3,
            "relu_5_1": 0.1,
        }
    )


@dataclass
class ModelConfig:
    """Feature extractor architecture and layer weights."""
    architecture: str = "vgg19"
    weights: str = "DEFAULT"
    requires_grad: bool = False
    layers: LayerConfig = field(default_factory=LayerConfig)


@dataclass
class LossConfig:
    """Loss function weighting and normalization parameters."""
    content_weight: float = 1.0
    style_weight: float = 5.0e4
    tv_weight: float = 1.0e-4
    loss_normalization: str = "mean"


@dataclass
class OptimizationConfig:
    """Optimizer algorithm hyperparameters."""
    optimizer: str = "lbfgs"  # "lbfgs" or "adam"
    max_iter: int = 100
    lr: float = 1.0
    tolerance_grad: float = 1.0e-7
    tolerance_change: float = 1.0e-9
    history_size: int = 50


@dataclass
class LoggingConfig:
    """Logging, callback, and snapshot frequencies."""
    log_frequency: int = 25
    save_intermediate: bool = False
    snapshot_frequency: int = 100


@dataclass
class NSTConfig:
    """Root configuration object uniting all sub-configurations."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.image.default_size < 32 or self.image.default_size > 2048:
            raise ValueError(
                f"image.default_size must be in range [32, 2048], got {self.image.default_size}"
            )
        if self.loss.content_weight < 0:
            raise ValueError("loss.content_weight cannot be negative")
        if self.loss.style_weight < 0:
            raise ValueError("loss.style_weight cannot be negative")
        if self.loss.tv_weight < 0:
            raise ValueError("loss.tv_weight cannot be negative")
        if self.optimization.optimizer.lower() not in ("lbfgs", "adam"):
            raise ValueError(
                f"Unsupported optimizer: '{self.optimization.optimizer}'. Must be 'lbfgs' or 'adam'"
            )
        if self.optimization.lr <= 0:
            raise ValueError("optimization.lr must be positive")
        if self.optimization.max_iter < 1:
            raise ValueError("optimization.max_iter must be at least 1")

    def get_device(self) -> torch.device:
        """Resolve target torch.device considering hardware and environment."""
        env_device = os.environ.get("NST_DEVICE", self.hardware.device).lower()
        if env_device == "cuda":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.hardware.cuda_device_id}")
            if self.hardware.allow_cpu_fallback:
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        if env_device == "cpu":
            return torch.device("cpu")
        # "auto" mode
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_yaml(cls, path: str | Path) -> NSTConfig:
        """Instantiate NSTConfig by reading and parsing a YAML configuration file."""
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path_obj}")

        with open(path_obj, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        hardware = HardwareConfig(**data.get("hardware", {}))
        image = ImageConfig(**data.get("image", {}))

        model_dict = data.get("model", {})
        layers_dict = model_dict.pop("layers", {}) if "layers" in model_dict else {}
        layers = LayerConfig(**layers_dict)
        model = ModelConfig(layers=layers, **model_dict)

        loss = LossConfig(**data.get("loss", {}))
        optimization = OptimizationConfig(**data.get("optimization", {}))
        logging = LoggingConfig(**data.get("logging", {}))

        return cls(
            hardware=hardware,
            image=image,
            model=model,
            loss=loss,
            optimization=optimization,
            logging=logging,
        )
