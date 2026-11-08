"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
import torch

from nst.config import NSTConfig, HardwareConfig, ImageConfig, LossConfig, OptimizationConfig


def test_default_config_instantiation():
    """Verify default values of NSTConfig."""
    config = NSTConfig()
    assert config.image.default_size == 256
    assert config.loss.content_weight == 1.0
    assert config.loss.style_weight == 5.0e4
    assert config.optimization.optimizer == "lbfgs"
    assert config.optimization.max_iter == 100


def test_config_validation_bounds():
    """Verify validation triggers on invalid bounds."""
    with pytest.raises(ValueError, match="image.default_size"):
        NSTConfig(image=ImageConfig(default_size=10))

    with pytest.raises(ValueError, match="content_weight"):
        NSTConfig(loss=LossConfig(content_weight=-1.0))

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        NSTConfig(optimization=OptimizationConfig(optimizer="invalid_opt"))

    with pytest.raises(ValueError, match="optimization.lr"):
        NSTConfig(optimization=OptimizationConfig(lr=-0.1))


def test_load_from_yaml(tmp_path: Path):
    """Verify loading and parsing from YAML file."""
    yaml_content = """
hardware:
  device: "cpu"
image:
  default_size: 128
loss:
  content_weight: 5.0
  style_weight: 5000.0
optimization:
  optimizer: "adam"
  max_iter: 150
  lr: 0.05
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = NSTConfig.from_yaml(yaml_file)
    assert config.hardware.device == "cpu"
    assert config.image.default_size == 128
    assert config.loss.content_weight == 5.0
    assert config.loss.style_weight == 5000.0
    assert config.optimization.optimizer == "adam"
    assert config.optimization.max_iter == 150
    assert config.optimization.lr == 0.05


def test_get_device_resolution():
    """Verify resolution of device based on configuration."""
    config = NSTConfig(hardware=HardwareConfig(device="cpu"))
    device = config.get_device()
    assert device.type == "cpu"
