"""Run with: python -m pytest tests/test_mask_blur.py

Use real torch/torchvision operations; substitute only ComfyUI dependencies.
"""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
import torch
import torchvision

from impact import config


@pytest.fixture
def utils(monkeypatch):
    for name in ("folder_paths", "nodes"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(get_torch_device=Mock())
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    # wildcards/conftest.py stubs impact.utils; load the real module under another name.
    spec = importlib.util.spec_from_file_location(
        "impact.blur_test_utils", Path(__file__).resolve().parents[1] / "modules/impact/utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(config, "cached_config", {"gpu_gaussian_blur": False})
    return module


@pytest.fixture
def blur_devices(monkeypatch):
    devices = []
    forward = torchvision.transforms.GaussianBlur.forward

    def record(self, mask):
        devices.append(mask.device.type)
        return forward(self, mask)

    monkeypatch.setattr(torchvision.transforms.GaussianBlur, "forward", record)
    return devices


@pytest.mark.parametrize("layout", ["numpy", "HW", "NHW", "NHWC"])
def test_cpu_blur_does_not_select_gpu(utils, blur_devices, layout):
    utils.comfy.model_management.get_torch_device.side_effect = AssertionError("GPU was requested")
    image = torch.linspace(0, 1, 48 * 64).reshape(48, 64)
    mask = {"numpy": image.numpy(), "HW": image, "NHW": image[None], "NHWC": image[None, ..., None]}[layout]
    original = image.clone()

    result = utils.tensor_gaussian_blur_mask(mask, 5)

    assert blur_devices == ["cpu"]
    expected = torchvision.transforms.functional.gaussian_blur(image[None, None], 11, 10.0)
    torch.testing.assert_close(result, expected[:, 0, ..., None], rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    assert result.device.type == "cpu"
    assert result.dtype == image.dtype


@pytest.mark.parametrize("gpu_enabled", [False, True])
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA or ROCm GPU")
def test_blur_device_and_return_device(utils, blur_devices, gpu_enabled, input_device):
    config.cached_config["gpu_gaussian_blur"] = gpu_enabled
    select_device = utils.comfy.model_management.get_torch_device
    select_device.return_value = torch.device("cuda:0")
    mask = torch.linspace(0, 1, 2 * 48 * 64).reshape(2, 48, 64).to(input_device)
    original = mask.clone()

    result = utils.tensor_gaussian_blur_mask(mask, 5)

    assert blur_devices == ["cuda" if gpu_enabled else "cpu"]
    assert select_device.call_count == int(gpu_enabled)
    assert result.device == mask.device
    assert result.dtype == mask.dtype
    expected = torchvision.transforms.functional.gaussian_blur(mask.cpu()[:, None], 11, 10.0)
    torch.testing.assert_close(result.cpu(), expected[:, 0, ..., None], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mask, original, rtol=0, atol=0)


def test_gpu_option_respects_comfy_cpu_mode(utils, blur_devices):
    config.cached_config["gpu_gaussian_blur"] = True
    utils.comfy.model_management.get_torch_device.return_value = torch.device("cpu")
    result = utils.tensor_gaussian_blur_mask(torch.ones(48, 64), 5)
    assert blur_devices == ["cpu"]
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.ones(1, 48, 64, 1))
