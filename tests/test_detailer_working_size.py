"""The Detailer samples at a size its VAE keeps whole, so the patch returns at the source's scale.

Uses the real Impact Detailer and ComfyUI's own encode crop on CPU. Only sampling and VAE inference are controlled.
IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_detailer_working_size.py -v
"""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
            send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the core import cycle
        from impact import core
        import comfy.sd
        assert Path(core.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
        yield SimpleNamespace(core=core, crop=comfy.sd.VAE.vae_encode_crop_pixels)


def fake_vae(runtime, ratio, crop_input):
    """A VAE that pools and interpolates by `ratio`. crop_input None stands for ComfyUI before 0.3.65: no attribute, always crops."""
    shared = dict(spacial_compression_encode=lambda: ratio, output_channels=3, pad_channel_value=None)
    cropper = SimpleNamespace(crop_input=crop_input is not False, **shared)

    def encode(pixels):
        return torch.nn.functional.avg_pool2d(runtime.crop(cropper, pixels).movedim(-1, 1), ratio)

    def decode(samples):
        return torch.nn.functional.interpolate(samples, scale_factor=ratio, mode="bilinear").movedim(1, -1)

    vae = SimpleNamespace(encode=Mock(side_effect=encode), decode=Mock(side_effect=decode), **shared)
    if crop_input is not None:
        vae.crop_input = crop_input
    return vae


def detail(runtime, monkeypatch, ratio, crop_input, guide_size, hook=None):
    """Detail a 200 x 120 crop holding one soft dot; return the size the VAE was given and the dot's shift in pixels."""
    vae = fake_vae(runtime, ratio, crop_input)
    monkeypatch.setattr(runtime.core.impact_sampling, "ksampler_wrapper", lambda *args, **kwargs: args[8])
    ys, xs = torch.meshgrid(torch.arange(120.), torch.arange(200.), indexing="ij")
    image = torch.exp(-((xs - 170) ** 2 + (ys - 95) ** 2) / 72).reshape(1, 120, 200, 1).repeat(1, 1, 1, 3)

    def centre(pixels):
        weights = pixels[0, :, :, 0]
        return torch.stack(((weights * xs).sum(), (weights * ys).sum())) / weights.sum()

    refined, _ = runtime.core.enhance_detail(
        image, SimpleNamespace(model_options={}), None, vae, guide_size, True, 4096, (0, 0, 200, 120),
        123, 20, 7.0, "euler", "normal", [], [], 0.5, None, True, detailer_hook=hook)
    assert refined.shape == image.shape
    encoded = vae.encode.call_args.args[0].shape
    return (encoded[2], encoded[1]), (centre(refined) - centre(image)).abs().max().item()


@pytest.mark.parametrize("ratio", [8, 16])
def test_patch_returns_at_the_source_scale(runtime, monkeypatch, ratio):
    # guide_size 315 on the 120 px side asks for 525 x 315, which neither ratio divides.
    size, shift = detail(runtime, monkeypatch, ratio, True, 315)

    assert size == (525 // ratio * ratio, 315 // ratio * ratio)
    assert shift < 0.25


def test_size_the_vae_keeps_whole_is_unchanged(runtime, monkeypatch):
    size, shift = detail(runtime, monkeypatch, 16, True, 192)

    assert size == (320, 192)
    assert shift < 0.25


def test_vae_that_does_not_crop_keeps_the_requested_size(runtime, monkeypatch):
    size, _ = detail(runtime, monkeypatch, 16, False, 315)

    assert size == (525, 315)


def test_comfyui_without_crop_input_is_snapped(runtime, monkeypatch):
    size, shift = detail(runtime, monkeypatch, 8, None, 315)

    assert size == (520, 312)
    assert shift < 0.25


@pytest.mark.parametrize("ratio,width,height", [(8, 6, 40), (16, 12, 200), (8, 7, 7)])
def test_crop_narrower_than_the_ratio_is_still_detailed(runtime, monkeypatch, ratio, width, height):
    # force_inpaint keeps the crop's own size; a side below the ratio must not round down to nothing.
    vae = fake_vae(runtime, ratio, True)
    monkeypatch.setattr(runtime.core.impact_sampling, "ksampler_wrapper", lambda *args, **kwargs: args[8])

    refined, _ = runtime.core.enhance_detail(
        torch.rand(1, height, width, 3), SimpleNamespace(model_options={}), None, vae, 4, True, 4096, (0, 0, width, height),
        123, 20, 7.0, "euler", "normal", [], [], 0.5, None, True)

    encoded = vae.encode.call_args.args[0].shape
    assert (encoded[2], encoded[1]) == (max(ratio, width // ratio * ratio), max(ratio, height // ratio * ratio))
    assert refined.shape == (1, height, width, 3)


def test_size_chosen_by_a_hook_is_snapped_too(runtime, monkeypatch):
    from impact.hooks import DetailerHook

    class FixedSize(DetailerHook):
        def touch_scaled_size(self, w, h):
            return 205, 123

    size, shift = detail(runtime, monkeypatch, 8, True, 315, FixedSize())

    assert size == (200, 120)
    assert shift < 0.25
