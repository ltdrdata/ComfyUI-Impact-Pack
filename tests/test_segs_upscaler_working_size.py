"""Upscaler (SEGS) samples each crop at a size its VAE keeps whole, so the patch returns at the source's scale.

Uses the real img2img_segs and ComfyUI's own encode crop on CPU. Only sampling and VAE inference are controlled.
IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_segs_upscaler_working_size.py -v
"""
from types import SimpleNamespace

import pytest
import torch

from test_detailer_working_size import fake_vae, runtime as runtime


def refine(runtime, monkeypatch, ratio, crop_input, width, height):
    """Refine a crop holding one soft dot; return the image, the image the VAE was given and the dot's shift in pixels."""
    from impact import segs_upscaler

    vae = fake_vae(runtime, ratio, crop_input)
    monkeypatch.setattr(segs_upscaler.impact_sampling, "ksampler_wrapper", lambda *args, **kwargs: args[8])
    ys, xs = torch.meshgrid(torch.arange(float(height)), torch.arange(float(width)), indexing="ij")
    image = torch.exp(-((xs - width * .8) ** 2 + (ys - height * .75) ** 2) / 1152).reshape(1, height, width, 1).repeat(1, 1, 1, 3)

    def centre(pixels):
        weights = pixels[0, :, :, 0]
        return torch.stack(((weights * xs).sum(), (weights * ys).sum())) / weights.sum()

    refined = segs_upscaler.img2img_segs(image, SimpleNamespace(model_options={}), None, vae, 123, 20, 7.0, "euler", "normal",
                                         [], [], 0.5, None)
    assert refined.shape == image.shape
    encoded = vae.encode.call_args.args[0]
    return image, encoded, (centre(refined) - centre(image)).abs().max().item()


def size(pixels):
    return pixels.shape[2], pixels.shape[1]


@pytest.mark.parametrize("ratio,width,height,expected", [
    (8, 821, 493, (832, 496)),   # enlarged to 834 x 501 for a multiple of 8, which 8 still does not divide
    (16, 821, 493, (832, 496)),
    (16, 808, 488, (800, 480)),  # a multiple of 8 is left as it is, and 16 does not divide it
])
def test_patch_returns_at_the_source_scale(runtime, monkeypatch, ratio, width, height, expected):
    _, encoded, shift = refine(runtime, monkeypatch, ratio, True, width, height)

    assert size(encoded) == expected
    assert shift < 0.25


@pytest.mark.parametrize("width,height", [(800, 480), (808, 488)])  # 16 does not divide the second; the ratio is the VAE's
def test_size_the_vae_keeps_whole_is_not_resized(runtime, monkeypatch, width, height):
    image, encoded, shift = refine(runtime, monkeypatch, 8, True, width, height)

    assert encoded is image
    assert shift < 0.25


def test_vae_that_does_not_crop_keeps_the_enlarged_size(runtime, monkeypatch):
    _, encoded, _ = refine(runtime, monkeypatch, 16, False, 821, 493)

    assert size(encoded) == (834, 501)


def test_comfyui_without_crop_input_is_snapped(runtime, monkeypatch):
    _, encoded, shift = refine(runtime, monkeypatch, 8, None, 821, 493)

    assert size(encoded) == (832, 496)
    assert shift < 0.25


@pytest.mark.parametrize("width,height,expected", [(8, 200, (16, 192)), (200, 8, (192, 16))])
def test_crop_narrower_than_the_ratio_is_still_refined(runtime, monkeypatch, width, height, expected):
    # 8 divides both sides, so the crop is not enlarged; rounding 8 down to a multiple of 16 must not leave nothing.
    _, encoded, _ = refine(runtime, monkeypatch, 16, True, width, height)

    assert size(encoded) == expected
