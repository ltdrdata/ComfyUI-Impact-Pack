from types import SimpleNamespace

import pytest
import torch

from test_detailer_working_size import fake_vae, runtime as runtime


def refine(runtime, monkeypatch, ratio, crop_input, width, height):
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
    (16, 821, 493, (832, 496)),
    (16, 808, 488, (800, 480)),
])
def test_patch_returns_at_the_source_scale(runtime, monkeypatch, ratio, width, height, expected):
    _, encoded, shift = refine(runtime, monkeypatch, ratio, True, width, height)

    assert size(encoded) == expected
    assert shift < 0.25


@pytest.mark.parametrize("width,height", [(800, 480), (808, 488)])
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
    _, encoded, _ = refine(runtime, monkeypatch, 16, True, width, height)

    assert size(encoded) == expected


def test_video_vae_gives_one_picture_of_the_size_of_the_input(runtime, monkeypatch):
    from impact import segs_upscaler

    vae = fake_vae(runtime, 8, True)
    decode = vae.decode.side_effect
    vae.decode.side_effect = lambda samples: decode(samples).unsqueeze(1)
    monkeypatch.setattr(segs_upscaler.impact_sampling, "ksampler_wrapper", lambda *args, **kwargs: args[8])
    image = torch.rand(1, 480, 800, 3)

    refined = segs_upscaler.img2img_segs(image, SimpleNamespace(model_options={}), None, vae, 123, 20, 7.0, "euler", "normal", [], [], 0.5, None)

    assert refined.shape == image.shape
