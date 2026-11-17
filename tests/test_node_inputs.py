"""A node leaves the values it was given as they were.

ComfyUI keeps the output of a node in its cache and hands the same object to every node that reads it, in later
prompts too. Run with IMPACT_COMFY_ROOT set.
"""
import inspect
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module")
def pack():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import impact_pack, special_samplers, util_nodes
        yield SimpleNamespace(TwoSamplersForMask=special_samplers.TwoSamplersForMask, StringListToString=util_nodes.StringListToString,
                              FaceDetailer=impact_pack.FaceDetailer)


class Sampler:
    """Stands for a KSAMPLER: it returns another latent, as KSamplerWrapper.sample does."""

    def __init__(self, value):
        self.value, self.given = value, []

    def sample(self, latent_image):
        self.given.append(dict(latent_image))
        return {**latent_image, "samples": torch.full_like(latent_image["samples"], self.value)}


@pytest.fixture
def mask():
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 32:] = 1.0
    return mask


@pytest.mark.parametrize("own_mask", [None, torch.full((1, 64, 64), 0.5)], ids=["without a noise mask", "with a noise mask"])
def test_two_samplers_for_mask_leaves_the_latent_it_was_given_as_it_was(pack, mask, own_mask):
    latent = {"samples": torch.zeros(1, 4, 8, 8), "downscale_ratio_spacial": 8}
    if own_mask is not None:
        latent["noise_mask"] = own_mask
    before = dict(latent)

    pack.TwoSamplersForMask.doit(latent, Sampler(1.0), Sampler(2.0), mask)

    assert latent.keys() == before.keys() and all(latent[key] is before[key] for key in before)
    assert torch.count_nonzero(latent["samples"]) == 0


def test_two_samplers_for_mask_gives_each_sampler_its_side_of_the_mask_and_a_result_without_one(pack, mask):
    latent = {"samples": torch.zeros(1, 4, 8, 8), "downscale_ratio_spacial": 8}
    base, masked = Sampler(1.0), Sampler(2.0)

    result = pack.TwoSamplersForMask.doit(latent, base, masked, mask)[0]

    assert torch.equal(base.given[0]["noise_mask"], 1.0 - mask) and torch.count_nonzero(base.given[0]["samples"]) == 0
    assert masked.given[0]["noise_mask"] is mask and torch.all(masked.given[0]["samples"] == 1.0)
    assert "noise_mask" not in result and torch.all(result["samples"] == 2.0) and result["downscale_ratio_spacial"] == 8


@pytest.mark.parametrize("given, joined", [("\\n", "a\nb"), (", ", "a, b")], ids=["backslash n", "another separator"])
def test_string_list_to_string_leaves_the_separator_it_was_given_as_it_was(pack, given, joined):
    join_with = [given]  # a list input is the list the server holds for the output it is linked to

    assert pack.StringListToString().doit(join_with, ["a", "b"]) == (joined,)
    assert join_with == [given]


def test_face_detailer_takes_its_default_prompt_off_the_detector_when_detection_fails(pack):
    class Detector:
        def __init__(self):
            self.aux = []

        def setAux(self, x):
            self.aux.append(x)

        def detect(self, *args, **kwargs):
            raise RuntimeError("interrupted")

    detector = Detector()
    given = dict.fromkeys(inspect.signature(pack.FaceDetailer.enhance_face).parameters)
    given["bbox_detector"] = detector

    with pytest.raises(RuntimeError, match="interrupted"):
        pack.FaceDetailer.enhance_face(**given)

    assert detector.aux == ["face", None]
