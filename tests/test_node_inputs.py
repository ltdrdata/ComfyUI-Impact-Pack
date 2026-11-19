"""A node leaves the values it was given as they were.

ComfyUI keeps the output of a node in its cache and hands the same object to every node that reads it, in later
prompts too. Run with IMPACT_COMFY_ROOT set.
"""
import inspect
import logging
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
                              FaceDetailer=impact_pack.FaceDetailer, SAMLoader=impact_pack.SAMLoader, impact_pack=impact_pack,
                              impact_server=impact_server)


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


@pytest.mark.parametrize("folder", ["sams", "checkpoints"])
def test_sam_loader_loads_a_file_named_sam3_as_sam3_from_either_folder(pack, monkeypatch, folder):
    files = {"sams": ["sam_vit_b_01ec64.pth"], "checkpoints": ["sd15.safetensors"]}
    files[folder].append("SAM3.1_multiplex_fp16.safetensors")
    folder_paths, loaded = pack.impact_pack.folder_paths, []
    # stands for the model ComfyUI loads, with the functions of its SAM3 that the pack calls
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
        forward_segment=lambda images, point_inputs=None, box_inputs=None: None,
        forward_video=lambda images, initial_masks, pbar=None, target_device=None, target_dtype=None: None)))
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda name: files[name])
    monkeypatch.setattr(folder_paths, "get_full_path", lambda name, file: f"/{name}/{file}" if file in files[name] else None)
    monkeypatch.setattr(pack.impact_pack.comfy.sd, "load_checkpoint_guess_config_model_only", lambda path: loaded.append(path) or model)

    # the file is offered once, and a checkpoint that is not SAM3 is not offered
    assert pack.SAMLoader.INPUT_TYPES()["required"]["model_name"][0] == ["sam_vit_b_01ec64.pth", "SAM3.1_multiplex_fp16.safetensors"]

    sam, = pack.SAMLoader().load_model("SAM3.1_multiplex_fp16.safetensors", "AUTO")
    assert loaded == [f"/{folder}/SAM3.1_multiplex_fp16.safetensors"]
    assert isinstance(sam, pack.impact_pack.core.SAM3Wrapper) and sam.model is model


def test_sam_loader_refuses_a_sam3_model_on_a_comfyui_without_sam3(pack, monkeypatch):
    monkeypatch.delattr(pack.impact_pack.comfy.model_base, "SAM3")
    with pytest.raises(Exception, match="SAM3 cannot be used with the current version of ComfyUI"):
        pack.SAMLoader().load_model("sam3.1_multiplex_fp16.safetensors", "AUTO")


@pytest.mark.parametrize("request_body", [{"1": {"class_type": "ImpactInt", "inputs": {"value": 1}}}, {"number": 1}, {"front": True}, {}],
                         ids=["a prompt as the body itself", "number", "front", "empty"])
def test_a_request_without_a_prompt_passes_the_prompt_handler_as_it_is(pack, caplog, request_body):
    """ComfyUI calls the prompt handlers with every POST /prompt, and refuses one without a prompt afterwards.
    A prompt that is posted as the body itself, not as the value of "prompt", is such a request."""
    given = dict(request_body)

    with caplog.at_level(logging.ERROR):
        assert pack.impact_server.onprompt(request_body) is request_body

    assert request_body == given
    assert caplog.records == []
