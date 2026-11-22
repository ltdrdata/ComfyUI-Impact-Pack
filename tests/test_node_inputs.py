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
        from impact import impact_server  # noqa: F401
        from impact import core, impact_pack, impact_sampling, special_samplers, util_nodes
        yield SimpleNamespace(TwoSamplersForMask=special_samplers.TwoSamplersForMask, StringListToString=util_nodes.StringListToString,
                              FaceDetailer=impact_pack.FaceDetailer, SAMLoader=impact_pack.SAMLoader, impact_pack=impact_pack,
                              impact_server=impact_server, core=core, special_samplers=special_samplers, impact_sampling=impact_sampling)


class Sampler:
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
    join_with = [given]

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
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
        forward_segment=lambda images, point_inputs=None, box_inputs=None: None,
        forward_video=lambda images, initial_masks, pbar=None, target_device=None, target_dtype=None: None)))
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda name: files[name])
    monkeypatch.setattr(folder_paths, "get_full_path", lambda name, file: f"/{name}/{file}" if file in files[name] else None)
    monkeypatch.setattr(pack.impact_pack.comfy.sd, "load_checkpoint_guess_config_model_only", lambda path: loaded.append(path) or model)

    assert pack.SAMLoader.INPUT_TYPES()["required"]["model_name"][0] == ["sam_vit_b_01ec64.pth", "SAM3.1_multiplex_fp16.safetensors"]

    sam, = pack.SAMLoader().load_model("SAM3.1_multiplex_fp16.safetensors", "AUTO")
    assert loaded == [f"/{folder}/SAM3.1_multiplex_fp16.safetensors"]
    assert isinstance(sam, pack.impact_pack.core.SAM3Wrapper) and sam.model is model


def test_sam_loader_refuses_a_sam3_model_on_a_comfyui_without_sam3(pack, monkeypatch):
    monkeypatch.delattr(pack.impact_pack.comfy.model_base, "SAM3")
    with pytest.raises(Exception, match="SAM3 cannot be used with the current version of ComfyUI"):
        pack.SAMLoader().load_model("sam3.1_multiplex_fp16.safetensors", "AUTO")


@pytest.mark.parametrize("request_body", [{"1": {"class_type": "ImpactInt", "inputs": {"value": 1}}}, {}],
                         ids=["a prompt as the body itself", "empty"])
def test_a_request_without_a_prompt_passes_the_prompt_handler_as_it_is(pack, caplog, request_body):
    """ComfyUI calls the prompt handlers with every POST /prompt, and refuses one without a prompt afterwards.
    A prompt that is posted as the body itself, not as the value of "prompt", is such a request."""
    given = dict(request_body)

    with caplog.at_level(logging.ERROR):
        assert pack.impact_server.onprompt(request_body) is request_body

    assert request_body == given
    assert caplog.records == []


class Denoiser:
    def __init__(self):
        import comfy.model_sampling
        sampling = type("Sampling", (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS), {})()
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=sampling), model_patcher=SimpleNamespace(get_model_object=lambda name: sampling))

    def __call__(self, x, sigma, **kwargs):
        denoised = x * 0.5
        for function in kwargs.get("model_options", {}).get("sampler_post_cfg_function", []):
            denoised = function({"denoised": denoised, "uncond_denoised": x * 0.4, "cond": x * 0.5, "uncond": x * 0.6})
        return denoised


def one_step_depends_on_the_seed(sampler):
    def one_step(seed):
        x = torch.linspace(-1, 1, 64).reshape(1, 4, 4, 4)
        return sampler.sampler_function(Denoiser(), x, torch.tensor([5.0, 4.0]), extra_args={"seed": seed}, callback=None, disable=True, **sampler.extra_options)

    return not torch.equal(one_step(1), one_step(2))


def test_a_sampler_of_comfyui_is_told_to_draw_its_noise_anew_in_every_call_exactly_when_a_call_of_one_step_depends_on_the_seed(pack):
    import comfy.samplers
    spread_over_the_schedule = ("dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu")

    told = {name: pack.impact_sampling.KSamplerAdvancedWrapper(None, 7.0, name, "normal", [], []).draws_noise_in_each_call() for name in comfy.samplers.SAMPLER_NAMES}
    draws = {name: name not in spread_over_the_schedule and one_step_depends_on_the_seed(comfy.samplers.sampler_object(name)) for name in comfy.samplers.SAMPLER_NAMES}

    assert told == draws
    assert {"euler_ancestral", "lcm"} <= {name for name in draws if draws[name]}


@pytest.mark.parametrize("name,options", [("euler_ancestral", {"eta": 1.0, "s_noise": 1.0}), ("euler_ancestral", {"eta": 0.0, "s_noise": 1.0}),
                                          ("euler_ancestral", {"eta": 1.0, "s_noise": 0.0}), ("er_sde", {"s_noise": 0.0}), ("er_sde", {"s_noise": 1.0}),
                                          ("res_multistep", {}), ("dpmpp_2m_sde", {}), ("euler", {})])
def test_a_sampler_given_as_an_object_is_told_apart_in_the_same_way(pack, name, options):
    import comfy.samplers
    sampler = comfy.samplers.ksampler(name, options)

    told = pack.impact_sampling.KSamplerAdvancedWrapper(None, 7.0, "euler", "normal", [], [], sampler_opt=sampler).draws_noise_in_each_call()

    assert told is one_step_depends_on_the_seed(sampler)


TRACE_OUTSIDE_OF_THE_MASK = 0.25
ADDED_NOISE = 100.0


def step_noise(seed):
    return torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(seed))


class StepSampler:
    def __init__(self, draws):
        self.draws, self.seeds = draws, []
        self.params = (SimpleNamespace(get_model_object=lambda name: SimpleNamespace(latent_channels=4, latent_dimensions=2)), 7.0, "stand-in", "normal", [], [], 1.0)

    def draws_noise_in_each_call(self):
        return self.draws

    def sample_advanced(self, add_noise, seed, steps, latent_image, start_at_step, end_at_step, return_with_leftover_noise, **kwargs):
        if start_at_step >= steps:
            return latent_image
        self.seeds.append(seed)
        samples = latent_image["samples"]
        inside = torch.ones_like(samples, dtype=torch.bool)
        if "noise_mask" in latent_image:
            import comfy.utils
            inside = comfy.utils.reshape_mask(latent_image["noise_mask"], samples.shape) > 0
        samples = samples + torch.where(inside, float(end_at_step - start_at_step), TRACE_OUTSIDE_OF_THE_MASK)
        if self.draws:
            samples = samples + step_noise(seed)
        if add_noise:
            samples = samples + ADDED_NOISE
        return dict(latent_image, samples=samples)


@pytest.mark.parametrize("node", ["RegionalSampler", "RegionalSamplerAdvanced"])
@pytest.mark.parametrize("draws", [True, False], ids=["a sampler that draws its noise", "a sampler without noise of its own"])
def test_the_regional_samplers_step_a_sampler_once_through_every_area(pack, monkeypatch, node, draws):
    monkeypatch.setattr(pack.core, "update_node_status", lambda *args: None)
    from comfy_extras.nodes_mask import LatentCompositeMasked
    monkeypatch.setitem(pack.core.nodes.NODE_CLASS_MAPPINGS, "LatentCompositeMasked", LatentCompositeMasked)
    base, region = StepSampler(draws), StepSampler(draws)
    mask = torch.zeros(64, 64)
    mask[:, :32] = 1.0
    prompts = [pack.core.REGIONAL_PROMPT(mask, region)]
    start = torch.full((1, 4, 8, 8), 0.5)
    seed, steps = 100, 4

    if node == "RegionalSampler":
        result, = pack.special_samplers.RegionalSampler.doit(seed, 0, "ignore", steps, 1, 1.0, {"samples": start.clone()}, base, prompts, 0, True,
                                                              "DISABLE", "AUTO", 0.3)
        stepped, base_only_steps, added_noise = range(steps), 1, ADDED_NOISE
    else:
        result, = pack.special_samplers.RegionalSamplerAdvanced.doit(False, seed, steps, 0, steps, 0, True, True, {"samples": start.clone()}, base, prompts,
                                                                      "DISABLE", "AUTO", 0.3, unique_id=None)
        closing_step = steps - 1
        stepped, base_only_steps, added_noise = range(closing_step), 0, 0.0

    if draws:
        expected = start + added_noise + len(stepped) + sum(step_noise(seed + step) for step in stepped)
        torch.testing.assert_close(result["samples"], expected, rtol=0, atol=1e-5)
    else:
        expected = start + added_noise + len(stepped)
        expected[..., :4] += TRACE_OUTSIDE_OF_THE_MASK * (len(stepped) - base_only_steps)
        torch.testing.assert_close(result["samples"], expected, rtol=0, atol=0)
        assert set(base.seeds + region.seeds) == {seed}


def test_comfyui_draws_noise_on_the_cpu_from_the_seed_of_any_step_of_the_last_seed(pack):
    from comfy.k_diffusion.sampling import default_noise_sampler
    last_seed = 2**64 - 1

    for step in range(4):
        seed = pack.special_samplers.seed_of_step(StepSampler(True), last_seed - step, step)
        default_noise_sampler(torch.zeros(1), seed=seed)(None, None)


@pytest.mark.parametrize("node", ["RegionalSampler", "RegionalSamplerAdvanced"])
def test_the_noise_that_the_base_sampler_adds_in_a_step_stays_in_the_regions(pack, monkeypatch, node):
    monkeypatch.setattr(pack.core, "update_node_status", lambda *args: None)
    from comfy_extras.nodes_mask import LatentCompositeMasked
    monkeypatch.setitem(pack.core.nodes.NODE_CLASS_MAPPINGS, "LatentCompositeMasked", LatentCompositeMasked)
    mask = torch.zeros(64, 64)
    mask[:, :32] = 1.0
    prompts = [pack.core.REGIONAL_PROMPT(mask, StepSampler(True))]
    start = torch.full((1, 4, 8, 8), 0.5)
    steps = 4

    if node == "RegionalSampler":
        base_only_steps = 0
        result, = pack.special_samplers.RegionalSampler.doit(100, 0, "ignore", steps, base_only_steps, 1.0, {"samples": start.clone()}, StepSampler(True), prompts, 0, True,
                                                              "DISABLE", "AUTO", 0.3)
    else:
        result, = pack.special_samplers.RegionalSamplerAdvanced.doit(True, 100, steps, 0, steps, 0, True, True, {"samples": start.clone()}, StepSampler(True), prompts,
                                                                      "DISABLE", "AUTO", 0.3, unique_id=None)

    in_the_region = (result["samples"] - start)[..., :4]
    assert ADDED_NOISE * 0.5 < in_the_region.min() and in_the_region.max() < ADDED_NOISE * 1.5
