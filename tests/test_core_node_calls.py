"""The pack's calls into ComfyUI's own nodes, against the node classes of the ComfyUI under test.

ComfyUI converts its nodes to the V3 schema one file at a time, and a converted node runs in `execute`; the method
it had before stays for a while, as an alias, or is gone. Run with IMPACT_COMFY_ROOT set.
"""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

EXTRA_SCHEDULERS = ["AYS SDXL", "AYS SD1", "AYS SVD", "GITS[coeff=1.2]", "LTXV[default]", "OSS FLUX", "OSS Wan", "OSS Chroma"]


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id=None), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import core, impact_sampling, special_samplers
        from comfy_extras import nodes_align_your_steps, nodes_gits, nodes_lt, nodes_mask, nodes_optimalsteps
        # ComfyUI registers these when it starts; a test session does not start it.
        for node in (nodes_align_your_steps.AlignYourStepsScheduler, nodes_gits.GITSScheduler, nodes_lt.LTXVScheduler,
                     nodes_optimalsteps.OptimalStepsScheduler, nodes_mask.LatentCompositeMasked):
            patch.setitem(core.nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
        yield SimpleNamespace(sampling=impact_sampling, samplers=special_samplers, core=core, nodes=core.nodes, gits=nodes_gits.GITSScheduler)


@pytest.mark.parametrize("scheduler", EXTRA_SCHEDULERS)
def test_a_scheduler_the_pack_adds_gives_a_noise_schedule(runtime, scheduler):
    sigmas = runtime.sampling.calculate_sigmas(None, "euler", scheduler, 10)
    assert sigmas.ndim == 1 and len(sigmas) > 2
    assert sigmas[-1] == 0 and torch.all(sigmas[:-1] > sigmas[1:])


def test_the_gits_scheduler_func_gives_the_schedule_of_the_core_node(runtime):
    func = runtime.samplers.GITSSchedulerFuncProvider().doit(1.2, 0.8)[0]
    node = runtime.gits()
    schedule = node.execute if hasattr(node, "execute") else node.get_sigmas
    assert torch.equal(func(None, None, 10), schedule(1.2, 10, 0.8)[0])


@pytest.mark.parametrize("scheduler, node_id, given", [
    ("AYS SDXL", "AlignYourStepsScheduler", (("SDXL", 10), {"denoise": 1.0})),
    ("GITS[coeff=1.2]", "GITSScheduler", ((1.2, 10), {"denoise": 1.0})),
    ("LTXV[default]", "LTXVScheduler", ((20, 2.05, 0.95, True, 0.1), {})),
    ("OSS FLUX", "OptimalStepsScheduler", (("FLUX", 10), {"denoise": 1.0})),
])
def test_a_schedule_is_read_from_a_core_node_from_before_its_conversion(runtime, monkeypatch, scheduler, node_id, given):
    class Scheduler:
        def get_sigmas(self, *args, **kwargs):
            assert (args, kwargs) == given
            return (torch.tensor([2.0, 1.0, 0.0]),)

    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, node_id, Scheduler)
    assert runtime.sampling.calculate_sigmas(None, "euler", scheduler, 10).tolist() == [2.0, 1.0, 0.0]


def test_the_gits_scheduler_func_reads_a_core_node_from_before_its_conversion(runtime, monkeypatch):
    class GITSScheduler:
        def get_sigmas(self, coeff, steps, denoise):
            return (torch.tensor([coeff, steps, denoise]),)

    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "GITSScheduler", GITSScheduler)
    assert runtime.samplers.GITSSchedulerFuncProvider().doit(1.25, 0.5)[0](None, None, 10).tolist() == [1.25, 10.0, 0.5]


def test_a_latent_is_composited_through_the_mask(runtime):
    destination, source = {"samples": torch.zeros(1, 4, 8, 8)}, {"samples": torch.ones(1, 4, 8, 8)}
    mask = torch.zeros(64, 64)
    mask[:, 32:] = 1.0
    result = runtime.sampling.composite_masked(destination, source, mask)["samples"]
    assert torch.all(result[..., :4] == 0) and torch.all(result[..., 4:] == 1)
    assert torch.all(destination["samples"] == 0)


def test_a_latent_is_composited_by_a_core_node_from_before_its_conversion(runtime, monkeypatch):
    class LatentCompositeMasked:
        def composite(self, destination, source, x, y, resize_source, mask=None):
            return ((destination, source, x, y, resize_source, mask),)

    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "LatentCompositeMasked", LatentCompositeMasked)
    assert runtime.sampling.composite_masked("destination", "source", "mask") == ("destination", "source", 0, 0, False, "mask")


@pytest.fixture
def half_masked(runtime, monkeypatch):
    """A latent of zeros masked on its right half, and a sampler that turns every latent it is given into ones."""
    mask = torch.zeros(64, 64)
    mask[:, 32:] = 1.0
    given = []

    def sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, *args, **kwargs):
        given.append(latent_image["samples"].clone())
        return {**latent_image, "samples": torch.ones_like(latent_image["samples"])}

    monkeypatch.setattr(runtime.sampling, "separated_sample", sample)
    return SimpleNamespace(latent={"samples": torch.zeros(1, 4, 8, 8), "noise_mask": mask}, given=given)


def sampled_only_under_the_mask(samples):
    return torch.all(samples[..., :4] == 0) and torch.all(samples[..., 4:] == 1)


def test_the_recovery_step_of_an_sde_sampler_starts_from_a_latent_sampled_only_under_the_mask(runtime, half_masked):
    wrapper = runtime.sampling.KSamplerAdvancedWrapper("model", 7.0, "dpmpp_sde", "simple", "positive", "negative")
    wrapper.sample_advanced(True, 0, 4, half_masked.latent, 0, 4, False)
    first, recovery = half_masked.given
    assert torch.all(first == 0) and sampled_only_under_the_mask(recovery)


def test_the_refiner_starts_from_a_latent_sampled_only_under_the_mask(runtime, half_masked):
    runtime.sampling.ksampler_wrapper("model", 0, 4, 7.0, "euler", "simple", "positive", "negative", half_masked.latent, 1.0, refiner_ratio=0.5,
                                      refiner_model="refiner", refiner_clip="clip", refiner_positive="positive", refiner_negative="negative")
    first, refiner = half_masked.given
    assert torch.all(first == 0) and sampled_only_under_the_mask(refiner)


class Fill:
    """A sampler that turns every latent it is given into one value."""
    params = ("model",)

    def __init__(self, value):
        self.value = value

    def sample_advanced(self, add_noise, seed, steps, latent_image, *args, **kwargs):
        return {**latent_image, "samples": torch.full_like(latent_image["samples"], self.value)}

    def draws_noise_in_each_call(self):
        return False


@pytest.fixture
def region(runtime, monkeypatch):
    """A latent of zeros, a base sampler that leaves zeros, and a prompt for the right half whose sampler leaves ones."""
    monkeypatch.setattr(runtime.samplers.comfy.sample, "fix_empty_latent_channels", lambda model, latent, *args, **kwargs: latent)
    mask = torch.zeros(64, 64)
    mask[:, 32:] = 1.0
    return SimpleNamespace(latent={"samples": torch.zeros(1, 4, 8, 8)}, base=Fill(0.0), prompts=[runtime.core.REGIONAL_PROMPT(mask, Fill(1.0))])


def test_the_regional_sampler_restores_the_latent_outside_each_region(runtime, region):
    result = runtime.samplers.RegionalSampler.doit(
        seed=0, seed_2nd=0, seed_2nd_mode="ignore", steps=2, base_only_steps=1, denoise=1.0, samples=region.latent, base_sampler=region.base,
        regional_prompts=region.prompts, overlap_factor=0, restore_latent=True, additional_mode="DISABLE", additional_sampler="AUTO", additional_sigma_ratio=0.3)[0]
    assert sampled_only_under_the_mask(result["samples"])


def test_the_advanced_regional_sampler_restores_the_latent_outside_each_region(runtime, region):
    result = runtime.samplers.RegionalSamplerAdvanced.doit(
        add_noise=True, noise_seed=0, steps=3, start_at_step=0, end_at_step=3, overlap_factor=0, restore_latent=True, return_with_leftover_noise=False,
        latent_image=region.latent, base_sampler=region.base, regional_prompts=region.prompts, additional_mode="DISABLE", additional_sampler="AUTO",
        additional_sigma_ratio=0.3, unique_id=None)[0]
    assert sampled_only_under_the_mask(result["samples"])
