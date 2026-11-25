import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def full_iterations():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import impact_server  # noqa: F401
        from impact import core, impact_pack

        patch.setattr(core, "latent_upscale_on_pixel_space_shape", lambda samples, *args, **kwargs: samples)

        def of_iterative_upscale(schedule, steps, step_mode="simple"):
            iterations = []

            def sampler(name):
                def sample(latent, hook=None):
                    iterations[-1].add(name)
                    return latent
                return SimpleNamespace(sample=sample)

            class Upscaler(core.TwoSamplersForMaskUpscaler):
                def upscale_shape(self, *args, **kwargs):
                    iterations.append(set())
                    return super().upscale_shape(*args, **kwargs)

            upscaler = Upscaler("nearest-exact", schedule, False, sampler("base"), sampler("mask"), torch.ones(1, 64, 64), vae=None,
                                full_sampler_opt=sampler("full"))
            impact_pack.IterativeLatentUpscale().doit({"samples": torch.zeros(1, 4, 8, 8)}, 2.0, steps, "", upscaler, step_mode)
            return "".join("F" if "full" in seen else "." for seen in iterations)

        yield of_iterative_upscale


@pytest.mark.parametrize("schedule, expected", [
    ("none", "......."),
    ("interleave1", ".F.F.F."),
    ("interleave2", "..F..F."),
    ("interleave3", "...F..."),
    ("last1", "......F"),
    ("last2", ".....FF"),
    ("interleave1+last1", ".F.F.FF"),
    ("interleave2+last1", "..F..FF"),
    ("interleave3+last1", "...F..F"),
])
def test_a_schedule_samples_in_full_in_the_iterations_its_name_says(full_iterations, schedule, expected):
    assert full_iterations(schedule, 7) == expected


@pytest.mark.parametrize("schedule, steps, expected", [
    ("last1", 1, "F"),
    ("last2", 1, "F"),
    ("interleave2+last1", 1, "F"),
    ("last1", 2, ".F"),
    ("last2", 2, "FF"),
    ("interleave3+last1", 2, ".F"),
])
def test_the_last_iterations_of_one_and_of_two_steps(full_iterations, schedule, steps, expected):
    assert full_iterations(schedule, steps) == expected


@pytest.mark.parametrize("schedule, expected", [("last2", ".....FF"), ("interleave3+last1", "...F..F")])
def test_the_geometric_step_mode_has_the_same_iterations(full_iterations, schedule, expected):
    assert full_iterations(schedule, 7, "geometric") == expected
