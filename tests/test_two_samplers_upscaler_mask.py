import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def noise_masks():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import core

        def given_to_the_samplers(mask, latent_size):
            seen = {}

            def sampler(name):
                def sample(latent, hook=None):
                    seen[name] = latent["noise_mask"].clone()
                    return latent
                return SimpleNamespace(sample=sample)

            upscaler = core.TwoSamplersForMaskUpscaler("nearest-exact", "none", False, sampler("base"), sampler("mask"), mask, vae=None)
            upscaler.do_samples((0, 3), sampler("base"), sampler("mask"), "none", mask, {"samples": torch.zeros(1, 4, *latent_size)})
            return seen

        yield given_to_the_samplers


def test_the_mask_sampler_gets_the_region_of_the_mask_and_the_base_sampler_the_rest(noise_masks):
    mask = torch.zeros(1, 64, 96)
    mask[:, 16:48, 48:] = 1.0
    seen = noise_masks(mask, (16, 24))
    region = torch.zeros(16, 24)
    region[4:12, 12:] = 1.0
    assert seen["mask"].shape == (1, 1, 16, 24)
    assert torch.equal(seen["mask"][0, 0] > 0.5, region > 0.5)
    assert torch.equal(seen["base"][0, 0], (seen["mask"][0, 0] != 1.0).float())
