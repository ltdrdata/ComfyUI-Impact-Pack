"""A sampler cloned with other conditionings stays the sampler it was.

Run with IMPACT_COMFY_ROOT set.
"""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def sampling():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import impact_sampling
        yield impact_sampling


def test_a_clone_with_other_conditionings_keeps_every_other_setting(sampling):
    def scheduler_func(model, sampler, steps):
        return None

    sampler = sampling.KSamplerAdvancedWrapper("model", 7.0, "euler", "simple", "positive", "negative",
                                               sampler_opt="sampler", sigma_factor=0.7, scheduler_func=scheduler_func)
    clone = sampler.clone_with_conditionings("other positive", "other negative")
    assert clone.params == ("model", 7.0, "euler", "simple", "other positive", "other negative", 0.7)
    assert clone.sampler_opt == "sampler" and clone.scheduler_func is scheduler_func
    assert sampler.params == ("model", 7.0, "euler", "simple", "positive", "negative", 0.7)


def test_a_clone_samples_with_the_sigma_factor_and_the_scheduler_func_of_its_sampler(sampling, monkeypatch):
    given = []
    monkeypatch.setattr(sampling, "separated_sample", lambda *args, **kwargs: given.append(kwargs) or args[9])
    func = object()
    sampler = sampling.KSamplerAdvancedWrapper("model", 7.0, "euler", "simple", "positive", "negative", sigma_factor=0.7, scheduler_func=func)
    sampler.clone_with_conditionings("other positive", "other negative").sample_advanced(True, 0, 4, {"samples": None}, 0, 4, False)
    assert given[0]["sigma_ratio"] == 0.7 and given[0]["scheduler_func"] is func
