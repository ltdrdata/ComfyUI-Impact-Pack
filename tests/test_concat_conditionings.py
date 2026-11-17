"""ConcatConditionings with one conditioning and with more.

Run with IMPACT_COMFY_ROOT set.
"""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module")
def node():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import special_samplers
        yield special_samplers.ConcatConditionings


def conditioning(value, tokens):
    return [[torch.full((1, tokens, 4), value), {"pooled_output": torch.full((1, 4), value)}]]


def test_one_conditioning_is_returned_as_it_is(node):
    first = conditioning(1.0, 3)

    assert node.doit(conditioning1=first)[0] is first


def test_more_conditionings_are_joined_along_the_tokens_onto_the_first(node):
    first, second, third = conditioning(1.0, 3), conditioning(2.0, 2), conditioning(3.0, 1)

    result = node.doit(conditioning1=first, conditioning2=second, conditioning3=third)[0]

    assert torch.equal(result[0][0], torch.cat((first[0][0], second[0][0], third[0][0]), 1))
    assert result[0][1] is not first[0][1] and torch.equal(result[0][1]["pooled_output"], first[0][1]["pooled_output"])
    assert [c[0][0].shape[1] for c in (first, second, third)] == [3, 2, 1]
