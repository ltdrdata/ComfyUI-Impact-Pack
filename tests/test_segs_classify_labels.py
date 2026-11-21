import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import core, hf_nodes
        yield SimpleNamespace(SEG=core.SEG, classify=hf_nodes.SEGS_Classify().doit)


AGES = ["Age0to2", "Age3to9", "Age10to19", "Age20to29", "Age30to39", "Age40to49", "Age50to59", "Age60to69", "Age70plus"]


def classifier_of(answers):
    left = iter(answers)
    return lambda picture: [{"label": label, "score": 1.0 / (rank + 2)} for rank, label in enumerate(next(left))]


def segs_of(runtime, count):
    return (64, 64), [runtime.SEG(torch.zeros((1, 16, 16, 3)), None, 1.0, (0, 0, 16, 16), (0, 0, 16, 16), "face", None) for _ in range(count)]


def test_the_labels_come_in_the_order_in_which_the_classifier_gave_them(runtime):
    labels = runtime.classify(classifier_of([AGES]), segs_of(runtime, 1), "Age20to29 > 0.1", "")[2]

    assert labels == AGES


def test_a_label_that_a_later_seg_adds_comes_after_those_of_the_segs_before_it(runtime):
    first, second = AGES[:5], AGES[8:2:-1]

    labels = runtime.classify(classifier_of([first, second]), segs_of(runtime, 2), "Age20to29 > 0.1", "")[2]

    assert labels == first + ["Age70plus", "Age60to69", "Age50to59", "Age40to49"]


def test_the_segs_are_split_by_the_expression_as_before(runtime):
    segs = segs_of(runtime, 2)

    filtered, remained, _ = runtime.classify(classifier_of([["female", "male"], ["male", "female"]]), segs, "#Female > #Male", "")

    assert filtered == ((64, 64), [segs[1][0]])
    assert remained == ((64, 64), [segs[1][1]])
