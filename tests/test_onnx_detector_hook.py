import os
from types import SimpleNamespace

import numpy as np
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
        from impact import core, hooks, impact_onnx
        # What the model finds in the picture: two objects over the threshold and one under it.
        found = np.array([3, 7, 5], dtype=np.int32), np.array([0.9, 0.6, 0.2], dtype=np.float32), np.array([[20, 30, 60, 50], [100, 10, 140, 70], [0, 0, 30, 30]], dtype=np.uint32)
        patch.setattr(impact_onnx, "onnx_inference", lambda image, onnx_model: found, raising=False)
        yield SimpleNamespace(
            hooks=hooks, onnx=impact_onnx,
            detect=lambda hook=None: core.ONNXDetector("model.onnx").detect(torch.zeros((1, 96, 160, 3)), 0.5, 0, 2.0, detailer_hook=hook))


def described(segs):
    return segs[0], [(seg.label, tuple(int(value) for value in seg.bbox), tuple(int(value) for value in seg.crop_region)) for seg in segs[1]]


def test_a_hook_that_changes_nothing_gives_what_the_detector_gives_without_a_hook(runtime):
    assert described(runtime.detect()) == ((96, 160), [("3", (20, 30, 60, 50), (0, 20, 80, 60)), ("7", (100, 10, 140, 70), (80, 0, 160, 96))])

    assert described(runtime.detect(runtime.hooks.DetailerHook())) == described(runtime.detect())


def test_the_hook_sets_the_crop_region_of_each_seg(runtime):
    class WholeImage(runtime.hooks.DetailerHook):
        def post_crop_region(self, w, h, item_bbox, crop_region):
            return 0, 0, w, h

    shape, (first, second) = runtime.detect(WholeImage())

    assert first.crop_region == second.crop_region == (0, 0, 160, 96)
    expected = np.zeros((96, 160))
    expected[30:50, 20:60] = 1
    assert np.array_equal(first.cropped_mask, expected)


def test_a_crop_region_that_holds_a_part_of_the_box_gives_that_part_of_the_mask(runtime):
    class RightHalf(runtime.hooks.DetailerHook):
        def post_crop_region(self, w, h, item_bbox, crop_region):
            return 40, 20, 80, 60

    shape, (first, second) = runtime.detect(RightHalf())

    expected = np.zeros((40, 40))
    expected[10:30, 0:20] = 1
    assert np.array_equal(first.cropped_mask, expected)


def test_the_hook_filters_what_the_detector_found(runtime):
    kept = runtime.detect(runtime.hooks.SEGSLabelFilterDetailerHook("7"))

    assert [seg.label for seg in kept[1]] == ["7"]



def test_an_error_of_the_model_is_the_error_of_the_detector(runtime, monkeypatch):
    def broken(image, onnx_model):
        raise RuntimeError("the model file is not an ONNX model")

    monkeypatch.setattr(runtime.onnx, "onnx_inference", broken)

    with pytest.raises(RuntimeError, match="the model file is not an ONNX model"):
        runtime.detect()
