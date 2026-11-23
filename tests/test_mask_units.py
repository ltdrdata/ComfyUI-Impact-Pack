import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def nodes():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import core, impact_onnx, impact_pack, segs_nodes
        found = np.array([3], dtype=np.int32), np.array([0.9], dtype=np.float32), np.array([[20, 30, 60, 50]], dtype=np.uint32)
        patch.setattr(impact_onnx, "onnx_inference", lambda image, onnx_model: found, raising=False)
        yield SimpleNamespace(
            flatten=lambda masks: impact_pack.FlattenMask.execute(masks).result[0],
            add=lambda mask1, mask2: impact_pack.AddMask.execute(mask1, mask2).result[0],
            subtract=lambda mask1, mask2: impact_pack.SubtractMask.execute(mask1, mask2).result[0],
            detected=lambda: segs_nodes.From_SEG_ELT.execute(
                core.ONNXDetector("model.onnx").detect(torch.zeros((1, 96, 160, 3)), 0.5, 0, 2.0)[1][0]).result[2])


def rectangle(top, left, bottom, right):
    mask = torch.zeros((64, 64))
    mask[top:bottom, left:right] = 1.0
    return mask


FIRST, SECOND, THIRD = rectangle(8, 8, 24, 24), rectangle(16, 16, 40, 40), rectangle(40, 40, 56, 56)
BOTH = torch.clamp(FIRST + SECOND, 0, 1)


def test_flatten_mask_batch_gives_a_float_mask_from_0_to_1(nodes):
    torch.testing.assert_close(nodes.flatten(torch.stack([FIRST, SECOND])), BOTH)


def test_a_flattened_mask_adds_to_and_subtracts_from_another_mask(nodes):
    flattened = nodes.flatten(torch.stack([FIRST, SECOND]))

    torch.testing.assert_close(nodes.add(flattened, THIRD), torch.clamp(BOTH + THIRD, 0, 1))
    torch.testing.assert_close(nodes.subtract(flattened, SECOND), torch.clamp(BOTH - SECOND, 0, 1))
    torch.testing.assert_close(nodes.add(flattened, flattened), BOTH)


def test_the_mask_of_an_onnx_detection_adds_to_another_mask(nodes):
    mask = nodes.detected()
    box = torch.zeros((40, 80))
    box[10:30, 20:60] = 1.0

    torch.testing.assert_close(mask, box)
    torch.testing.assert_close(nodes.add(mask, torch.zeros((40, 80))), box)
