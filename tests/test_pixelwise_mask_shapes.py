import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def nodes():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import impact_server  # noqa: F401
        from impact import core, impact_pack, segs_nodes
        yield SimpleNamespace(
            seg=lambda top, left, size: core.SEG(None, np.ones((size, size), dtype=np.float32), 1.0, [left, top, left + size, top + size], [left, top, left + size, top + size], "seg", None),
            apart=lambda segs1, segs2: segs_nodes.SEGSIntersectionFilter.execute(segs1, segs2, 0.5).result[0][1],
            add=lambda mask1, mask2: impact_pack.AddMask.execute(mask1, mask2).result[0],
            subtract=lambda mask1, mask2: impact_pack.SubtractMask.execute(mask1, mask2).result[0],
            overlap=lambda mask1, mask2: impact_pack.BitwiseAndMask.execute(mask1, mask2).result[0],
            sent=server.send_sync)


def rectangle(top, left, bottom, right):
    mask = torch.zeros((64, 64))
    mask[top:bottom, left:right] = 1.0
    return mask


BATCH = torch.stack([rectangle(8, 8, 24, 24), rectangle(16, 16, 40, 40), rectangle(40, 40, 56, 56)])
SINGLE = rectangle(16, 16, 48, 48)
LEFT_HALF = torch.zeros((16, 32))
LEFT_HALF[:, :16] = 1.0


@pytest.mark.parametrize("single", [SINGLE, SINGLE[None]], ids=["height by width", "batch of one"])
def test_a_single_mask_goes_with_every_mask_of_a_batch(nodes, single):
    torch.testing.assert_close(nodes.add(single, BATCH), torch.clamp(BATCH + SINGLE, 0, 1))
    torch.testing.assert_close(nodes.add(BATCH, single), torch.clamp(BATCH + SINGLE, 0, 1))
    torch.testing.assert_close(nodes.subtract(single, BATCH), torch.clamp(SINGLE - BATCH, 0, 1))
    torch.testing.assert_close(nodes.subtract(BATCH, single), torch.clamp(BATCH - SINGLE, 0, 1))
    torch.testing.assert_close(nodes.overlap(single, BATCH), BATCH * SINGLE)
    torch.testing.assert_close(nodes.overlap(BATCH, single), BATCH * SINGLE)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.uint8, torch.bool], ids=str)
def test_a_mask_of_another_dtype_counts_as_its_values(nodes, dtype):
    other = SINGLE.to(dtype)
    for mask1, mask2 in ((BATCH, other), (other, BATCH), (other, other)):
        float1, float2 = mask1.float(), mask2.float()
        torch.testing.assert_close(nodes.add(mask1, mask2), torch.clamp(float1 + float2, 0, 1))
        torch.testing.assert_close(nodes.subtract(mask1, mask2), torch.clamp(float1 - float2, 0, 1))
        torch.testing.assert_close(nodes.overlap(mask1, mask2), float1 * float2)


@pytest.mark.parametrize("mask1", [torch.ones((64, 64)), torch.ones((1, 64, 64)), torch.ones((3, 64, 64))],
                         ids=["height by width", "batch of one", "batch of three"])
def test_a_mask_of_another_size_is_stretched_to_the_first_mask(nodes, mask1):
    removed = nodes.subtract(mask1, LEFT_HALF)
    kept = nodes.overlap(mask1, LEFT_HALF)
    added = nodes.add(torch.zeros_like(mask1), LEFT_HALF)

    for result in (removed, kept, added):
        assert result.shape == mask1.reshape((-1, 64, 64)).shape
    assert removed[..., :31].max() == 0 and removed[..., 33:].min() == 1
    for result in (kept, added):
        assert result[..., :31].min() == 1 and result[..., 33:].max() == 0


@pytest.mark.parametrize("node", ["add", "subtract", "overlap"])
def test_two_batches_of_different_length_give_the_first_and_a_warning_on_the_node(nodes, caplog, node):
    from comfy_execution.utils import CurrentNodeContext
    text = "returned mask1 unchanged: mask1 is a batch of 2 and mask2 a batch of 3"

    with caplog.at_level(logging.WARNING), CurrentNodeContext("prompt", "12"):
        result = getattr(nodes, node)(BATCH[:2], BATCH)

    torch.testing.assert_close(result, BATCH[:2])
    assert caplog.messages == [f"[Impact Pack] Pixelwise: {text}"]
    nodes.sent.assert_called_once_with("impact-node-warning", {"node_id": "12", "text": text}, "page")


def test_a_call_from_outside_a_prompt_warns_in_the_log_only(nodes, caplog):
    with caplog.at_level(logging.WARNING):
        result = nodes.add(BATCH[:2], BATCH)

    torch.testing.assert_close(result, BATCH[:2])
    assert len(caplog.messages) == 1
    nodes.sent.assert_not_called()


@pytest.mark.parametrize("mask2", [BATCH, SINGLE, LEFT_HALF], ids=["one shape", "single mask", "another size"])
def test_masks_that_go_together_give_no_warning(nodes, caplog, mask2):
    with caplog.at_level(logging.WARNING):
        nodes.add(BATCH, mask2)

    assert not caplog.messages
    nodes.sent.assert_not_called()


def test_the_intersection_filter_compares_segs_of_images_of_two_sizes(nodes):
    corner, middle = nodes.seg(0, 0, 16), nodes.seg(24, 24, 16)
    other = (128, 128), [nodes.seg(0, 0, 32)]

    assert nodes.apart(((64, 64), [corner, middle]), other) == [middle]
