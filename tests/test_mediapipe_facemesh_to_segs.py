import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import segs_nodes
        yield SimpleNamespace(sent=server.send_sync, segs=lambda image: segs_nodes.MediaPipeFaceMeshToSEGS.execute(
            image, 3.0, False, 50, 1, 0, True, False, False, False, False, False, False, False).result[0])


def mesh(*ovals):
    # The face oval in the colour comfyui_controlnet_aux draws it in, one frame for each (top, left, bottom, right).
    image = torch.zeros((len(ovals), 64, 64, 3))
    for frame, (top, left, bottom, right) in zip(image, ovals):
        frame[top:bottom, left:right] = (torch.tensor((0x0A, 0xC8, 0x0A)) + 0.5) / 255
    return image


LARGE, SMALL = (6, 10, 58, 54), (20, 24, 44, 40)
FIRST_ONLY = "used the first image only: the batch holds 2 images. Put Image Batch to Image List before this node to get the SEGS of every image"


def test_one_frame_gives_the_seg_of_its_face(runtime, caplog):
    with caplog.at_level(logging.WARNING):
        shape, (face,) = runtime.segs(mesh(LARGE))

    assert shape == (64, 64)
    assert (face.label, tuple(face.bbox)) == ("face", (10, 6, 54, 58))
    assert not caplog.messages
    runtime.sent.assert_not_called()


def test_a_batch_gives_the_segs_of_its_first_frame_and_marks_the_node(runtime, caplog):
    from comfy_execution.utils import CurrentNodeContext
    alone = runtime.segs(mesh(SMALL))

    with caplog.at_level(logging.WARNING), CurrentNodeContext("prompt", "7"):
        shape, (face,) = runtime.segs(mesh(SMALL, LARGE))

    assert shape == alone[0]
    assert (face.label, tuple(face.bbox), tuple(face.crop_region)) == (alone[1][0].label, tuple(alone[1][0].bbox), tuple(alone[1][0].crop_region))
    assert tuple(face.bbox) == (24, 20, 40, 44)
    assert np.array_equal(face.cropped_mask, alone[1][0].cropped_mask)
    assert caplog.messages == [f"[Impact Pack] MediaPipeFaceMeshToSEGS: {FIRST_ONLY}"]
    runtime.sent.assert_called_once_with("impact-node-warning", {"node_id": "7", "text": FIRST_ONLY}, "page")


def test_a_batch_outside_a_prompt_warns_in_the_log_only(runtime, caplog):
    with caplog.at_level(logging.WARNING):
        runtime.segs(mesh(SMALL, LARGE))

    assert caplog.messages == [f"[Impact Pack] MediaPipeFaceMeshToSEGS: {FIRST_ONLY}"]
    runtime.sent.assert_not_called()
