import json
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
        from impact import core, segs_nodes
        from comfy_extras.nodes_rtdetr import DrawBBoxes
        yield SimpleNamespace(SEG=core.SEG, boxes=lambda segs: segs_nodes.SEGSToBoundingBoxes.execute(segs).result[0], draw=DrawBBoxes)


def test_each_seg_gives_its_box_with_its_label_and_score(runtime):
    face = runtime.SEG(None, np.ones((40, 60), dtype=np.float32), np.array([0.75], dtype=np.float32), (10, 20, 70, 60), np.array([20, 25, 50, 55]), "face", None)
    hand = runtime.SEG(None, np.ones((30, 30), dtype=np.float32), 0.5, (100, 0, 130, 30), (104.5, 2.0, 120.5, 22.0), "hand", None)

    boxes = runtime.boxes(((128, 160), [face, hand]))

    assert boxes == [[{"x": 20.0, "y": 25.0, "width": 30.0, "height": 30.0, "label": "face", "score": 0.75},
                      {"x": 104.5, "y": 2.0, "width": 16.0, "height": 20.0, "label": "hand", "score": 0.5}]]
    assert json.loads(json.dumps(boxes)) == boxes


def test_segs_without_a_seg_give_one_frame_without_a_box(runtime):
    assert runtime.boxes(((128, 160), [])) == [[]]


def test_a_seg_of_a_video_gives_the_box_of_the_object_in_each_frame(runtime):
    masks = torch.zeros((3, 50, 100))
    masks[0, 10:30, 5:25] = 1.0
    masks[2, 12:32, 60:90] = 1.0
    moving = runtime.SEG(None, masks, 1.0, (200, 100, 300, 150), (205, 110, 290, 132), "A", None)

    boxes = runtime.boxes(((480, 640), [moving]))

    assert boxes == [[{"x": 205.0, "y": 110.0, "width": 20.0, "height": 20.0, "label": "A", "score": 1.0}],
                     [],
                     [{"x": 260.0, "y": 112.0, "width": 30.0, "height": 20.0, "label": "A", "score": 1.0}]]


def test_draw_bboxes_of_comfyui_draws_the_boxes_of_each_frame(runtime):
    masks = torch.zeros((2, 50, 100))
    masks[0, 10:30, 5:25] = 1.0
    masks[1, 10:30, 65:85] = 1.0
    moving = runtime.SEG(None, masks, 1.0, (20, 20, 120, 70), (25, 30, 105, 50), "A", None)
    frames = torch.zeros((2, 96, 160, 3))

    drawn = runtime.draw.execute(bboxes=runtime.boxes(((96, 160), [moving])), image=frames).result[0]

    left_edges = [int(torch.nonzero(frame.sum(dim=2))[:, 1].min()) for frame in drawn]
    assert left_edges == [25, 85]
