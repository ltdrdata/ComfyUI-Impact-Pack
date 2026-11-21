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
        from impact import segs_nodes
        from comfy_extras.nodes_images import BoundingBox
        yield SimpleNamespace(
            segs=lambda bboxes, image, crop_factor=3.0: segs_nodes.BoundingBoxesToSEGS.execute(bboxes, image, crop_factor).result[0],
            boxes=lambda segs: segs_nodes.SEGSToBoundingBoxes.execute(segs).result[0],
            mask=lambda segs: segs_nodes.SegsToCombinedMask.execute(segs).result[0],
            primitive=BoundingBox)


FACE = {"x": 20.0, "y": 30.0, "width": 40.0, "height": 20.0, "label": "face", "score": 0.75}
HAND = {"x": 100.5, "y": 2.25, "width": 15.5, "height": 20.0}


def image(frames=1):
    return torch.rand((frames, 96, 160, 3), generator=torch.Generator().manual_seed(1))


def test_each_box_becomes_a_seg_with_the_mask_a_bbox_detector_gives_for_the_box(runtime):
    picture = image()

    shape, (face, hand) = runtime.segs([[FACE, HAND]], picture, crop_factor=2.0)

    assert shape == (96, 160)
    assert (tuple(face.bbox), face.label, face.confidence) == ((20.0, 30.0, 60.0, 50.0), "face", 0.75)
    assert (tuple(hand.bbox), hand.label, hand.confidence) == ((100.5, 2.25, 116.0, 22.25), "", 1.0)
    assert tuple(face.crop_region) == (0, 20, 80, 60)
    assert torch.equal(face.cropped_image, picture[:, 20:60, 0:80])
    expected = torch.zeros((1, 96, 160))
    expected[0, 30:51, 20:61] = 1.0
    expected[0, 2:23, 100:117] = 1.0
    assert torch.equal(runtime.mask((shape, [face, hand])), expected)


@pytest.mark.parametrize("boxes", [FACE, [FACE], [[FACE]]], ids=["a box", "a list of boxes", "a list for one frame"])
def test_each_form_of_the_boxes_gives_the_same_segs(runtime, boxes):
    shape, (face,) = runtime.segs(boxes, image())

    assert (tuple(face.bbox), face.label, face.confidence) == ((20, 30, 60, 50), "face", 0.75)


def test_the_box_of_comfyuis_bounding_box_node_becomes_a_seg(runtime):
    box = runtime.primitive.execute(x=20, y=30, width=40, height=20).result[0]

    shape, (seg,) = runtime.segs(box, image())

    assert tuple(seg.bbox) == (20, 30, 60, 50)


def test_segs_come_back_the_same_through_bounding_boxes(runtime):
    first = runtime.segs([[FACE, HAND]], image())

    again = runtime.segs(runtime.boxes(first), image())

    assert [(tuple(seg.bbox), seg.label, seg.confidence, tuple(seg.crop_region)) for seg in again[1]] == \
           [(tuple(seg.bbox), seg.label, seg.confidence, tuple(seg.crop_region)) for seg in first[1]]
    assert all(np.array_equal(a.cropped_mask, b.cropped_mask) for a, b in zip(again[1], first[1]))


def test_a_box_is_cut_to_the_image_and_a_box_outside_it_gives_no_seg(runtime):
    partly = {"x": 150.0, "y": -10.0, "width": 40.0, "height": 30.0}
    outside = {"x": 200.0, "y": 10.0, "width": 30.0, "height": 30.0}

    shape, segs = runtime.segs([partly, outside], image())

    assert [tuple(seg.bbox) for seg in segs] == [(150, 0, 160, 20)]


def test_no_box_gives_segs_without_a_seg(runtime):
    assert runtime.segs([[]], image()) == ((96, 160), [])
    assert runtime.segs([], image()) == ((96, 160), [])


def test_the_boxes_of_more_than_one_frame_are_refused(runtime):
    with pytest.raises(Exception, match="more than one frame"):
        runtime.segs([[FACE], [FACE]], image())


def test_a_batch_of_images_is_refused(runtime):
    with pytest.raises(Exception, match="image batches"):
        runtime.segs([[FACE]], image(frames=2))
