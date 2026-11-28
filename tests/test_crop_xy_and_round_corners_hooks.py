import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

# What the detector model finds in a 160x96 image: two boxes above the threshold of 0.5 and one below it.
FOUND = (np.array([3, 7, 5], dtype=np.int32), np.array([0.9, 0.6, 0.2], dtype=np.float32),
         np.array([[20, 30, 60, 50], [100, 10, 140, 70], [0, 0, 30, 30]], dtype=np.uint32))


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import core, hooks, hook_nodes, impact_onnx, impact_pack, utils
        patch.setattr(impact_onnx, "onnx_inference", lambda image, onnx_model: FOUND, raising=False)

        def detect(hook, dilation=0, crop_factor=2.0):
            return core.ONNXDetector("model.onnx").detect(torch.zeros((1, 96, 160, 3)), 0.5, dilation, crop_factor, detailer_hook=hook)

        yield SimpleNamespace(
            core=core, hooks=hooks, utils=utils, pack=impact_pack, detect=detect,
            crop=lambda x, y: impact_pack.CropRegionXYFactorDetailerHookProvider.doit(x, y)[0],
            round=lambda radius: hook_nodes.SEGSRoundCornersDetailerHookProvider.doit(radius)[0])


def rounded(runtime, mask, radius):
    seg = runtime.core.SEG(None, mask, 1.0, (0, 0, mask.shape[1], mask.shape[0]), (0, 0, mask.shape[1], mask.shape[0]), "face")
    return runtime.round(radius).post_detection((mask.shape, [seg]))[1][0].cropped_mask


@pytest.mark.parametrize("box", [(462, 487, 562, 537), (5, 10, 45, 30), (980, 900, 1024, 1024)])
@pytest.mark.parametrize("factor", [1.0, 2.5, 3.0])
def test_equal_factors_give_the_region_of_that_crop_factor_in_place_of_the_detectors(runtime, box, factor):
    made_by_detector = runtime.utils.make_crop_region(1024, 1024, box, 1.5)

    region = runtime.crop(factor, factor).post_crop_region(1024, 1024, box, made_by_detector)

    assert list(region) == runtime.utils.make_crop_region(1024, 1024, box, factor)


@pytest.mark.parametrize("factors,expected", [((2.0, 4.0), [412, 412, 612, 612]), ((4.0, 2.0), [312, 462, 712, 562])])
def test_the_width_follows_crop_factor_x_and_the_height_crop_factor_y(runtime, factors, expected):
    box = 462, 487, 562, 537  # 100 wide and 50 high, around the middle of the image

    assert list(runtime.crop(*factors).post_crop_region(1024, 1024, box, [0, 0, 1024, 1024])) == expected


@pytest.mark.parametrize("box", [(5, 10, 45, 30), (140, 70, 160, 96), (0, 40, 20, 60)])
@pytest.mark.parametrize("factors", [(3.0, 5.0), (4.0, 2.0), (10.0, 10.0)])
def test_at_a_border_the_region_stays_inside_the_image_and_holds_the_box(runtime, box, factors):
    x1, y1, x2, y2 = runtime.crop(*factors).post_crop_region(160, 96, box, [0, 0, 160, 96])

    assert 0 <= x1 <= box[0] and box[2] <= x2 <= 160
    assert 0 <= y1 <= box[1] and box[3] <= y2 <= 96
    assert (x2 - x1, y2 - y1) == (min(160, (box[2] - box[0]) * factors[0]), min(96, (box[3] - box[1]) * factors[1]))


def test_the_detector_crops_each_seg_to_the_region_of_the_hook(runtime):
    shape, segs = runtime.detect(runtime.crop(1.5, 3.0))

    assert [tuple(seg.crop_region) for seg in segs] == [(10, 10, 70, 70), (90, 0, 150, 96)]
    assert [seg.cropped_mask.shape for seg in segs] == [(60, 60), (96, 60)]
    assert segs[0].cropped_mask[20:40, 10:50].all() and segs[0].cropped_mask.sum() == 40 * 20


def test_mask_to_segs_crops_each_seg_to_the_region_of_the_hook(runtime):
    mask = np.zeros((96, 160), np.float32)
    mask[30:50, 20:60] = 1

    shape, (seg,) = runtime.core.mask_to_segs(mask, False, 2.0, False, detailer_hook=runtime.crop(1.5, 3.0))

    assert tuple(seg.crop_region) == (10, 10, 70, 70) and seg.cropped_mask.shape == (60, 60)


def test_a_rectangle_loses_its_corners_and_keeps_its_middle_and_edges(runtime):
    result = rounded(runtime, np.ones((48, 64), np.float32), 16)

    assert [result[y, x] for y, x in ((0, 0), (0, 63), (47, 0), (47, 63))] == [0, 0, 0, 0]
    assert [result[y, x] for y, x in ((24, 32), (0, 32), (47, 32), (24, 0), (24, 63))] == [1, 1, 1, 1, 1]
    assert result[16, 0] == result[0, 16] == 1 and result[4, 4] == 0


def test_radius_0_leaves_the_mask_as_it_was(runtime):
    mask = np.zeros((48, 64), np.float32)
    mask[8:40, 4:60] = 1
    mask[20, 30] = 0.5

    assert np.array_equal(rounded(runtime, mask, 0), mask)


def test_a_radius_beyond_half_the_shorter_side_is_that_half(runtime):
    mask = np.ones((48, 64), np.float32)

    clamped = rounded(runtime, mask, 500)

    assert np.array_equal(clamped, rounded(runtime, mask, 24))
    assert clamped[0, 0] == 0 and clamped[0, 32] == clamped[24, 0] == 1
    assert not np.array_equal(clamped, rounded(runtime, mask, 16))


def test_a_mask_that_does_not_fill_its_crop_is_rounded_at_its_own_bounds(runtime):
    mask = np.zeros((64, 64), np.float32)
    mask[16:48, 8:56] = 1

    result = rounded(runtime, mask, 8)

    assert [result[y, x] for y, x in ((16, 8), (16, 55), (47, 8), (47, 55))] == [0, 0, 0, 0]
    assert [result[y, x] for y, x in ((16, 32), (47, 32), (32, 8), (32, 55), (32, 32))] == [1, 1, 1, 1, 1]
    assert not result[:16].any() and not result[:, :8].any()


def test_a_dilated_mask_gets_round_and_not_cut_corners(runtime):
    mask = np.zeros((64, 64), np.float32)
    mask[20:44, 16:48] = 1
    dilated = runtime.utils.dilate_mask(mask, 9)
    assert dilated[16:48, 12:52].all() and dilated.sum() == 32 * 40

    result = rounded(runtime, dilated, 12)

    # Corner of the dilated bounds at (16, 12): the arc keeps (21, 17), 5 px in along the diagonal, which a 45 degree cut would drop.
    assert result[16, 12] == 0 and result[18, 14] == 0 and result[21, 17] == 1
    assert result[16, 32] == result[47, 32] == result[32, 12] == result[32, 51] == 1


def test_the_detector_rounds_the_mask_it_dilated_at_the_dilated_bounds(runtime):
    shape, segs = runtime.detect(runtime.round(8), dilation=9, crop_factor=3.0)

    # The first box is rows 20..39 and columns 20..59 of its crop region; the dilation of 9 widens it by 4 px on each side.
    first = segs[0].cropped_mask
    assert tuple(segs[0].crop_region) == (0, 10, 120, 70)
    assert first[16, 16] == first[16, 63] == first[43, 16] == first[43, 63] == 0
    assert first[16, 40] == first[43, 40] == first[30, 16] == first[30, 63] == first[30, 40] == 1


def test_an_empty_mask_passes_and_every_other_field_of_a_seg_stays(runtime):
    full = np.full((24, 32), 0.5, np.float32)
    empty = np.zeros((24, 32), np.float32)
    image = np.zeros((1, 24, 32, 3), np.float32)
    segs = (96, 160), [runtime.core.SEG(image, full, 0.9, (8, 8, 40, 32), (12, 12, 36, 28), "face", "cnet"),
                       runtime.core.SEG(None, empty, 0.7, (0, 0, 32, 24), (4, 4, 28, 20), "hand")]

    shape, (first, second) = runtime.round(4).post_detection(segs)

    assert shape == (96, 160)
    assert first.cropped_mask.dtype == np.float32 and first.cropped_mask.shape == (24, 32)
    assert first.cropped_mask[0, 0] == 0 and first.cropped_mask[12, 16] == 0.5
    assert first.cropped_image is image
    assert (first.confidence, first.crop_region, first.bbox, first.label, first.control_net_wrapper) == (0.9, (8, 8, 40, 32), (12, 12, 36, 28), "face", "cnet")
    assert np.array_equal(second.cropped_mask, empty) and second._replace(cropped_mask=None) == segs[1][1]._replace(cropped_mask=None)
    assert np.array_equal(full, np.full((24, 32), 0.5, np.float32)), "the mask of the detector is not written into"


@pytest.mark.parametrize("order", ["crop first", "round first"])
def test_both_hooks_act_when_they_are_combined(runtime, order):
    crop, round_corners = runtime.crop(1.5, 3.0), runtime.round(8)
    hook = runtime.hooks.DetailerHookCombine(*((crop, round_corners) if order == "crop first" else (round_corners, crop)))

    shape, segs = runtime.detect(hook)

    assert [tuple(seg.crop_region) for seg in segs] == [(10, 10, 70, 70), (90, 0, 150, 96)]
    first = segs[0].cropped_mask
    assert first.shape == (60, 60) and first[20, 10] == 0 and first[20, 30] == first[30, 10] == first[30, 30] == 1


def test_face_detailer_hands_both_hooks_to_its_detector(runtime):
    hook = runtime.hooks.DetailerHookCombine(runtime.crop(1.5, 3.0), runtime.round(8))
    params = dict(image=torch.full((1, 96, 160, 3), 0.25), model="DUMMY", clip=None, vae=None, guide_size=64, guide_size_for=True, max_size=64, seed=0, steps=1,
                  cfg=1.0, sampler_name="euler", scheduler="normal", positive=[], negative=[], denoise=0.5, feather=0, noise_mask=True,
                  force_inpaint=True, bbox_threshold=0.5, bbox_dilation=0, bbox_crop_factor=2.0, sam_detection_hint="center-1",
                  sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative="False",
                  drop_size=10, bbox_detector=runtime.core.ONNXDetector("model.onnx"), wildcard="", detailer_hook=hook)

    output, cropped, _, mask, _, _ = runtime.pack.FaceDetailer().doit(**params)

    assert sorted(tuple(crop.shape) for crop in cropped) == [(1, 60, 60, 3), (1, 96, 60, 3)]
    assert mask.shape == (96, 160) and mask[30, 20] == 0 and mask[30, 40] == mask[40, 20] == mask[40, 40] == 1
