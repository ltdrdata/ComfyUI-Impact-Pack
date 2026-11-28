import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

LARGE = 4096, 4096
IMAGE = 1024, 768
# int() cuts less than a pixel from the start and from the end of a side, and the whole-pixel region the detector
# gives the hook moves the margin by less than a pixel, half of it on each side.
TOLERANCE = 2.5
TALL = 2000, 1800, 2100, 2200
WIDE = 1800, 2000, 2200, 2120
SQUARE = 1950, 1950, 2150, 2150
ODD = 1903, 2011, 1990, 2240
FLOAT = 1903.4, 2011.7, 1990.2, 2240.9
CROP_FACTORS = 1.0, 1.1, 1.5, 2.0, 3.0, 4.5
FOUND = [[100, 100, 140, 260], [400, 300, 560, 340], [600, 50, 680, 130]]


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
        from impact import core, detectors, hooks, impact_onnx, impact_pack, utils
        hook = impact_pack.CropRegionEqualMarginDetailerHookProvider.execute().result[0]

        def regions(image, box, crop_factor):
            made = utils.make_crop_region(*image, box, crop_factor)
            return made, hook.post_crop_region(*image, box, made)
        yield SimpleNamespace(regions=regions, hook=hook, core=core, detectors=detectors, hooks=hooks, onnx=impact_onnx)


def centred(box, crop_factor):
    margin = min(box[2] - box[0], box[3] - box[1]) * (crop_factor - 1)
    return box[0] - margin / 2, box[1] - margin / 2, box[2] + margin / 2, box[3] + margin / 2


def near(region, expected):
    return all(abs(got - want) <= TOLERANCE for got, want in zip(region, expected))


def sizes(region):
    return region[2] - region[0], region[3] - region[1]


@pytest.mark.parametrize("box", [TALL, WIDE, SQUARE, ODD, FLOAT], ids=["tall", "wide", "square", "odd", "float"])
@pytest.mark.parametrize("crop_factor", CROP_FACTORS)
def test_in_the_middle_the_long_side_gets_the_margin_of_the_short_side(runtime, box, crop_factor):
    made, region = runtime.regions(LARGE, box, crop_factor)

    assert near(region, centred(box, crop_factor))


@pytest.mark.parametrize("image, box", [(LARGE, SQUARE), (LARGE, (1951, 1953, 2088, 2090)), (IMAGE, (0, 300, 200, 500)),
                                        (IMAGE, (824, 568, 1024, 768))],
                         ids=["middle", "odd size", "border", "corner"])
@pytest.mark.parametrize("crop_factor", CROP_FACTORS)
def test_a_square_bbox_keeps_the_region_the_detector_made(runtime, image, box, crop_factor):
    made, region = runtime.regions(image, box, crop_factor)

    assert region == made


@pytest.mark.parametrize("image, box", [(LARGE, TALL), (LARGE, WIDE), (LARGE, FLOAT), (IMAGE, (0, 0, 60, 240)), (IMAGE, (784, 708, 1024, 768))],
                         ids=["tall", "wide", "float", "tall in a corner", "wide in a corner"])
@pytest.mark.parametrize("crop_factor", CROP_FACTORS)
def test_the_short_side_keeps_the_region_the_detector_made(runtime, image, box, crop_factor):
    made, region = runtime.regions(image, box, crop_factor)

    short = (0, 2) if box[2] - box[0] < box[3] - box[1] else (1, 3)
    assert [region[i] for i in short] == [made[i] for i in short]


BORDERS = {
    "tall at the left": (3, 300, 63, 540),
    "tall at the right": (961, 300, 1021, 540),
    "tall at the top": (500, 2, 560, 242),
    "tall at the bottom": (500, 526, 560, 766),
    "tall in a corner": (0, 0, 60, 240),
    "wide at the left": (1, 400, 241, 460),
    "wide at the top": (400, 4, 640, 64),
    "wide in a corner": (784, 708, 1024, 768),
}


@pytest.mark.parametrize("box", BORDERS.values(), ids=BORDERS.keys())
@pytest.mark.parametrize("crop_factor", [1.5, 3.0])
def test_at_a_border_the_region_stays_in_the_image_and_keeps_its_size(runtime, box, crop_factor):
    made, region = runtime.regions(IMAGE, box, crop_factor)

    x1, y1, x2, y2 = region
    assert 0 <= x1 and 0 <= y1 and x2 <= IMAGE[0] and y2 <= IMAGE[1]
    assert x1 <= box[0] and y1 <= box[1] and box[2] <= x2 and box[3] <= y2
    assert near(sizes(region), sizes(centred(box, crop_factor)))


@pytest.mark.parametrize("image, box, crop_factor, expected", [
    (IMAGE, (412, 84, 612, 684), 3.0, (212, 0, 812, 768)),
    (IMAGE, (412, 84, 612, 684), 1.5, (362, 34, 662, 734)),
    (IMAGE, (112, 309, 912, 459), 3.0, (0, 159, 1024, 609)),
    (IMAGE, (112, 309, 912, 459), 1.5, (74.5, 271.5, 949.5, 496.5)),
    ((160, 96), (50, 8, 110, 88), 3.0, (0, 0, 160, 96)),
], ids=["tall, the image cuts the long side", "tall, the margin fits", "wide, the image cuts the long side",
        "wide, the margin fits", "the image is smaller than the region on both sides"])
def test_an_image_smaller_than_the_region_cuts_only_what_does_not_fit(runtime, image, box, crop_factor, expected):
    made, region = runtime.regions(image, box, crop_factor)

    assert near(region, expected)


def assert_each_seg_is_cut_with_its_region(segs, crop_factor):
    assert sorted(tuple(int(value) for value in seg.bbox) for seg in segs[1]) == sorted(tuple(box) for box in FOUND)
    for seg in segs[1]:
        box = tuple(int(value) for value in seg.bbox)
        x1, y1, x2, y2 = seg.crop_region
        assert near(seg.crop_region, centred(box, crop_factor))
        assert tuple(seg.cropped_mask.shape) == (y2 - y1, x2 - x1)
        assert float(seg.cropped_mask.sum()) == (box[2] - box[0]) * (box[3] - box[1])


@pytest.fixture
def detect(runtime, monkeypatch):
    found = np.array([1, 2, 3], dtype=np.int32), np.array([0.9, 0.8, 0.7], dtype=np.float32), np.array(FOUND, dtype=np.uint32)
    monkeypatch.setattr(runtime.onnx, "onnx_inference", lambda image, onnx_model: found, raising=False)

    def detect(hook):
        segs, = runtime.detectors.BboxDetectorForEach().doit(
            runtime.core.ONNXDetector("model.onnx"), torch.zeros((1, 512, 768, 3)), 0.5, 0, 2.0, 1, "all", hook)
        return segs
    return detect


def test_the_bbox_detector_cuts_each_seg_with_the_region_of_the_hook(runtime, detect):
    segs = detect(runtime.hook)

    assert segs[0] == (512, 768)
    assert_each_seg_is_cut_with_its_region(segs, 2.0)


def test_a_region_that_a_hook_before_it_cut_inside_the_bbox_is_kept(runtime, detect):
    square = runtime.hooks.CoreMLHook("512x512")

    combined = detect(runtime.hooks.DetailerHookCombine(square, runtime.hook))

    assert [tuple(seg.crop_region) for seg in combined[1]] == [tuple(seg.crop_region) for seg in detect(square)[1]]


def test_a_mask_becomes_segs_cut_with_the_region_of_the_hook(runtime):
    mask = np.zeros((512, 768), dtype=np.float32)
    for x1, y1, x2, y2 in FOUND:
        mask[y1:y2, x1:x2] = 1

    segs = runtime.core.mask_to_segs(mask, False, 2.0, False, detailer_hook=runtime.hook)

    assert_each_seg_is_cut_with_its_region(segs, 2.0)
