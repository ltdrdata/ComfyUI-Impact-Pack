import os
from types import SimpleNamespace

import pytest

IMAGE = 1024, 576
MIDDLE = 594, 183, 684, 302
NEAR_TOP = 366, 97, 453, 230
NEAR_BOTTOM = 366, 346, 453, 479
AT_RIGHT = 944, 100, 1024, 200
MODES = "512x512", "512x768", "768x512"


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
        from impact import hooks, utils

        def regions(box, mode, crop_factor=3.0):
            made = tuple(utils.make_crop_region(*IMAGE, box, crop_factor))
            return made, tuple(hooks.CoreMLHook(mode).post_crop_region(*IMAGE, box, made))
        yield SimpleNamespace(regions=regions)


def room(box, region):
    return box[0] - region[0], box[1] - region[1], region[2] - box[2], region[3] - box[3]


@pytest.mark.parametrize("box", [MIDDLE, NEAR_TOP, NEAR_BOTTOM, AT_RIGHT])
@pytest.mark.parametrize("mode", MODES)
def test_the_crop_region_gets_the_shape_of_the_mode_inside_the_region_it_was(runtime, box, mode):
    made, fitted = runtime.regions(box, mode)

    width, height = (int(side) for side in mode.split("x"))
    assert abs((fitted[2] - fitted[0]) / (fitted[3] - fitted[1]) - width / height) < 0.02
    assert made[0] <= fitted[0] and made[1] <= fitted[1] and fitted[2] <= made[2] and fitted[3] <= made[3]


@pytest.mark.parametrize("crop_factor", [3.0, 1.5])
@pytest.mark.parametrize("mode", MODES)
def test_a_box_in_the_middle_of_its_crop_region_stays_in_the_middle(runtime, mode, crop_factor):
    made, fitted = runtime.regions(MIDDLE, mode, crop_factor)

    left, top, right, bottom = room(MIDDLE, fitted)
    assert abs(left - right) <= 3 and abs(top - bottom) <= 3


@pytest.mark.parametrize("box", [NEAR_TOP, NEAR_BOTTOM])
@pytest.mark.parametrize("mode", MODES)
def test_a_box_that_a_border_pushed_off_the_middle_keeps_its_place_in_the_crop_region(runtime, mode, box):
    made, fitted = runtime.regions(box, mode)

    before, after = room(box, made), room(box, fitted)
    assert min(after) >= 0
    assert abs(after[1] - (after[1] + after[3]) * before[1] / (before[1] + before[3])) < 1.5


@pytest.mark.parametrize("mode", ["512x768", "768x768"])
def test_a_box_that_ends_where_its_crop_region_ends_stays_whole(runtime, mode):
    made, fitted = runtime.regions(AT_RIGHT, mode)

    left, top, right, bottom = room(AT_RIGHT, fitted)
    assert right == 0 and left >= 0


@pytest.mark.parametrize("mode", MODES)
def test_a_crop_region_that_is_the_box_is_cut_by_the_same_amount_on_both_sides(runtime, mode):
    made, fitted = runtime.regions(MIDDLE, mode, crop_factor=1.0)

    left, top, right, bottom = room(MIDDLE, fitted)
    assert made == MIDDLE and abs(left - right) <= 1 and abs(top - bottom) <= 1
