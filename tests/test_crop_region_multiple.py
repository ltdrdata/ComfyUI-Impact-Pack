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
        from impact import core, hooks, impact_pack, segs_nodes, utils
        yield SimpleNamespace(
            fit=utils.crop_region_to_multiple,
            hook=lambda multiple=64: impact_pack.CropRegionMultipleDetailerHookProvider.execute(multiple).result[0],
            apply=lambda segs, hook: segs_nodes.SEGSApplyCropRegionHook.execute(segs, hook).result[0],
            tiles=lambda image, **inputs: segs_nodes.MakeTileSEGS.execute(image, min_overlap=5, filter_segs_dilation=20, **inputs).result[0],
            with_images=lambda segs, image: segs_nodes.DefaultImageForSEGS.execute(segs, image, True).result[0],
            video_segs=lambda mask: segs_nodes.MaskToSEGS_for_AnimateDiff.execute(mask, False, 1.3, False, 1).result[0],
            core=core, hooks=hooks, detailer=impact_pack.DetailerForEach)


IMAGE = torch.rand((1, 300, 500, 3), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("size, region, multiple, expected", [
    ((1000, 700), [100, 100, 228, 292], 64, [100, 100, 228, 292]),
    ((1000, 700), [100, 100, 200, 250], 64, [86, 79, 214, 271]),
    ((1000, 700), [0, 0, 100, 150], 64, [0, 0, 128, 192]),
    ((1000, 700), [900, 550, 1000, 700], 64, [872, 508, 1000, 700]),
    ((300, 200), [0, 0, 300, 200], 64, [22, 4, 278, 196]),
    ((50, 700), [0, 100, 50, 200], 64, [0, 86, 50, 214]),
    ((40, 30), [5, 5, 35, 25], 64, [5, 5, 35, 25]),
    ((1000, 700), [100, 100, 203, 250], 8, [100, 99, 204, 251]),
], ids=["already a multiple", "grows on both sides", "at the left and top border", "at the right and bottom border",
        "image too small for the grown size", "image narrower than the multiple", "image smaller than the multiple", "a multiple of 8"])
def test_a_crop_region_becomes_the_next_multiple_inside_the_image(runtime, size, region, multiple, expected):
    assert runtime.fit(*size, region, multiple) == expected


@pytest.mark.parametrize("start", [0, 13, 140, 230])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 150])
def test_a_grown_crop_region_holds_the_region_it_was(runtime, start, length):
    region = [start, start // 2, min(start + length, 300), min(start // 2 + length, 200)]

    x1, y1, x2, y2 = runtime.fit(300, 200, region, 64)

    assert (x2 - x1) % 64 == 0 and (y2 - y1) % 64 == 0
    assert 0 <= x1 <= region[0] and region[2] <= x2 <= 300
    assert 0 <= y1 <= region[1] and region[3] <= y2 <= 200


@pytest.mark.parametrize("multiple, region, expected", [(64, [100, 100, 200, 250], [86, 79, 214, 271]),
                                                       (8, [100, 100, 203, 250], [100, 99, 204, 251])])
def test_the_hook_of_the_provider_changes_the_crop_region_a_detector_asks_it_for(runtime, multiple, region, expected):
    hook = runtime.hook(multiple)

    assert hook.post_crop_region(1000, 700, (120, 120, 180, 230), region) == expected
    assert runtime.hooks.DetailerHookCombine(runtime.hooks.DetailerHook(), hook).post_crop_region(1000, 700, (120, 120, 180, 230), region) == expected


def test_a_detector_given_the_hook_gives_crop_regions_of_the_multiple(runtime):
    mask = np.zeros((300, 500), dtype=np.float32)
    mask[100:150, 200:260] = 1.0

    shape, (seg,) = runtime.core.mask_to_segs(mask, False, 1.5, False, detailer_hook=runtime.hook())

    x1, y1, x2, y2 = seg.crop_region
    assert (x2 - x1, y2 - y1) == (128, 128)
    assert seg.cropped_mask.shape == (128, 128) and np.count_nonzero(seg.cropped_mask) == 50 * 60


def placed(shape, seg):
    mask = torch.as_tensor(seg.cropped_mask)
    canvas = torch.zeros((*mask.shape[:-2], *shape))
    x1, y1, x2, y2 = seg.crop_region
    canvas[..., y1:y2, x1:x2] = mask
    return canvas


@pytest.mark.parametrize("irregularity", [0, 0.5])
def test_each_tile_of_make_tile_segs_gets_a_crop_region_of_the_multiple_with_its_mask_in_place(runtime, irregularity):
    before = runtime.with_images(runtime.tiles(IMAGE, bbox_size=128, crop_factor=1.3, mask_irregularity=irregularity), IMAGE)
    assert all((x2 - x1) % 64 and (y2 - y1) % 64 for x1, y1, x2, y2 in (seg.crop_region for seg in before[1]))

    after = runtime.apply(before, runtime.hook())

    assert after[0] == (300, 500) and len(after[1]) == len(before[1]) > 1
    for old, new in zip(before[1], after[1]):
        x1, y1, x2, y2 = new.crop_region
        bx1, by1, bx2, by2 = new.bbox
        assert (x2 - x1) % 64 == 0 and (y2 - y1) % 64 == 0
        assert 0 <= x1 <= bx1 and bx2 <= x2 <= 500 and 0 <= y1 <= by1 and by2 <= y2 <= 300
        assert type(new.cropped_mask) is type(old.cropped_mask) and new.cropped_mask.shape == (y2 - y1, x2 - x1)
        assert np.count_nonzero(new.cropped_mask) == np.count_nonzero(old.cropped_mask)
        assert torch.equal(placed((300, 500), new), placed((300, 500), old))
        assert new.cropped_image is None
        assert (tuple(new.bbox), new.label, new.confidence) == (tuple(old.bbox), old.label, old.confidence)


def test_each_frame_of_the_mask_of_video_segs_stays_in_place(runtime):
    mask = torch.zeros((2, 300, 500))
    mask[0, 100:150, 200:260] = 1.0
    mask[1, 110:160, 210:250] = 1.0
    before = runtime.video_segs(mask)

    after = runtime.apply(before, runtime.hook())

    (old,), (new,) = before[1], after[1]
    x1, y1, x2, y2 = new.crop_region
    assert (x2 - x1) % 64 == 0 and (y2 - y1) % 64 == 0
    assert isinstance(new.cropped_mask, torch.Tensor) and new.cropped_mask.shape == (2, y2 - y1, x2 - x1)
    assert torch.equal(placed((300, 500), new), placed((300, 500), old))


def test_a_seg_whose_crop_region_the_hook_keeps_keeps_its_image(runtime):
    before = runtime.with_images(runtime.tiles(IMAGE, bbox_size=128, crop_factor=1.5), IMAGE)

    after = runtime.apply(before, runtime.hook())

    assert len(after[1]) == len(before[1]) > 1
    for old, new in zip(before[1], after[1]):
        assert list(new.crop_region) == list(old.crop_region)
        assert new.cropped_image is not None and torch.equal(new.cropped_image, old.cropped_image)


def test_the_detailer_crops_each_tile_at_its_new_crop_region_from_its_image(runtime):
    segs = runtime.apply(runtime.with_images(runtime.tiles(IMAGE, bbox_size=128, crop_factor=1.3), IMAGE), runtime.hook())

    *_, detailed = runtime.detailer.do_detail(IMAGE, segs, "DUMMY", None, None, 512, True, 1024, 0, 20, 8.0, "euler", "normal",
                                              "", "", 0.5, 5, True, True)

    assert len(detailed[1]) == len(segs[1]) > 1
    for seg in detailed[1]:
        x1, y1, x2, y2 = seg.crop_region
        assert (x2 - x1) % 64 == 0 and (y2 - y1) % 64 == 0
        np.testing.assert_allclose(seg.cropped_image, IMAGE[:, y1:y2, x1:x2].numpy(), rtol=0, atol=1e-5)
