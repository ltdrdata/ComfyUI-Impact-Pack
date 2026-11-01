"""CPU integration checks; run separately from the stubbed wildcard tests.

IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_mask_to_segs.py -v
Only unused server registration is substituted.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    comfy_root = os.environ.get("IMPACT_COMFY_ROOT")
    if not comfy_root:
        pytest.fail("Set IMPACT_COMFY_ROOT to the ComfyUI directory containing nodes.py and comfy/.", pytrace=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(comfy_root)
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        import nodes
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from comfy_extras.nodes_mask import ImageToMask
        from impact import core, segs_nodes
        assert Path(core.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
        yield SimpleNamespace(core=core, segs_nodes=segs_nodes, nodes=nodes, ImageToMask=ImageToMask)


@pytest.mark.parametrize("value,expected_count", [(-1.0, 0), (-0.125, 0), (256 / 255, 1), (2.0, 1), (65504.0, 1)])
@pytest.mark.parametrize("contour_fill", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_out_of_range_rectangle(runtime, value, expected_count, contour_fill, dtype):
    mask = torch.zeros((1, 32, 48), dtype=dtype)
    mask[:, 4:20, 8:32] = value
    before = mask.clone()
    shape, segments = runtime.segs_nodes.MaskToSEGS.doit(mask, False, 1.0, False, 1, contour_fill)[0]
    assert shape == (32, 48)
    assert len(segments) == expected_count
    for seg in segments:
        assert seg.bbox == (8, 4, 32, 20)
        assert tuple(seg.crop_region) == seg.bbox
        np.testing.assert_array_equal(seg.cropped_mask, np.ones((16, 24), dtype=np.float32))
    torch.testing.assert_close(mask, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("combined,contour_fill", [(False, False), (False, True), (True, False)])
def test_normal_masks_preserve_values_geometry_and_batch_order(runtime, dtype, batched, combined, contour_fill):
    mask = np.zeros((2, 32, 48), dtype=dtype)
    mask[0, 4:20, 8:32] = 0.375
    mask[1, 8:24, 16:40] = 0.75
    if not batched:
        mask = mask[0]
    before = mask.copy()
    shape, segments = runtime.core.mask_to_segs(mask, combined, 1.0, False, 1, is_contour=contour_fill)
    assert shape == (32, 48)
    assert len(segments) == (2 if batched else 1)
    for index, seg in enumerate(segments):
        x, y = 8 + 8 * index, 4 + 4 * index
        # The combined path's historical max coordinates are inclusive.
        width, height = (23, 15) if combined else (24, 16)
        assert seg.bbox == (x, y, x + width, y + height)
        assert tuple(seg.crop_region) == seg.bbox
        value = 1.0 if contour_fill else (0.375, 0.75)[index]
        np.testing.assert_array_equal(seg.cropped_mask, np.full((height, width), value))
        assert seg.cropped_image is None and seg.confidence == 1.0 and seg.label == 'A'
        assert seg.control_net_wrapper is None
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("contour_fill", [False, True])
def test_adjacent_values_at_quantization_boundary(runtime, dtype, contour_fill):
    threshold = dtype(1 / 255)
    values = [np.nextafter(threshold, dtype(0)), threshold, np.nextafter(threshold, dtype(1))]
    mask = np.zeros((24, 64), dtype=dtype)
    for index, value in enumerate(values):
        mask[4:20, 4 + index * 20:16 + index * 20] = value
    before = mask.copy()
    shape, segments = runtime.core.mask_to_segs(mask, False, 1.0, False, 1, is_contour=contour_fill)
    assert shape == (24, 64)
    segments.sort(key=lambda seg: seg.bbox)
    # Existing arithmetic in each input dtype excludes only the predecessor.
    assert [seg.bbox for seg in segments] == [(24, 4, 36, 20), (44, 4, 56, 20)]
    for seg, value in zip(segments, values[1:]):
        expected = 1.0 if contour_fill else value
        expected_dtype = np.float32 if contour_fill else np.result_type(dtype, np.float32)
        assert seg.cropped_mask.dtype == expected_dtype
        np.testing.assert_array_equal(seg.cropped_mask, np.full((16, 12), expected, dtype=expected_dtype))
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("contour_fill", [False, True])
@pytest.mark.parametrize("tensor_input", [False, True], ids=["numpy_core", "tensor_node"])
def test_mixed_ranges_in_every_batch(runtime, dtype, contour_fill, tensor_input):
    values = [(0.375, -1.0, 256 / 255), (-0.125, 256 / 255, 0.75), (256 / 255, 0.5, -1.0)]
    mask = np.zeros((3, 24, 64), dtype=dtype)
    for batch, row in enumerate(values):
        for column, value in enumerate(row):
            mask[batch, 4:20, 4 + column * 20:16 + column * 20] = value
    before = mask.copy()
    if tensor_input:
        shape, segments = runtime.segs_nodes.MaskToSEGS.doit(
            torch.from_numpy(mask), False, 1.0, False, 1, contour_fill)[0]
    else:
        shape, segments = runtime.core.mask_to_segs(mask, False, 1.0, False, 1, is_contour=contour_fill)
    assert shape == (24, 64)
    assert len(segments) == 6
    for batch, row in enumerate(values):
        expected = [(column, value) for column, value in enumerate(row) if value > 0]
        items = sorted(segments[batch * 2:batch * 2 + 2], key=lambda seg: seg.bbox)
        for seg, (column, value) in zip(items, expected):
            assert seg.bbox == (4 + column * 20, 4, 16 + column * 20, 20)
            assert tuple(seg.crop_region) == seg.bbox
            expected_value = 1.0 if contour_fill or value > 1 else value
            np.testing.assert_array_equal(seg.cropped_mask, np.full((16, 12), expected_value))
    np.testing.assert_array_equal(mask, before)


def test_empty_masks_and_combined_path_remain_unchanged(runtime):
    assert runtime.core.mask_to_segs(None, False, 1.0, False) == ([],)
    assert runtime.core.mask_to_segs(np.zeros((2, 32, 48)), False, 1.0, False) == ((32, 48), [])
    mask = np.zeros((32, 48), dtype=np.float32)
    mask[4:20, 8:32] = 2.0
    _, segments = runtime.core.mask_to_segs(mask, True, 1.0, False)
    np.testing.assert_array_equal(segments[0].cropped_mask, np.full((15, 23), 2.0))


@pytest.mark.parametrize("contour_fill", [False, True])
def test_bicubic_image_to_mask(runtime, contour_fill):
    image = torch.zeros((1, 64, 64, 3))
    image[:, 16:48, 16:48, :] = 1
    scaled = runtime.nodes.ImageScaleBy().upscale(image, "bicubic", 4.0)[0]
    mask = runtime.ImageToMask.execute(scaled, "red")[0]
    assert mask.min() < 0 and mask.max() > 1
    before = mask.clone()
    shape, segments = runtime.segs_nodes.MaskToSEGS.doit(mask, False, 1.0, False, 10, contour_fill)[0]
    assert shape == (256, 256)
    # Pixel baseline: PyTorch 2.10.0 and OpenCV 4.13.0.
    # Investigate dependency changes before updating these expectations.
    assert [seg.bbox for seg in segments] == [(59, 59, 197, 197)]
    if contour_fill:
        assert np.count_nonzero(segments[0].cropped_mask) == 17460
    else:
        assert 0 <= segments[0].cropped_mask.min() <= segments[0].cropped_mask.max() <= 1
    torch.testing.assert_close(mask, before, rtol=0, atol=0)


@pytest.mark.parametrize("bbox_fill", [False, True])
@pytest.mark.parametrize("contour_fill", [False, True])
def test_tutorial_holes_and_expanded_crop(runtime, bbox_fill, contour_fill):
    # mask_to_segs.png explains holes/fill; segs-manipulation.png uses crop_factor=3.
    outer = np.zeros((64, 80), dtype=np.float32)
    outer[4:28, 4:28] = 1
    for y in (4, 24):
        for x in (4, 24):
            outer[y:y + 4, x:x + 4] = 0
    mask = outer * 0.375
    mask[10:22, 10:22] = 0
    mask[14:18, 14:18] = 0.75  # Nested island must not become an extra SEG.
    before = mask.copy()
    segs = runtime.segs_nodes.MaskToSEGS.doit(
        torch.from_numpy(mask).unsqueeze(0), False, 3.0, bbox_fill, 10, contour_fill)[0]
    assert segs[0] == (64, 80) and len(segs[1]) == 1
    seg = segs[1][0]
    assert seg.bbox == (4, 4, 28, 28)
    assert tuple(seg.crop_region) == (0, 0, 72, 64)
    expected = outer.copy() if contour_fill else before.copy()
    if bbox_fill:
        expected[4:28, 4:28] = 1
    np.testing.assert_array_equal(seg.cropped_mask, expected[:, :72])
    # The tutorial's mask-list consumer restores image coordinates and quantizes.
    masks = runtime.segs_nodes.SEGSToMaskList().doit(segs)[0]
    assert len(masks) == 1 and masks[0].shape == (1, 64, 80)
    quantized = expected.copy()
    quantized[expected == 0.375] = 95 / 255
    quantized[expected == 0.75] = 191 / 255
    np.testing.assert_array_equal(masks[0][0].numpy(), quantized)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("drop_size", [10, 50])
def test_tutorial_drop_size_boundary(runtime, drop_size):
    # Values saved in mask_to_segs.png and segs-manipulation.png.
    size = drop_size + 1
    mask = np.zeros((size + 8, 3 * (size + 8)), dtype=np.float32)
    for index, (height, width) in enumerate([(size, drop_size), (drop_size, size), (size, size)]):
        x = 4 + index * (size + 8)
        mask[4:4 + height, x:x + width] = 1
    segs = runtime.segs_nodes.MaskToSEGS.doit(torch.from_numpy(mask), False, 1.0, False, drop_size)[0]
    x = 4 + 2 * (size + 8)
    assert [seg.bbox for seg in segs[1]] == [(x, 4, x + size, 4 + size)]


def test_tutorial_combined_disconnected_regions(runtime):
    # maskpointer.json stores both combined modes with crop_factor=2.
    mask = np.zeros((64, 96), dtype=np.float32)
    mask[8:24, 8:24] = 0.5
    mask[32:48, 56:72] = 1
    before = mask.copy()
    separate = runtime.segs_nodes.MaskToSEGS.doit(torch.from_numpy(mask), False, 2.0, False, 10)[0]
    assert sorted(seg.bbox for seg in separate[1]) == [(8, 8, 24, 24), (56, 32, 72, 48)]
    combined = runtime.segs_nodes.MaskToSEGS.doit(torch.from_numpy(mask), True, 2.0, False, 10)[0]
    assert len(combined[1]) == 1 and combined[1][0].bbox == (8, 8, 71, 47)
    assert tuple(combined[1][0].crop_region) == (0, 0, 96, 64)
    np.testing.assert_array_equal(combined[1][0].cropped_mask, before)
    np.testing.assert_array_equal(mask, before)
