"""SAM mask contracts with real Impact/ComfyUI code and controlled predictions.

Run separately from the wildcard tests, which stub Impact dependencies:
IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_sam_masks.py -v
No weights, GPU, or running server are required.
"""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = 24, 32
THRESHOLD = 0.93


@pytest.fixture(scope="module")
def core():
    comfy_root = os.environ.get("IMPACT_COMFY_ROOT")
    if not comfy_root:
        pytest.fail("Set IMPACT_COMFY_ROOT to the ComfyUI directory containing nodes.py and comfy/.", pytrace=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(comfy_root))
        patch.syspath_prepend(str(ROOT / "modules"))
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        module = importlib.import_module("impact.core")
        patch.setattr(module.utils.config, "cached_config", {"disable_gpu_opencv": True})
        yield module


def candidates(index):
    masks = np.zeros((3, HEIGHT, WIDTH), dtype=bool)
    for candidate in range(3):
        masks[candidate, 2 + index * 7:5 + index * 7, 3 + candidate * 5:6 + candidate * 5] = True
    return masks


def sam_model(core, monkeypatch, responses):
    model = SimpleNamespace(responses=iter(responses), calls=[])

    class Predictor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image, image_format):
            assert image.shape == (HEIGHT, WIDTH, 3)
            assert image.dtype == np.uint8
            assert image_format == "RGB"

        def predict(self, **kwargs):
            self.model.calls.append(kwargs)
            masks, scores = next(self.model.responses)
            return masks.copy(), np.array(scores), None

    monkeypatch.setattr(core, "SamPredictor", Predictor)
    model.sam_wrapper = core.SAMWrapper(model, is_auto_mode=False)
    return model


def segments(core, count):
    return ((HEIGHT, WIDTH), [
        core.SEG(None, np.ones((5, 18)), 1.0, (1, i * 7, 19, i * 7 + 5),
                 (1, i * 7, 19, i * 7 + 5), str(i))
        for i in range(count)
    ])


def detect(core, model, segs, *, hint="center-1", dilation=0, combined=False, bbox_expansion=0, negative="False"):
    function = core.make_sam_mask if combined else core.make_sam_mask_segmented
    return function(model, segs, torch.zeros((1, HEIGHT, WIDTH, 3)), hint,
                    dilation, THRESHOLD, bbox_expansion, 0.7, negative)


@pytest.mark.parametrize("scores,selected", [
    ([[.99, .6, .3], [.6, .99, .3]], [[0], [1]]),
    ([[.99, .95, .3], [.3, .95, .99]], [[0, 1], [1, 2]]),
    ([[.99, .95, .94], [.3, .99, .6]], [[0, 1, 2], [1]]),
    ([[.99, .6, .3], [.99, .95, .94]], [[0], [0, 1, 2]]),
    ([[.99, .95, .3], [.3, .99, .6], [.99, .95, .94]], [[0, 1], [1], [0, 1, 2]]),
    ([[.5, .92, .7], [.8, .7, .91]], [[1], [2]]),
    ([[.99, THRESHOLD, .3], [.6, THRESHOLD, .99]], [[0, 1], [1, 2]]),
    ([[.99, .95, .94], [.94, .95, .99]], [[0, 1, 2], [0, 1, 2]]),
], ids=["one-each", "two-each", "three-then-one", "one-then-three", "mixed-counts",
        "best-score-fallback", "threshold-equality", "all-three"])
def test_segmented_masks_preserve_detection_boundaries(core, monkeypatch, scores, selected):
    responses = [(candidates(i), values) for i, values in enumerate(scores)]
    model = sam_model(core, monkeypatch, responses)
    combined, batch = detect(core, model, segments(core, len(scores)))
    expected = np.stack([np.logical_or.reduce(candidates(i)[indices]) for i, indices in enumerate(selected)])

    # Check combined pixels first: correcting grouping must not alter selection.
    np.testing.assert_array_equal(combined.numpy(), np.logical_or.reduce(expected))
    assert combined.dtype == torch.float32
    assert combined.device.type == "cpu"
    assert batch.dtype == torch.uint8
    assert batch.device.type == "cpu"
    assert tuple(batch.shape) == (len(scores), 1, HEIGHT, WIDTH)
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
    # Follow the tutorial's mask-to-SEGS refinement boundary used by Detailer.
    input_segs = segments(core, len(scores))
    refined = core.apply_mask_to_each_seg(input_segs, batch)
    assert refined[0] == input_segs[0]
    assert len(refined[1]) == len(input_segs[1])
    for original, actual, mask in zip(input_segs[1], refined[1], expected, strict=True):
        x1, y1, x2, y2 = original.crop_region
        np.testing.assert_array_equal(actual.cropped_mask, mask[y1:y2, x1:x2])
        assert (actual.label, actual.bbox, actual.crop_region) == (original.label, original.bbox, original.crop_region)
    for call, seg in zip(model.calls, segments(core, len(scores))[1], strict=True):
        np.testing.assert_array_equal(call["box"], [seg.bbox])


@pytest.mark.parametrize("dilation", [-3, 0, 3])
def test_combined_pixels_and_dilation_are_unchanged(core, monkeypatch, dilation):
    responses = [(candidates(0), [.99, .95, .3]), (candidates(1), [.6, .3, .99])]
    expected = candidates(0)[0] | candidates(0)[1] | candidates(1)[2]
    if dilation:
        operation = cv2.dilate if dilation > 0 else cv2.erode
        expected = operation(expected.astype(np.float32), np.ones((3, 3), np.uint8))
    segs = segments(core, 2)
    model = sam_model(core, monkeypatch, responses)
    combined, batch = detect(core, model, segs, dilation=dilation)
    np.testing.assert_array_equal(combined.numpy(), expected)
    model = sam_model(core, monkeypatch, responses)
    combined_only = detect(core, model, segs, dilation=dilation, combined=True)
    np.testing.assert_array_equal(combined_only[0].numpy(), expected)
    # The existing dilation applies only to combined_mask, not batch_masks.
    np.testing.assert_array_equal(batch[:, 0].numpy(),
                                  [candidates(0)[0] | candidates(0)[1], candidates(1)[2]])


@pytest.mark.parametrize("negative", ["False", "Small"], ids=["positive-points", "small-negative"])
def test_mask_points_remains_one_joint_prediction(core, monkeypatch, negative):
    segs = segments(core, 2)
    segs[1][1] = segs[1][1]._replace(bbox=(1, 7, 5, 12))
    masks = candidates(0) | candidates(1)
    model = sam_model(core, monkeypatch, [(masks, [.99, .95, .3])])
    combined, batch = detect(core, model, segs, hint="mask-points", negative=negative)
    expected = masks[0] | masks[1]
    np.testing.assert_array_equal(combined.numpy(), expected)
    np.testing.assert_array_equal(batch.numpy(), expected[None, None])
    assert len(model.calls) == 1
    assert model.calls[0]["box"] is None
    np.testing.assert_array_equal(model.calls[0]["point_coords"], [[10, 2.5], [3, 9.5]])
    np.testing.assert_array_equal(model.calls[0]["point_labels"], [1, 0 if negative == "Small" else 1])


@pytest.mark.parametrize("hint", ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "none"])
def test_tutorial_hints_and_expanded_boxes_preserve_masks(core, monkeypatch, hint):
    # basic-refine.png uses bbox_expansion=240; clip to the non-square image.
    responses = [(candidates(0), [.99, .95, .3]), (candidates(1), [.3, .99, .2])]
    expected = [candidates(0)[0] | candidates(0)[1], candidates(1)[1]]
    for combined_only in (False, True):
        model = sam_model(core, monkeypatch, responses)
        result = detect(core, model, segments(core, 2), hint=hint, bbox_expansion=240, combined=combined_only)
        np.testing.assert_array_equal(result[0].numpy(), expected[0] | expected[1])
        if not combined_only:
            np.testing.assert_array_equal(result[1][:, 0].numpy(), expected)
        for index, call in enumerate(model.calls):
            y = 2.5 + 7 * index
            points = {"center-1": [[10, y]], "horizontal-2": [[32 / 3, y], [64 / 3, y]],
                      "vertical-2": [[10, 8], [10, 16]],
                      "rect-4": [[32 / 3, y], [64 / 3, y], [10, 8], [10, 16]],
                      "diamond-4": [[32 / 3, 8], [64 / 3, 8], [32 / 3, 16], [64 / 3, 16]],
                      "none": None}[hint]
            np.testing.assert_array_equal(call["box"], [[0, 0, WIDTH, HEIGHT]])
            np.testing.assert_array_equal(call["point_coords"], points)
            np.testing.assert_array_equal(call["point_labels"], None if points is None else [1] * len(points))
        assert len(model.calls) == 2


def test_mask_area_preserves_positive_negative_and_ignored_hints(core, monkeypatch):
    segs = segments(core, 2)
    for seg in segs[1]:
        seg.cropped_mask.fill(.5)
    segs[1][0].cropped_mask[0, 0], segs[1][0].cropped_mask[0, 3], segs[1][0].cropped_mask[3, 6] = .8, 0, .9
    segs[1][1].cropped_mask[0, 0], segs[1][1].cropped_mask[3, 0] = 0, .8
    model = sam_model(core, monkeypatch, [(candidates(0), [.99, .95, .3]), (candidates(1), [.3, .99, .2])])
    combined, batch = detect(core, model, segs, hint="mask-area", negative="Small")
    expected = [candidates(0)[0] | candidates(0)[1], candidates(1)[1]]
    np.testing.assert_array_equal(combined.numpy(), expected[0] | expected[1])
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
    assert len(model.calls) == 2
    for call, points, labels in zip(model.calls, [[[1, 0], [4, 0], [7, 3]], [[1, 7], [1, 10]]], [[1, 0, 1], [0, 1]], strict=True):
        np.testing.assert_array_equal(call["point_coords"], points)
        np.testing.assert_array_equal(call["point_labels"], labels)


@pytest.mark.parametrize("count", [0, 2], ids=["no-segments", "no-positive-scores"])
def test_empty_prediction_contract_is_preserved(core, monkeypatch, count):
    responses = [(candidates(i), [0., 0., 0.]) for i in range(count)]
    for combined_only in (False, True):
        model = sam_model(core, monkeypatch, responses)
        result = detect(core, model, segments(core, count), combined=combined_only)
        if combined_only:
            assert tuple(result.shape) == (1, HEIGHT, WIDTH)
            mask = result[0]
        else:
            mask, batch = result
            assert batch is None
        assert tuple(mask.shape) == (HEIGHT, WIDTH)
        assert mask.dtype == torch.float32
        assert torch.count_nonzero(mask) == 0
        assert len(model.calls) == count


@pytest.mark.parametrize("empty_index", [0, 1, 2], ids=["first", "middle", "last"])
def test_empty_prediction_does_not_mix_or_reorder_surviving_masks(core, monkeypatch, empty_index):
    responses = [(candidates(i), [0., 0., 0.] if i == empty_index else [.99, .95, .3])
                 for i in range(3)]
    expected = np.stack([candidates(i)[0] | candidates(i)[1] for i in range(3) if i != empty_index])
    model = sam_model(core, monkeypatch, responses)
    combined, batch = detect(core, model, segments(core, 3))
    np.testing.assert_array_equal(combined.numpy(), expected[0] | expected[1])
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
    assert len(model.calls) == 3


@pytest.mark.parametrize("count,zero_index", [(1, 0), (3, 0), (3, 1), (3, 2)],
                         ids=["only-mask", "first", "middle", "last"])
def test_selected_zero_mask_keeps_its_batch_entry(core, monkeypatch, count, zero_index):
    masks = [np.zeros_like(candidates(i)) if i == zero_index else candidates(i) for i in range(count)]
    model = sam_model(core, monkeypatch, [(mask, [.99, .3, .2]) for mask in masks])
    combined, batch = detect(core, model, segments(core, count))
    expected = np.stack([mask[0] for mask in masks])
    np.testing.assert_array_equal(combined.numpy(), np.logical_or.reduce(expected))
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
    assert batch.dtype == torch.uint8
    assert tuple(batch.shape) == (count, 1, HEIGHT, WIDTH)
    assert torch.count_nonzero(batch[zero_index]) == 0


def test_overlapping_candidates_union_within_each_detection(core, monkeypatch):
    first, second = np.zeros_like(candidates(0)), np.zeros_like(candidates(1))
    first[0, 2:10, 2:10] = True
    first[1, 6:14, 6:14] = True
    second[0, 7:15, 9:17] = True
    second[1, 10:18, 14:22] = True
    first[2, 20:22, 26:30] = second[2, 20:22, 26:30] = True
    expected = np.stack([first[0] | first[1], second[0] | second[1]])
    assert (first[0] & first[1]).any(), "Candidates within a detection must overlap"
    assert (expected[0] & expected[1]).any(), "Different detections must also overlap"
    model = sam_model(core, monkeypatch, [(first, [.99, .95, .3]), (second, [.99, .95, .3])])
    combined, batch = detect(core, model, segments(core, 2))
    np.testing.assert_array_equal(combined.numpy(), expected[0] | expected[1])
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
    assert set(batch.unique().tolist()) == {0, 1}


@pytest.mark.parametrize("dtype,value", [(torch.bool, True), (torch.uint8, 255)], ids=["boolean", "byte-mask"])
def test_single_candidate_wrapper_keeps_each_detection(core, dtype, value):
    class SingleCandidateWrapper:
        def prepare_device(self):
            self.index = 0

        def predict(self, image, points, labels, bbox, threshold):
            mask = candidates(self.index)[0]
            self.index += 1
            return [torch.from_numpy(mask).to(dtype) * value]

        def release_device(self):
            self.released = True

    wrapper = SingleCandidateWrapper()
    model = SimpleNamespace(sam_wrapper=wrapper)
    combined, batch = detect(core, model, segments(core, 2))
    np.testing.assert_array_equal(combined.numpy(), (candidates(0)[0] | candidates(1)[0]) * value)
    np.testing.assert_array_equal(batch[:, 0].numpy(), [candidates(0)[0], candidates(1)[0]])
    assert batch.dtype == torch.uint8
    assert wrapper.released


def test_device_is_released_when_prediction_fails(core, monkeypatch):
    model = sam_model(core, monkeypatch, [])
    released = []
    monkeypatch.setattr(model.sam_wrapper, "release_device", lambda: released.append(True))
    with pytest.raises(StopIteration):
        detect(core, model, segments(core, 1))
    assert released == [True]
