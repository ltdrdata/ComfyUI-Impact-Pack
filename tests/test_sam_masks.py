import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import safetensors.torch
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


@pytest.fixture(scope="module")
def detectors(core):
    from aiohttp import web
    from server import PromptServer

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=Mock()), raising=False)
        yield importlib.import_module("impact.detectors")


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

    np.testing.assert_array_equal(combined.numpy(), np.logical_or.reduce(expected))
    assert combined.dtype == torch.float32
    assert combined.device.type == "cpu"
    assert batch.dtype == torch.uint8
    assert batch.device.type == "cpu"
    assert tuple(batch.shape) == (len(scores), 1, HEIGHT, WIDTH)
    np.testing.assert_array_equal(batch[:, 0].numpy(), expected)
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


def tiny_sam(checkpoint=None):
    """The checkpoint contract of the segment_anything builders: no checkpoint leaves the weights unloaded, a checkpoint goes through torch.load."""
    model = torch.nn.Linear(3, 2)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            model.load_state_dict(torch.load(f))
    return model


@pytest.mark.parametrize("name", ["sam_vit_b.pth", "sam_vit_b.safetensors", "SAM_VIT_B.SAFETENSORS"])
def test_load_sam_reads_the_weights_of_the_file(core, monkeypatch, tmp_path, name):
    monkeypatch.setattr(core, "sam_model_registry", {"vit_b": tiny_sam})
    weights = {"weight": torch.arange(6.0).reshape(2, 3), "bias": torch.tensor([7.0, 8.0])}
    path = str(tmp_path / name)
    if name.endswith(".pth"):
        torch.save(weights, path)
    else:
        safetensors.torch.save_file(weights, path)
    loaded = core.load_sam("vit_b", path).state_dict()
    assert loaded.keys() == weights.keys()
    for key, value in weights.items():
        assert torch.equal(loaded[key], value)


def test_load_sam_rejects_a_safetensors_file_of_another_model(core, monkeypatch, tmp_path):
    monkeypatch.setattr(core, "sam_model_registry", {"vit_b": tiny_sam})
    path = str(tmp_path / "other.safetensors")
    safetensors.torch.save_file({"weight": torch.zeros(2, 3)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        core.load_sam("vit_b", path)


@pytest.mark.parametrize("combined", [False, True], ids=["segmented", "combined"])
def test_a_sam2_model_is_taken_by_both_detectors(core, monkeypatch, combined):
    model = core.SAM2Wrapper(config=None, modelname=None, is_auto_mode=False)
    monkeypatch.setattr(model, "predict", lambda image, points, plabs, bbox, threshold: [candidates(0)[0]])
    result = detect(core, model, segments(core, 2), combined=combined)

    if combined:
        np.testing.assert_array_equal(result[0].numpy(), candidates(0)[0])
    else:
        mask, batch = result
        np.testing.assert_array_equal(mask.numpy(), candidates(0)[0])
        np.testing.assert_array_equal(batch.numpy()[:, 0], np.stack([candidates(0)[0]] * 2))


@pytest.mark.parametrize("device_mode", ["AUTO", "Prefer GPU", "CPU"])
def test_samloader_returns_a_sam2_model_that_a_detector_takes_in_every_device_mode(core, detectors, monkeypatch, tmp_path, device_mode):
    pack = importlib.import_module("impact.impact_pack")
    file = tmp_path / "sam2.1_hiera_tiny.pt"
    file.write_bytes(b"")
    monkeypatch.setattr(pack.folder_paths, "get_full_path", lambda folder, name: str(file))

    model, = pack.SAMLoader().load_model(file.name, device_mode)

    monkeypatch.setattr(model, "predict", lambda image, points, plabs, bbox, threshold: [candidates(0)[0]])
    np.testing.assert_array_equal(detect(core, model, segments(core, 1), combined=True)[0].numpy(), candidates(0)[0])


class TrackedNowhere(torch.nn.Identity):
    image_size, device = 16, torch.device("cpu")

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        pass

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points, labels, box):
        inference_state["obj_ids"].append(obj_id)

    def propagate_in_video(self, inference_state):
        for frame in range(inference_state["num_frames"]):
            yield frame, inference_state["obj_ids"], torch.zeros(len(inference_state["obj_ids"]), 1, self.image_size, self.image_size)


@pytest.mark.parametrize("device_mode", ["AUTO", "Prefer GPU", "CPU"])
@pytest.mark.parametrize("used_for", ["an image", "a video"])
def test_a_sam2_model_is_built_on_the_cpu_and_goes_to_the_gpu_only_as_its_device_mode_says(core, monkeypatch, used_for, device_mode):
    sampling_device = torch.device("cuda:1")
    built_on, moved_to = [], []

    def builder(model):
        def build(config, modelname, device="cuda"):
            built_on.append(torch.device(device))
            return model
        return build

    monkeypatch.setattr(core, "is_sam2_available", True)
    monkeypatch.setattr(core, "build_sam2", builder(torch.nn.Identity()), raising=False)
    monkeypatch.setattr(core, "build_sam2_video_predictor", builder(TrackedNowhere()), raising=False)
    monkeypatch.setattr(core, "SAM2ImagePredictor", lambda model: SimpleNamespace(
        model=model, set_image=lambda image: None, predict=lambda **prompt: ([candidates(0)[0]], [1.0], None)), raising=False)
    monkeypatch.setattr(core.comfy.model_management, "get_torch_device", lambda: sampling_device)
    mover = SimpleNamespace(to_device=lambda model, device: moved_to.append(device))
    model = core.SAM2Wrapper(config=None, modelname=None, is_auto_mode=device_mode == "AUTO", safe_to_gpu=mover, device_mode=device_mode)

    if used_for == "an image":
        model.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), [[1, 1]], [1], None, THRESHOLD)
    else:
        model.predict_video_segs(torch.zeros(2, HEIGHT, WIDTH, 3), segments(core, 1))

    assert built_on == [torch.device("cpu")]
    assert moved_to == ([] if device_mode == "CPU" else [sampling_device])


@pytest.mark.parametrize("device_mode", ["AUTO", "Prefer GPU", "CPU"])
def test_samloader_moves_a_sam_model_to_the_gpu_at_once_only_with_prefer_gpu(core, detectors, monkeypatch, tmp_path, device_mode):
    pack = importlib.import_module("impact.impact_pack")
    file = tmp_path / "sam_vit_b.pth"
    file.write_bytes(b"")
    sampling_device = torch.device("cuda:1")
    moved_to = []
    monkeypatch.setattr(pack.folder_paths, "get_full_path", lambda folder, name: str(file))
    monkeypatch.setattr(core, "load_sam", lambda kind, path: torch.nn.Identity())
    monkeypatch.setattr(core, "SafeToGPU", lambda size: SimpleNamespace(to_device=lambda model, device: moved_to.append(device)))
    monkeypatch.setattr(pack.comfy.model_management, "get_torch_device", lambda: sampling_device)

    pack.SAMLoader().load_model(file.name, device_mode)

    assert moved_to == ([sampling_device] if device_mode == "Prefer GPU" else [])


def test_an_object_that_samloader_did_not_return_is_refused(core):
    for combined in (False, True):
        with pytest.raises(Exception, match="Invalid SAMLoader is connected"):
            detect(core, SimpleNamespace(), segments(core, 1), combined=combined)


def sam3_model(core, monkeypatch, tracks="packed masks"):
    """A SAM3 model as SAMLoader returns it, on a stand-in for ComfyUI's model. Its mask is the inside of the prompted box, or nothing without a box.
    ComfyUI's tracker returns packed masks for SAM3.1 and logits for SAM3."""
    model = SimpleNamespace(calls=[], loaded=[])

    def forward_segment(frame, point_inputs=None, box_inputs=None):
        model.calls.append({"frame": tuple(frame.shape), "points": point_inputs, "box": box_inputs})
        logits = torch.full((1, 1, 1008, 1008), -10.0)
        if box_inputs is not None:
            (x1, y1), (x2, y2) = box_inputs[0].round().int().tolist()
            logits[..., y1:y2, x1:x2] = 10.0
        return logits

    def forward_video(images, initial_masks, pbar=None, target_device=None, target_dtype=None):
        from comfy.ldm.sam3.tracker import pack_masks
        model.video = {"frames": tuple(images.shape), "initial": initial_masks.clone()}
        tracked = torch.stack([torch.roll(initial_masks[:, 0], frame, dims=-1) for frame in range(images.shape[0])])
        if tracks == "logits":
            return torch.where(tracked > 0, 0.2, -10.0)
        return {"packed_masks": pack_masks(tracked), "n_frames": images.shape[0], "scores": [1.0] * initial_masks.shape[0]}

    model.model = SimpleNamespace(get_dtype=lambda: torch.float32,
                                  diffusion_model=SimpleNamespace(forward_segment=forward_segment, forward_video=forward_video))
    monkeypatch.setattr(core.comfy.model_management, "load_model_gpu", model.loaded.append)
    return core.SAM3Wrapper(model)


@pytest.mark.parametrize("combined", [False, True], ids=["segmented", "combined"])
def test_sam3_masks_each_detection_at_the_size_of_the_image(core, monkeypatch, combined):
    sam = sam3_model(core, monkeypatch)
    model = sam.model
    result = detect(core, sam, segments(core, 2), combined=combined)

    expected = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    for i in range(2):
        expected[i, i * 7:i * 7 + 5, 1:19] = True

    if combined:
        np.testing.assert_array_equal(result[0].numpy(), np.logical_or.reduce(expected))
    else:
        mask, batch = result
        np.testing.assert_array_equal(mask.numpy(), np.logical_or.reduce(expected))
        np.testing.assert_array_equal(batch.numpy()[:, 0], expected)

    assert model.loaded == [model]
    assert [call["frame"] for call in model.calls] == [(1, 3, 1008, 1008)] * 2
    assert model.calls[1]["box"].tolist() == [[[31.5, 294.0], [598.5, 504.0]]]
    assert model.calls[1]["points"]["point_coords"].tolist() == [[[315.0, 399.0]]]
    assert model.calls[1]["points"]["point_labels"].tolist() == [[1]]


def test_sam3_is_not_asked_without_a_prompt(core, monkeypatch):
    sam = sam3_model(core, monkeypatch)
    result = detect(core, sam, segments(core, 0), hint="mask-points", combined=True)

    assert tuple(result.shape) == (1, HEIGHT, WIDTH)
    assert torch.count_nonzero(result) == 0
    assert sam.model.calls == []


@pytest.mark.parametrize("tracks", ["packed masks", "logits"])
def test_the_sam3_video_detector_tracks_each_detection_of_the_first_frame(core, detectors, monkeypatch, tracks):
    sam = sam3_model(core, monkeypatch, tracks)
    model = sam.model
    bbox_detector = SimpleNamespace(detect=lambda image, threshold, dilation, crop_factor, drop_size: segments(core, 2))
    frames = torch.zeros((3, HEIGHT, WIDTH, 3))

    (shape, segs), = detectors.SAM3VideoDetectorSEGS.execute(frames, bbox_detector, sam, 0.5, 3.0, 1).result

    assert model.video["frames"] == (3, 3, HEIGHT, WIDTH)
    boxes = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    for i in range(2):
        boxes[i, i * 7:i * 7 + 5, 1:19] = True
    np.testing.assert_array_equal(model.video["initial"][:, 0].numpy() > 0, boxes)

    assert shape == (HEIGHT, WIDTH)
    assert len(segs) == 2
    for i, seg in enumerate(segs):
        x1, y1, x2, y2 = seg.crop_region
        tracked = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        tracked[:, y1:y2, x1:x2] = seg.cropped_mask.numpy()
        np.testing.assert_array_equal(tracked, np.stack([np.roll(boxes[i], frame, axis=-1) for frame in range(3)]))


def test_the_sam3_video_detector_tracks_back_from_the_last_frame_when_the_first_has_no_detection(core, detectors, monkeypatch):
    sam = sam3_model(core, monkeypatch)
    last_frame = torch.ones((HEIGHT, WIDTH, 3))
    frames = torch.cat([torch.zeros((2, HEIGHT, WIDTH, 3)), last_frame.unsqueeze(0)])
    bbox_detector = SimpleNamespace(detect=lambda image, threshold, dilation, crop_factor, drop_size: segments(core, 1 if torch.equal(image[0], last_frame) else 0))

    (shape, segs), = detectors.SAM3VideoDetectorSEGS.execute(frames, bbox_detector, sam, 0.5, 3.0, 1).result

    box = np.zeros((HEIGHT, WIDTH), dtype=bool)
    box[0:5, 1:19] = True
    x1, y1, x2, y2 = segs[0].crop_region
    tracked = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
    tracked[:, y1:y2, x1:x2] = segs[0].cropped_mask.numpy()
    np.testing.assert_array_equal(tracked, np.stack([np.roll(box, 2 - frame, axis=-1) for frame in range(3)]))


def test_the_sam3_video_detector_returns_no_segs_without_a_detection(core, detectors, monkeypatch):
    sam = sam3_model(core, monkeypatch)
    bbox_detector = SimpleNamespace(detect=lambda image, threshold, dilation, crop_factor, drop_size: segments(core, 0))

    (shape, segs), = detectors.SAM3VideoDetectorSEGS.execute(torch.zeros((3, HEIGHT, WIDTH, 3)), bbox_detector, sam, 0.5, 3.0, 1).result

    assert shape == (HEIGHT, WIDTH)
    assert segs == []


def test_each_video_detector_takes_only_its_own_model(core, detectors, monkeypatch):
    frames = torch.zeros((2, HEIGHT, WIDTH, 3))
    sam2 = core.SAM2Wrapper(config=None, modelname=None, is_auto_mode=False)
    for model in (sam_model(core, monkeypatch, []), sam2):
        with pytest.raises(Exception, match="a SAM3 model must be provided"):
            detectors.SAM3VideoDetectorSEGS.execute(frames, None, model, 0.5, 3.0, 1)
    for model in (sam_model(core, monkeypatch, []), sam3_model(core, monkeypatch)):
        with pytest.raises(Exception, match="a SAM2 model must be provided"):
            detectors.SAM2VideoDetectorSEGS.doit(None, model, frames, 0.5, 0.5, 3.0, 1)


def test_the_sam2_video_detector_cuts_the_tracked_logits_at_its_threshold(core, detectors, monkeypatch):
    sam2 = core.SAM2Wrapper(config=None, modelname=None, is_auto_mode=False)
    logits = torch.full((3, HEIGHT, WIDTH, 1), -30.0)
    logits[:, 4:12, 6:20] = 10.0
    logits[:, 6:10, 8:16] = 40.0
    monkeypatch.setattr(sam2, "predict_video_segs", lambda frames, segs: {0: logits.clone()})
    bbox_detector = SimpleNamespace(detect=lambda image, threshold, dilation, crop_factor, drop_size: segments(core, 1))

    def tracked(threshold):
        ((shape, segs), ) = detectors.SAM2VideoDetectorSEGS.doit(bbox_detector, sam2, torch.zeros((3, HEIGHT, WIDTH, 3)), 0.5, threshold, 3.0, 1)
        x1, y1, x2, y2 = segs[0].crop_region
        mask = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        mask[:, y1:y2, x1:x2] = segs[0].cropped_mask.numpy()
        return mask

    assert tracked(0.5).sum() == 3 * 8 * 14
    assert tracked(0.8).sum() == 3 * 4 * 8


@pytest.mark.parametrize("change", ["forward_video as in ComfyUI v0.20.0", "no forward_segment", "no unpack_masks"])
def test_sam3_is_refused_on_a_comfyui_whose_sam3_it_cannot_call(core, monkeypatch, change):
    model = sam3_model(core, monkeypatch).model
    if change == "forward_video as in ComfyUI v0.20.0":
        model.model.diffusion_model.forward_video = lambda images, initial_masks, pbar=None: None
    elif change == "no forward_segment":
        del model.model.diffusion_model.forward_segment
    else:
        import comfy.ldm.sam3.tracker
        monkeypatch.delattr(comfy.ldm.sam3.tracker, "unpack_masks")

    with pytest.raises(Exception, match="SAM3 cannot be used with the current version of ComfyUI"):
        core.SAM3Wrapper(model)
