import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TYPES = {"float32": "tensor(float)", "int32": "tensor(int32)"}


class FakeSession:
    def __init__(self, input_shape, outputs, metadata=None):
        self.input_shape = input_shape
        self.outputs = outputs
        self.metadata = metadata or {}
        self.fed = None

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.input_shape)]

    def get_outputs(self):
        return [SimpleNamespace(name=f"output{index}", type=TYPES[output.dtype.name], shape=list(output.shape)) for index, output in enumerate(self.outputs)]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def run(self, names, feed):
        self.fed = feed["images"]
        return self.outputs


@pytest.fixture
def inference():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from impact import impact_onnx

        def run(model, image):
            patch.setattr(impact_onnx.onnxruntime, "InferenceSession", lambda path, providers: model)
            labels, scores, boxes = impact_onnx.onnx_inference(image, "model.onnx")
            return [(label if isinstance(label, str) else int(label), round(float(score), 2), tuple(int(value) for value in box)) for label, score, box in zip(labels, scores, boxes)]

        yield run


def retinanet(labels, scores, boxes):
    return FakeSession(["batch", "height", "width", 3], [np.array([labels], dtype=np.int32), np.array([scores], dtype=np.float32), np.array([boxes], dtype=np.float32)])


def yolo(boxes, class_scores, input_shape=(1, 3, 320, 320), anchors=2100, metadata=None):
    output = np.zeros((1, 4 + len(class_scores[0]), anchors), dtype=np.float32)
    for column, (box, scores) in enumerate(zip(boxes, class_scores)):
        output[0, :4, column] = box
        output[0, 4:, column] = scores
    return FakeSession(list(input_shape), [output], metadata)


def test_the_slots_that_the_retinanet_model_marks_with_minus_one_are_left_out(inference):
    model = retinanet([3, 7, -1, -1], [0.9, 0.6, -1, -1], [[20, 30, 60, 50], [100, 10, 140, 70], [-1, -1, -1, -1], [-1, -1, -1, -1]])

    assert inference(model, torch.zeros((1, 96, 160, 3))) == [(3, 0.9, (20, 30, 60, 50)), (7, 0.6, (100, 10, 140, 70))]


def test_a_retinanet_model_that_fills_every_slot_gives_them_all(inference):
    model = retinanet([3, 7], [0.9, 0.6], [[20, 30, 60, 50], [100, 10, 140, 70]])

    assert inference(model, torch.zeros((1, 96, 160, 3))) == [(3, 0.9, (20, 30, 60, 50)), (7, 0.6, (100, 10, 140, 70))]


def test_the_retinanet_model_gets_the_picture_as_it_did_before(inference):
    model = retinanet([-1], [-1], [[-1, -1, -1, -1]])

    inference(model, torch.ones((1, 96, 160, 3)))

    assert model.fed.shape == (1, 96, 160, 3)
    assert np.allclose(model.fed[0, 0, 0], [255 - 103.939, 255 - 116.779, 255 - 123.68])


def test_a_yolo_model_gives_its_boxes_in_the_pixels_of_the_picture(inference):
    model = yolo(boxes=[(100, 130, 40, 20), (250, 200, 20, 60)], class_scores=[(0.9, 0.1), (0.2, 0.7)])

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [(0, 0.9, (160, 80, 240, 120)), (1, 0.7, (480, 180, 520, 300))]


def test_a_yolo_model_gets_the_picture_fitted_into_its_input_size(inference):
    model = yolo(boxes=[], class_scores=[(0.0, 0.0)])
    image = torch.zeros((1, 320, 640, 3))
    image[..., 0] = 1.0

    inference(model, image)

    assert model.fed.shape == (1, 3, 320, 320) and model.fed.dtype == np.float32
    assert np.allclose(model.fed[0, :, 160, 160], [1.0, 0.0, 0.0])
    assert np.allclose(model.fed[0, :, 40, 160], 114 / 255)


def test_a_yolo_model_with_a_free_input_size_gets_the_size_it_was_exported_with(inference):
    model = yolo(boxes=[], class_scores=[(0.0,)], input_shape=["batch", 3, "height", "width"], metadata={"imgsz": "[480, 480]"})

    inference(model, torch.zeros((1, 320, 640, 3)))

    assert model.fed.shape == (1, 3, 480, 480)


def test_boxes_of_one_class_on_the_same_object_become_one_box(inference):
    model = yolo(boxes=[(100, 130, 40, 20), (101, 131, 40, 20), (100, 130, 40, 20)], class_scores=[(0.6, 0.0), (0.9, 0.0), (0.0, 0.8)])

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [(0, 0.9, (162, 82, 242, 122)), (1, 0.8, (160, 80, 240, 120))]


def test_a_yolo_model_names_its_classes(inference):
    model = yolo(boxes=[(100, 130, 40, 20)], class_scores=[(0.1, 0.9)], metadata={"names": "{0: 'hand', 1: 'face'}"})

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [("face", 0.9, (160, 80, 240, 120))]


def test_a_yolo_model_with_masks_gives_its_boxes(inference):
    model = yolo(boxes=[(100, 130, 40, 20)], class_scores=[(0.1, 0.9, 0.99, 0.99)], metadata={"names": "{0: 'hand', 1: 'face'}"})

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [("face", 0.9, (160, 80, 240, 120))]


def test_a_box_that_leaves_the_picture_is_cut_to_it(inference):
    model = yolo(boxes=[(10, 90, 40, 40)], class_scores=[(0.9,)])

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [(0, 0.9, (0, 0, 60, 60))]


def test_a_yolo_model_that_filters_its_own_boxes_gives_them_as_they_are(inference):
    rows = np.zeros((1, 300, 6), dtype=np.float32)
    rows[0, 0] = [80, 120, 120, 140, 0.9, 1]
    rows[0, 1] = [81, 121, 121, 141, 0.8, 1]
    model = FakeSession([1, 3, 320, 320], [rows], {"names": "{0: 'hand', 1: 'face'}"})

    assert inference(model, torch.zeros((1, 320, 640, 3))) == [("face", 0.9, (160, 80, 240, 120)), ("face", 0.8, (162, 82, 242, 122))]


@pytest.mark.parametrize("model", [
    FakeSession([1, 3, 480, 640], [np.zeros((1, 1000, 1, 4), dtype=np.float32), np.zeros((1, 1000, 1), dtype=np.float32)]),
    FakeSession([1, 3, 640, 640], [np.zeros((1, 16800, 4), dtype=np.float32), np.zeros((1, 16800, 2), dtype=np.float32)]),
    FakeSession([1, 3, 640, 640], [np.zeros((1, 25200, 85), dtype=np.float32)]),
    FakeSession([1, 1, 28, 28], [np.zeros((1, 10), dtype=np.float32)]),
    FakeSession(["batch", "height", "width", 3], [np.zeros((1, 3), dtype=np.float32)]),
    FakeSession(["batch", "height", "width", 3], [np.zeros((4,), dtype=np.int32), np.zeros((1, 3), dtype=np.float32)]),
    FakeSession([8], [np.zeros((8,), dtype=np.float32)]),
], ids=["boxes and scores apart", "rows of four values", "rows with an object score", "not a picture of three channels",
        "channels last without slots", "channels last with a whole number and no slots", "not a picture"])
def test_a_model_of_another_format_is_refused_with_the_formats_that_work(inference, model):
    with pytest.raises(ValueError, match="Ultralytics YOLO.*detector_v2_base_checkpoint.onnx"):
        inference(model, torch.zeros((1, 320, 640, 3)))


def test_an_onnxruntime_that_cannot_be_imported_is_the_error_of_the_detector():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from impact import additional_dependencies, impact_onnx
        patch.setattr(additional_dependencies, "ensure_onnx_package", lambda: None)
        patch.setitem(sys.modules, "onnxruntime", None)

        with pytest.raises(ImportError, match="onnxruntime"):
            importlib.reload(impact_onnx)
    importlib.reload(impact_onnx)
