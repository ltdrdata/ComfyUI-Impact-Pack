import io
import json

import numpy as np
import pytest
from PIL import Image

from impact import sam_editor


def snapshot():
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def test_snapshot_coordinates_and_confidence():
    image, points, labels, threshold = sam_editor.read_request(snapshot(), json.dumps({
        "positive_points": [[1.25, 1]], "negative_points": [[2, 0]], "threshold": 0.85
    }))
    np.testing.assert_array_equal(image, np.full((2, 3, 3), [20, 40, 60], dtype=np.uint8))
    assert points == [[1.25, 1], [2, 0]]
    assert labels == [1, 0]
    assert threshold == 0.85


@pytest.mark.parametrize("prompts", [
    [], {}, {"positive_points": "bad"}, {"positive_points": [[1]]},
    {"positive_points": [[1, float("nan")]]},
    {"positive_points": [[1, 1]], "threshold": -1},
    {"positive_points": [[1, 1]], "threshold": "0.7"},
])
def test_invalid_prompts_are_rejected_before_model_loading(prompts):
    with pytest.raises(ValueError):
        sam_editor.read_request(snapshot(), json.dumps(prompts))


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_request_owned_model_returns_to_cpu_and_uses_the_supplied_snapshot(monkeypatch, fail):
    moves = []
    images = []

    class Model:
        def to(self, device):
            moves.append(device)

        def cpu(self):
            moves.append("cpu")

    class Predictor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image, color):
            assert color == "RGB"
            images.append(image.copy())

        def predict(self, point_coords, point_labels, box):
            if fail:
                raise RuntimeError("inference failed")
            np.testing.assert_array_equal(point_coords, [[1, 1]])
            np.testing.assert_array_equal(point_labels, [1])
            assert box is None
            masks = np.array([[[False, True, False], [False, True, False]],
                              [[False, False, False], [False, True, False]]])
            return masks, np.array([0.9, 0.8]), None

    monkeypatch.setitem(sam_editor.sam_model_registry, "vit_b", lambda checkpoint: Model())
    monkeypatch.setattr(sam_editor, "SamPredictor", Predictor)
    values = sam_editor.read_request(snapshot(), '{"positive_points": [[1, 1]]}')
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            sam_editor.detect_mask(*values, "sam_vit_b.pth", "test-device")
    else:
        result = sam_editor.detect_mask(*values, "sam_vit_b.pth", "test-device")
        with Image.open(io.BytesIO(result)) as mask:
            np.testing.assert_array_equal(np.asarray(mask), [[0, 255, 0], [0, 255, 0]])
    assert moves == ["test-device", "cpu"]
    np.testing.assert_array_equal(images[0], values[0])
