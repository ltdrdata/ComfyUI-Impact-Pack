import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for YOLOv2 (ONNX Zoo / COCO).
    CONTRACT ENFORCEMENT:
    - Input: Strict 416x416 NCHW, normalized [0, 1].
    - Output: Single tensor [1, 425, 13, 13] (5 anchors * 85 channels).
    - Decoding: Manual 13x13 grid decoding with Sigmoid/Softmax activation.
    - Anchors: Uses fixed Official YOLOv2 COCO anchors.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLOv2-Zoo"
        # Standard COCO 80 classes
        self.classes = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        # Official YOLOv2 Anchors for COCO
        self.anchors = [
            (0.57273, 0.677385),
            (1.87446, 2.06253),
            (3.33843, 5.47434),
            (7.88282, 3.52778),
            (9.77052, 9.16828),
        ]

    def get_score(self, model_metadata):
        """
        Aggressive Discovery: Uses the unique 425-channel signature.
        This must beat the generic YOLO adapter's 70%.
        """
        score = 0.0
        name = model_metadata.get("name", "").lower()
        out_shapes = model_metadata.get("output_shapes", [])

        # 1. Structural Match: The [1, 425, 13, 13] signature is unmistakable
        if any(s == [1, 425, 13, 13] for s in out_shapes):
            score += 1.1  # Instant priority

        # 2. Name Match: specifically for yolov2
        if "yolov2" in name:
            score += 0.2

        return min(score, 1.3)  # Max priority to override others

    def preprocess(self, image_rgb, model_metadata):
        """YOLOv2 expects strict 416x416 NCHW normalized [0, 1]."""
        target_res = (416, 416)
        img_res = cv2.resize(
            image_rgb, target_res, interpolation=cv2.INTER_LINEAR
        )
        img_f = img_res.astype(np.float32) / 255.0
        processed_input = np.transpose(img_f, (2, 0, 1))[None, ...]

        return processed_input.astype(np.float32), {
            "orig_res": image_rgb.shape[:2]
        }

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Decodes the 13x13 grid into labels, scores, and boxes."""
        h_orig, w_orig = original_shape

        # Shape normalization
        data = np.squeeze(outputs[0])  # [425, 13, 13]
        if data.shape != (425, 13, 13):
            return self._get_empty_results()

        # Reshape to [5 anchors, 85 values (5+80), 13, 13]
        data = data.reshape(5, 85, 13, 13).transpose(
            0, 2, 3, 1
        )  # [5, 13, 13, 85]

        # Activation: Sigmoid for XY and Objectness
        data[..., 0:2] = 1 / (1 + np.exp(-np.clip(data[..., 0:2], -20, 20)))
        data[..., 4] = 1 / (1 + np.exp(-np.clip(data[..., 4], -20, 20)))

        # Grid offsets for relative-to-cell position
        grid_y, grid_x = np.mgrid[0:13, 0:13]
        data[..., 0] += grid_x
        data[..., 1] += grid_y

        # Anchor scaling for W, H
        for i in range(5):
            data[i, ..., 2] = np.exp(data[i, ..., 2]) * self.anchors[i][0]
            data[i, ..., 3] = np.exp(data[i, ..., 3]) * self.anchors[i][1]

        # Softmax for class probabilities
        logits = data[..., 5:]
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)

        # Confidence calculation: Objectness * Max Class Prob
        max_class_probs = np.max(probs, axis=-1)
        final_scores = data[..., 4] * max_class_probs

        mask = final_scores >= threshold
        if not np.any(mask):
            return self._get_empty_results()

        # Transform 13-grid coordinates to pixel space
        # x_ctr, y_ctr, w, h are in 13x13 units
        raw_boxes = data[mask][..., 0:4]
        cx, cy, w, h = (
            raw_boxes[:, 0] / 13,
            raw_boxes[:, 1] / 13,
            raw_boxes[:, 2] / 13,
            raw_boxes[:, 3] / 13,
        )

        fx1, fy1, fx2, fy2 = (
            (cx - w / 2) * w_orig,
            (cy - h / 2) * h_orig,
            (cx + w / 2) * w_orig,
            (cy + h / 2) * h_orig,
        )

        return (
            np.argmax(probs[mask], axis=-1).astype(np.int32),
            final_scores[mask].astype(np.float32),
            np.stack([fx1, fy1, fx2, fy2], axis=1).astype(np.float32),
        )
