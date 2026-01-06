import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for SSD-Mobilenet V1/V2 (TensorFlow API).
    CONTRACT ENFORCEMENT:
    - Input: NHWC format [1, H, W, 3].
    - Data Type: UINT8 [0-255] (No float normalization).
    - Output: 4 Tensors [Boxes, Scores, Classes, Num_Detections].
    - Source: Supports models exported via TensorFlow Object Detection API.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "SSD-Mobilenet-V1"
        # Generic adapter: No default classes.
        # Falls back to 'class_ID' via BaseAdapter.
        self.classes = []

    def get_score(self, model_metadata):
        score = 0.0
        out_names = [n.lower() for n in model_metadata.get("output_names", [])]

        if model_metadata.get("output_count") == 4:
            score += 0.4
            if any("detection_boxes" in n for n in out_names):
                score += 0.5

        if (
            "ssd" in model_metadata["name"]
            and "mobile" in model_metadata["name"]
        ):
            score += 0.1

        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """
        SSD-Mobilenet TF-API:
        - Input: uint8
        - Layout: NHWC
        - Range: 0-255 (no normalization)
        """
        i_shape = model_metadata["input_shape"]

        # TF-API SSD uses NHWC [1, H, W, 3]
        target_h = i_shape[1] if isinstance(i_shape[1], int) else 300
        target_w = i_shape[2] if isinstance(i_shape[2], int) else 300

        img_resized = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # FORCE uint8 – model expects this
        processed_input = img_resized.astype(np.uint8)[None, ...]  # NHWC

        run_params = {
            "output_names": model_metadata.get("output_names", []),
            "target_res": (target_h, target_w),
        }
        return processed_input, run_params

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Strictly maps TF-API detection outputs."""
        out_names = run_params.get("output_names", [])
        h_orig, w_orig = original_shape

        try:
            idx_boxes = next(
                i
                for i, n in enumerate(out_names)
                if "detection_boxes" in n.lower()
            )
            idx_scores = next(
                i
                for i, n in enumerate(out_names)
                if "detection_scores" in n.lower()
            )
            idx_labels = next(
                i
                for i, n in enumerate(out_names)
                if "detection_classes" in n.lower()
            )
            idx_count = next(
                i
                for i, n in enumerate(out_names)
                if "num_detections" in n.lower()
            )
        except StopIteration:
            return self._get_empty_results()

        raw_boxes = outputs[idx_boxes][0]
        raw_scores = outputs[idx_scores][0]
        raw_labels = outputs[idx_labels][0]
        num_det = int(outputs[idx_count][0])

        valid_mask = raw_scores[:num_det] >= threshold
        if not np.any(valid_mask):
            return self._get_empty_results()

        boxes = raw_boxes[:num_det][valid_mask]
        scores = raw_scores[:num_det][valid_mask]
        labels = raw_labels[:num_det][valid_mask].astype(np.int32)

        # El modelo devuelve [y1, x1, y2, x2] normalizados
        y1, x1, y2, x2 = (
            boxes[:, 0] * h_orig,
            boxes[:, 1] * w_orig,
            boxes[:, 2] * h_orig,
            boxes[:, 3] * w_orig,
        )

        rx1, ry1 = np.clip(x1, 0, w_orig - 1), np.clip(y1, 0, h_orig - 1)
        rx2, ry2 = np.clip(x2, 0, w_orig - 1), np.clip(y2, 0, h_orig - 1)

        boxes_final = np.stack([rx1, ry1, rx2, ry2], axis=1).astype(np.int32)
        return labels, scores.astype(np.float32), boxes_final
