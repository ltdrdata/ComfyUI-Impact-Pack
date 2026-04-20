import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for specialized YOLOv4 Head Detection (480x640).
    CONTRACT ENFORCEMENT:
    - Input: NCHW format [1, 3, 480, 640] (RGB).
    - Output 0: Bounding boxes [1, 18900, 1, 4] in [x1, y1, x2, y2] format.
    - Output 1: Confidence scores [1, 18900, 1].
    - Aspect Ratio: Rectangular Letterbox.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLOv4-Head-Split"
        self.classes = ["head"]

    def get_score(self, model_metadata):
        """Deterministic heuristic: 2 outputs and 480x640 input resolution."""
        score = 0.0
        i_shape = model_metadata.get("input_shape", [])
        output_count = model_metadata.get("output_count", 0)
        name = model_metadata["name"]

        if output_count == 2 and len(i_shape) == 4:
            score += 0.4
            if i_shape[2] == 480 and i_shape[3] == 640:
                score += 0.3

        if "head" in name:
            score += 0.2
        if "yolov4" in name:
            score += 0.1

        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """Rectangular Letterbox (480x640) maintaining RGB color space."""
        i_shape = model_metadata["input_shape"]
        target_h, target_w = i_shape[2], i_shape[3]  # 480, 640

        h_orig, w_orig = image_rgb.shape[:2]
        ratio = min(target_h / h_orig, target_w / w_orig)
        new_unpad = int(round(w_orig * ratio)), int(round(h_orig * ratio))

        dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

        img_resized = cv2.resize(
            image_rgb, new_unpad, interpolation=cv2.INTER_LINEAR
        )
        img_padded = cv2.copyMakeBorder(
            img_resized,
            int(round(dh - 0.1)),
            int(round(dh + 0.1)),
            int(round(dw - 0.1)),
            int(round(dw + 0.1)),
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Keep RGB space as confirmed in YOLOv8 fix
        img_float = img_padded.astype(np.float32) / 255.0
        processed_input = np.transpose(img_float, (2, 0, 1))[None, ...]

        run_params = {
            "ratio": ratio,
            "pad": (dw, dh),
            "target_res": (target_h, target_w),
        }
        return processed_input, run_params

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Decodes [x1, y1, x2, y2] coordinates with dynamic scale detection."""
        # 1. Shape Extraction
        raw_boxes = outputs[0].reshape(-1, 4).astype(np.float32)
        raw_scores = outputs[1].flatten().astype(np.float32)

        # 2. Threshold Filtering
        mask = raw_scores >= threshold
        if not np.any(mask):
            return self._get_empty_results()

        boxes = raw_boxes[mask]
        scores = raw_scores[mask]

        # 3. Dynamic Decoding (Fixed axis order to [x1, y1, x2, y2])
        t_h, t_w = run_params["target_res"]

        # HEURISTIC: Detect normalization
        if np.max(boxes) <= 1.5:
            # Normalized case: Scale x by width, y by height
            x1, y1, x2, y2 = (
                boxes[:, 0] * t_w,
                boxes[:, 1] * t_h,
                boxes[:, 2] * t_w,
                boxes[:, 3] * t_h,
            )
        else:
            # Pixel-space case: x is index 0/2, y is index 1/3
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # 4. Inverse Letterbox Mapping
        ratio = run_params["ratio"]
        dw, dh = run_params["pad"]

        rx1, ry1 = (x1 - dw) / ratio, (y1 - dh) / ratio
        rx2, ry2 = (x2 - dw) / ratio, (y2 - dh) / ratio

        # 5. Final Safety Clipping
        h_orig, w_orig = original_shape
        rx1, ry1 = np.clip(rx1, 0, w_orig - 1), np.clip(ry1, 0, h_orig - 1)
        rx2, ry2 = np.clip(rx2, 0, w_orig - 1), np.clip(ry2, 0, h_orig - 1)

        # 6. Contract Compliance
        labels = np.zeros(len(rx1), dtype=np.int32)
        scores_final = scores.astype(np.float32)
        boxes_final = np.stack([rx1, ry1, rx2, ry2], axis=1).astype(np.int32)

        return labels, scores_final, boxes_final
