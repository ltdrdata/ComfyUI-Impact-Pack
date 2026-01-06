import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for YOLOv8, YOLOv10, and YOLOv11 models.
    CONTRACT ENFORCEMENT:
    - Input: NCHW format (Batch=1, Channels=3, Height, Width).
    - Output Layout: Flat [Attributes, Predictions] or [Predictions, Attributes].
    - Attributes: [cx, cy, w, h, classes...] (min 5 columns for single-class).
    - Color Space: RGB [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLOv8/v10/v11"
        # Generic adapter: No default classes.
        # Falls back to 'class_ID' via BaseAdapter.
        self.classes = []

    def get_score(self, model_metadata):
        """Deterministic heuristic scoring based on output architecture and names."""
        score = 0.0
        output_count = model_metadata.get("output_count", 0)
        input_shape = model_metadata.get("input_shape", [])

        # Check for single output NCHW architecture
        if output_count == 1:
            score += 0.4
            if len(input_shape) == 4 and input_shape[1] == 3:
                score += 0.3

        # Explicit family name check
        if any(
            x in model_metadata["name"]
            for x in ["yolov8", "yolov10", "yolov11"]
        ):
            score += 0.3

        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """Handles Letterboxing and NCHW Batching in RGB space."""
        # 1. Target resolution from metadata
        i_shape = model_metadata["input_shape"]
        target_h, target_w = i_shape[2], i_shape[3]

        # 2. Letterbox (Preserve original aspect ratio)
        h_orig, w_orig = image_rgb.shape[:2]
        ratio = min(target_h / h_orig, target_w / w_orig)
        new_unpad = int(round(w_orig * ratio)), int(round(h_orig * ratio))

        dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

        img_resized = cv2.resize(
            image_rgb, new_unpad, interpolation=cv2.INTER_LINEAR
        )

        # Padding with neutral gray (114) for YOLO standard
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # 3. Normalization (Model expects RGB 0.0-1.0)
        img_float = img_padded.astype(np.float32) / 255.0

        # Result must be [1, 3, H, W] to fulfill the contract
        processed_input = np.transpose(img_float, (2, 0, 1))[None, ...]

        run_params = {
            "ratio": ratio,
            "pad": (dw, dh),
            "target_res": (target_h, target_w),
        }
        return processed_input, run_params

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Decodes box coordinates and filters by score with 5-column support."""
        # Safety check: Ensure raw output exists and is float32
        raw = outputs[0][0].astype(np.float32)

        # Auto-transpose to ensure layout [Predictions, Attributes]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        # Validation: Ensure at least 4 coords + 1 score column (Total 5)
        # This fix allows specialized models like lindevs-face to pass.
        if raw.shape[1] < 5:
            return self._get_empty_results()

        # Layout: [cx, cy, w, h, classes...]
        cx, cy, bw, bh = raw[:, :4].T
        scores = np.max(raw[:, 4:], axis=1)

        # 1. Filter by confidence threshold
        mask = scores >= threshold
        if not np.any(mask):
            return self._get_empty_results()

        # 2. Decode coordinates: [cx, cy, w, h] -> [x1, y1, x2, y2]
        x1, y1 = cx[mask] - bw[mask] / 2, cy[mask] - bh[mask] / 2
        x2, y2 = cx[mask] + bw[mask] / 2, cy[mask] + bh[mask] / 2

        # 3. Inverse Mapping (Letterbox removal)
        ratio = run_params["ratio"]
        dw, dh = run_params["pad"]

        rx1, ry1 = (x1 - dw) / ratio, (y1 - dh) / ratio
        rx2, ry2 = (x2 - dw) / ratio, (y2 - dh) / ratio

        # 4. Final Safety Clipping against original resolution
        h_orig, w_orig = original_shape
        rx1, ry1 = np.clip(rx1, 0, w_orig - 1), np.clip(ry1, 0, h_orig - 1)
        rx2, ry2 = np.clip(rx2, 0, w_orig - 1), np.clip(ry2, 0, h_orig - 1)

        # 5. Return strict data types
        labels = np.zeros(len(rx1), dtype=np.int32)
        scores_final = scores[mask].astype(np.float32)
        boxes_final = np.stack([rx1, ry1, rx2, ry2], axis=1).astype(np.int32)

        return labels, scores_final, boxes_final
