import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for ONNX Zoo models with internal NMS layers (e.g., Tiny YOLOv3).
    CONTRACT ENFORCEMENT - NMS SPECIFIC:
    - Inputs: Requires two inputs [image, image_shape].
    - Normalization: Float32 [0, 1].
    - Aspect Ratio: Direct Warp Resize (Standard for this Zoo family).
    - Coordinates: Decoded internally by the graph based on input image_shape.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLO-Zoo-InternalNMS"

    def get_score(self, model_metadata):
        """Heuristic: 2 inputs (one is shape) and 'yolonms' in output names."""
        score = 0.0
        in_names = [
            n.lower()
            for n in model_metadata.get(
                "input_name_list", [model_metadata.get("input_name", "")]
            )
        ]
        out_names = [n.lower() for n in model_metadata.get("output_names", [])]

        # Check for the characteristic 2nd input 'image_shape'
        if (
            model_metadata.get("output_count") == 3
            and len(model_metadata.get("input_shape_list", [])) >= 2
        ):
            score += 0.5

        if any("nms" in n for n in out_names):
            score += 0.4

        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """Handles the dual-tensor input requirement."""
        # 1. Image preparation (NCHW + Warp)
        # Using a default 416x416 if metadata is dynamic (0,0)
        i_shape = model_metadata["input_shape"]
        target_h = i_shape[2] if i_shape[2] > 0 else 416
        target_w = i_shape[3] if i_shape[3] > 0 else 416

        img_resized = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        img_transposed = np.transpose(img_resized, (2, 0, 1))
        img_float = (img_transposed.astype(np.float32) / 255.0)[None, ...]

        # 2. image_shape input
        # This model needs the original [H, W] to scale boxes internally
        h_orig, w_orig = image_rgb.shape[:2]
        shape_tensor = np.array([[h_orig, w_orig]], dtype=np.float32)

        # WE CHANGE THE CONTRACT HERE: preprocess returns a DICT for multi-input
        processed_input = {
            model_metadata["input_name"]: img_float,
            "image_shape": shape_tensor,  # Hardcoded name for this family
        }

        run_params = {"target_res": (target_h, target_w)}
        return processed_input, run_params

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Extracts pre-filtered boxes from internal NMS layer."""
        # Output 0: Boxes [1, N, 4]
        # Output 1: Scores [1, 80, N] -> Careful, many Zoo models transpose scores
        raw_boxes = outputs[0].reshape(-1, 4).astype(np.float32)
        raw_scores = outputs[1].reshape(80, -1).T.astype(np.float32)  # [N, 80]

        # Find max score per box
        scores = np.max(raw_scores, axis=1)
        labels = np.argmax(raw_scores, axis=1).astype(np.int32)

        # 1. Filter by threshold (though NMS layer usually does this, we re-verify)
        mask = scores >= threshold
        if not np.any(mask):
            return self._get_empty_results()

        # 2. Coordinate Handling
        # Since we passed 'image_shape' to the model, boxes ARE ALREADY in pixels!
        boxes = raw_boxes[mask]

        # The model usually returns [y1, x1, y2, x2]
        y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # 3. Degenerate Box Filter & Clipping
        valid_mask = (x2 > x1) & (y2 > y1)

        h_orig, w_orig = original_shape
        rx1, ry1 = np.clip(x1[valid_mask], 0, w_orig - 1), np.clip(
            y1[valid_mask], 0, h_orig - 1
        )
        rx2, ry2 = np.clip(x2[valid_mask], 0, w_orig - 1), np.clip(
            y2[valid_mask], 0, h_orig - 1
        )

        if len(rx1) == 0:
            return self._get_empty_results()

        boxes_final = np.stack([rx1, ry1, rx2, ry2], axis=1).astype(np.int32)
        return labels[valid_mask], scores[mask][valid_mask], boxes_final
