import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for UltraFace-RFB (Face Detection).
    CONTRACT ENFORCEMENT:
    - Input: Direct Resize (Distorted) to target resolution.
    - Normalization: (Pixel - 127) / 128.
    - Output: Two tensors [Scores, Boxes] (Indices vary by export).
    - Decoding: Custom RFB Anchor generation and SSD-style decoding.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "UltraFace-RFB"
        self.classes = ["face"]

        # UltraFace RFB architecture parameters
        self.strides = [8, 16, 32, 64]
        self.min_boxes = [
            [10, 16, 24],  # Stride 8
            [32, 48],  # Stride 16
            [64, 96],  # Stride 32
            [128, 192, 256],  # Stride 64
        ]

    def get_score(self, model_metadata):
        """
        Adapter confidence scoring used to select the best adapter
        for a given ONNX model.
        """
        score = 0.0
        name = model_metadata.get("name", "").lower()

        # Prefer UltraFace models explicitly
        if "ultraface" in name:
            score += 0.2

        out_names = [n.lower() for n in model_metadata.get("output_names", [])]
        if model_metadata.get("output_count") == 2:
            if any("score" in n for n in out_names) and any(
                "box" in n for n in out_names
            ):
                score += 1.1

        return min(score, 1.5)

    def preprocess(self, image_rgb, model_metadata):
        """
        Preprocess input image by resizing directly to the model
        input resolution.

        NOTE:
        UltraFace RFB models are trained using direct resize
        (no letterboxing), so aspect ratio distortion is expected
        and correct for this architecture.
        """
        i_shape = model_metadata.get("input_shape", [1, 3, 240, 320])
        target_h, target_w = i_shape[2], i_shape[3]

        img_res = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        img_f = (img_res.astype(np.float32) - 127.0) / 128.0
        processed_input = np.transpose(img_f, (2, 0, 1))[None, ...]

        return processed_input.astype(np.float32), {
            "target_res": (target_h, target_w)
        }

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """
        Decode model outputs, apply confidence thresholding,
        and return bounding boxes in original image coordinates.
        """
        h_orig, w_orig = original_shape
        t_h, t_w = run_params["target_res"]
        out_names = run_params.get("output_names", [])

        # Resolve output tensor indices
        try:
            idx_scores = next(
                i for i, n in enumerate(out_names) if "score" in n.lower()
            )
            idx_boxes = next(
                i for i, n in enumerate(out_names) if "box" in n.lower()
            )
        except StopIteration:
            # Fallback based on tensor shape
            idx_scores = 0 if outputs[0].shape[-1] == 2 else 1
            idx_boxes = 1 if idx_scores == 0 else 0

        raw_scores = np.squeeze(outputs[idx_scores])
        raw_boxes = np.squeeze(outputs[idx_boxes])

        # Generate anchors dynamically based on model resolution
        anchors = self._generate_anchors(t_w, t_h)

        # Critical consistency check
        if len(anchors) != raw_scores.shape[0]:
            run_params["error"] = (
                f"[UltraFace] Anchor mismatch: generated {len(anchors)} anchors, "
                f"but model returned {raw_scores.shape[0]} predictions."
            )
            return self._get_empty_results()

        scores = raw_scores[:, 1]
        mask = scores >= threshold

        if not np.any(mask):
            return self._get_empty_results()

        decoded_boxes = self._decode(raw_boxes[mask], anchors[mask])

        # Map boxes back to original image coordinates
        final_boxes = np.stack(
            [
                decoded_boxes[:, 0] * w_orig,
                decoded_boxes[:, 1] * h_orig,
                decoded_boxes[:, 2] * w_orig,
                decoded_boxes[:, 3] * h_orig,
            ],
            axis=1,
        )

        return (
            np.zeros(len(final_boxes), dtype=np.int32),
            scores[mask].astype(np.float32),
            final_boxes.astype(np.float32),
        )

    def _generate_anchors(self, t_w, t_h):
        """
        Generate SSD anchors for UltraFace RFB.
        """
        feature_maps = [
            [int(np.ceil(t_h / s)), int(np.ceil(t_w / s))] for s in self.strides
        ]

        anchors = []
        for i, (f_h, f_w) in enumerate(feature_maps):
            for y in range(f_h):
                for x in range(f_w):
                    for size in self.min_boxes[i]:
                        cx = (x + 0.5) * self.strides[i] / t_w
                        cy = (y + 0.5) * self.strides[i] / t_h
                        s_kx = size / t_w
                        s_ky = size / t_h
                        anchors.append([cx, cy, s_kx, s_ky])

        return np.array(anchors, dtype=np.float32)

    def _decode(self, deltas, anchors):
        """
        Standard SSD box decoding using UltraFace variances.
        """
        cx = anchors[:, 0] + deltas[:, 0] * 0.1 * anchors[:, 2]
        cy = anchors[:, 1] + deltas[:, 1] * 0.1 * anchors[:, 3]
        w = anchors[:, 2] * np.exp(deltas[:, 2] * 0.2)
        h = anchors[:, 3] * np.exp(deltas[:, 3] * 0.2)

        return np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
        )
