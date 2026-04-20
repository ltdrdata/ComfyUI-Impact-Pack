import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for Tiny YOLOv2 (Pascal VOC).
    CONTRACT ENFORCEMENT:
    - Input: Dictionary with 3 keys ('image', 'scalerPreprocessor_scale', 'scalerPreprocessor_bias').
    - Layout: NCHW 416x416 for image tensor.
    - Output: Single tensor [1, 125, 13, 13] (Grid).
    - Specifics: Handles CoreML-converted ONNX models with static scaler inputs.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "Tiny-YOLO-v2"
        # Pascal VOC 20 classes
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        # Official Tiny YOLO VOC anchors
        self.anchors = [
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),
        ]

    def get_score(self, model_metadata):
        score = 0.0
        out_names = model_metadata.get("output_names", [])
        # Score high if 'grid' is found and input is 416x416
        if any("grid" in name for name in out_names):
            score += 0.5
            if model_metadata.get("input_shape", [])[2] == 416:
                score += 0.4
        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """Prepares the image and the mandatory scaler inputs for tinyyolov2-7."""
        img_resized = cv2.resize(
            image_rgb, (416, 416), interpolation=cv2.INTER_LINEAR
        )
        img_float = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_float, (2, 0, 1))[None, ...]

        # MAPPING THE 3 INPUTS FROM YOUR RADIOGRAPH
        processed_inputs = {
            "image": img_tensor,
            "scalerPreprocessor_scale": np.array([1.0], dtype=np.float32),
            "scalerPreprocessor_bias": np.array(
                [0.0, 0.0, 0.0], dtype=np.float32
            ).reshape(3, 1, 1),
        }

        return processed_inputs, {"orig_res": image_rgb.shape[:2]}

    def postprocess(self, outputs, original_shape, threshold, run_params):
        # [1, 125, 13, 13] -> [5 anchors, 25 features, 13, 13 grid]
        prediction = outputs[0][0].reshape(5, 25, 13, 13)
        h_orig, w_orig = original_shape
        all_labels, all_scores, all_boxes = [], [], []

        # Diagnostic tracker
        max_conf_found = 0.0

        for b in range(5):
            # Sigmoid for coordinates and objectness
            x_grid = self._sigmoid(prediction[b, 0])
            y_grid = self._sigmoid(prediction[b, 1])
            obj_score = self._sigmoid(prediction[b, 4])

            w_grid = np.exp(prediction[b, 2]) * self.anchors[b][0]
            h_grid = np.exp(prediction[b, 3]) * self.anchors[b][1]

            # Softmax for the 20 classes
            class_probs = self._softmax(prediction[b, 5:])

            for cy in range(13):
                for cx in range(13):
                    max_cls_idx = np.argmax(class_probs[:, cy, cx])
                    confidence = (
                        obj_score[cy, cx] * class_probs[max_cls_idx, cy, cx]
                    )
                    max_conf_found = max(max_conf_found, confidence)

                    if confidence > threshold:
                        # Normalize relative to the 13x13 grid
                        bx = (cx + x_grid[cy, cx]) / 13.0
                        by = (cy + y_grid[cy, cx]) / 13.0
                        bw = w_grid[cy, cx] / 13.0
                        bh = h_grid[cy, cx] / 13.0

                        # Project back to original image resolution
                        all_boxes.append(
                            [
                                (bx - bw / 2) * w_orig,
                                (by - bh / 2) * h_orig,
                                (bx + bw / 2) * w_orig,
                                (by + bh / 2) * h_orig,
                            ]
                        )
                        all_labels.append(max_cls_idx)
                        all_scores.append(float(confidence))

        if not all_boxes:
            return self._get_empty_results()

        return (
            np.array(all_labels, dtype=np.int32),
            np.array(all_scores, dtype=np.float32),
            np.array(all_boxes, dtype=np.int32),
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
