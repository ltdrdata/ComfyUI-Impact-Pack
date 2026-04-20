import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for YOLOv4 Zoo (3 Identity outputs).
    CONTRACT ENFORCEMENT - ZOO SPECIFIC:
    - Input: NHWC format [1, 416, 416, 3] (Channels Last).
    - Outputs: 3 Tensors [Identity, Identity_1, Identity_2] with Grid format.
    - Anchors: Standard COCO anchors used for decoding.
    - Normalization: Float32 [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLOv4-Identity-Zoo"
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
        # Standard COCO Anchors for YOLOv4
        self.anchors = [
            [[12, 16], [19, 36], [40, 28]],  # 52x52
            [[36, 75], [76, 55], [72, 146]],  # 26x26
            [[142, 110], [192, 243], [459, 401]],  # 13x13
        ]
        # Precise scaling factors for YOLOv4 Mish
        self.strides = [8, 16, 32]
        self.xyscale = [1.2, 1.1, 1.05]

    def get_score(self, model_metadata):
        """Heuristic: 3 outputs + NHWC input layout + 'Identity' naming."""
        score = 0.0
        out_names = [n.lower() for n in model_metadata.get("output_names", [])]
        i_shape = model_metadata.get("input_shape", [])

        if model_metadata.get("output_count") == 3:
            score += 0.4
            if any("identity" in n for n in out_names):
                score += 0.3

        if len(i_shape) == 4 and i_shape[3] == 3:
            score += 0.3

        return min(score, 1.0)

    def preprocess(self, image_rgb, model_metadata):
        """Direct Warp Resize to model dimensions in NHWC."""
        i_shape = model_metadata["input_shape"]
        target_h, target_w = i_shape[1], i_shape[2]

        img_resized = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        processed_input = (img_bgr.astype(np.float32) / 255.0)[None, ...]

        # Calculate padding and ratio for inverse mapping
        h_orig, w_orig = image_rgb.shape[:2]
        ratio = min(target_w / w_orig, target_h / h_orig)
        dw = (target_w - w_orig * ratio) / 2
        dh = (target_h - h_orig * ratio) / 2

        run_params = {
            "target_res": (target_h, target_w),
            "ratio": ratio,
            "pad": (dw, dh),
        }
        return processed_input, run_params

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Decodes 3 grid-based outputs using anchors, strides and xyscale."""
        dw, dh = run_params["pad"]
        ratio = run_params["ratio"]
        all_boxes, all_scores, all_labels = [], [], []

        # Sort outputs by grid size descending (52, 26, 13)
        sorted_outputs = sorted(outputs, key=lambda x: x.shape[1], reverse=True)

        for i, output in enumerate(sorted_outputs):
            grid_size = output.shape[1]
            raw = output[0].reshape(grid_size, grid_size, 3, 85)

            # Confidence and Class probability
            conf = self._sigmoid(raw[..., 4:5])
            prob = self._sigmoid(raw[..., 5:])
            scores_matrix = conf * prob

            # Filter detections by threshold (index 0 is PERSON)
            mask = scores_matrix[..., 0] >= threshold
            if not np.any(mask):
                continue

            idx = np.where(mask)
            for y, x, a in zip(idx[0], idx[1], idx[2]):
                # Precise coordinate decoding with xyscale
                raw_xy = raw[y, x, a, 0:2]  # Get raw values before sigmoid
                pred_x = (
                    (self._sigmoid(raw_xy[0]) * self.xyscale[i])
                    - 0.5 * (self.xyscale[i] - 1)
                    + x
                ) * self.strides[i]
                pred_y = (
                    (self._sigmoid(raw_xy[1]) * self.xyscale[i])
                    - 0.5 * (self.xyscale[i] - 1)
                    + y
                ) * self.strides[i]

                # BBox size decoding
                pred_w = np.exp(raw[y, x, a, 2]) * self.anchors[i][a][0]
                pred_h = np.exp(raw[y, x, a, 3]) * self.anchors[i][a][1]

                # Inverse Letterbox mapping
                all_boxes.append(
                    [
                        (pred_x - pred_w / 2 - dw) / ratio,
                        (pred_y - pred_h / 2 - dh) / ratio,
                        (pred_x + pred_w / 2 - dw) / ratio,
                        (pred_y + pred_h / 2 - dh) / ratio,
                    ]
                )
                all_scores.append(scores_matrix[y, x, a, 0])
                all_labels.append(0)

        if not all_boxes:
            return self._get_empty_results()

        return (
            np.array(all_labels, dtype=np.int32),
            np.array(all_scores, dtype=np.float32),
            np.array(all_boxes, dtype=np.int32),
        )

    def _sigmoid(self, x):
        """Numerical stable sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
