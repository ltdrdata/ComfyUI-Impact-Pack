import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    RetinaNet Adapter (COCO Standards).
    CONTRACT ENFORCEMENT:
    - Input: Fixed 480x640 RGB with ImageNet Mean/Std normalization.
    - Output: 10 Tensors (5 Classification Heads + 5 Regression Heads).
    - Architecture: Feature Pyramid Network (FPN) with levels P3-P7.
    - Decoding: Manual anchor generation and sigmoid activation.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "RetinaNet-Pyramid"

        # Standard COCO 80 classes (w/background class)
        self.classes = [
            "__background__",
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

        self.num_anchors = 9
        self.num_classes = 80

    def get_score(self, model_metadata):
        """Identifies RetinaNet by pyramid head signatures."""
        score = 0.0
        if model_metadata.get("output_count") == 10:
            score += 0.9
        if "retina" in model_metadata.get("name", "").lower():
            score += 0.3
        return min(score, 1.2)

    def preprocess(self, image_rgb, model_metadata):
        """Hardened 480x640 resize with ImageNet stats."""
        target_h, target_w = 480, 640
        img_res = cv2.resize(
            image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        img_f = img_res.astype(np.float32) / 255.0
        img_n = (img_f - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return np.transpose(img_n, (2, 0, 1))[None, ...].astype(np.float32), {
            "target_res": (target_h, target_w)
        }

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """
        FIXED: Corrected class mapping (no slicing) and re-projection logic.
        """
        h_orig, w_orig = original_shape
        t_h, t_w = run_params["target_res"]
        all_labels, all_scores, all_boxes = [], [], []

        # 1. Discovery and Sorting
        cls_heads = sorted(
            [o for o in outputs if o.shape[1] == 720],
            key=lambda x: x.shape[2],
            reverse=True,
        )
        reg_heads = sorted(
            [o for o in outputs if o.shape[1] == 36],
            key=lambda x: x.shape[2],
            reverse=True,
        )
        strides = [8, 16, 32, 64, 128]

        for i in range(len(cls_heads)):
            cls_map, reg_map = cls_heads[i][0], reg_heads[i][0]
            stride = strides[i]
            _, fh, fw = cls_map.shape

            # 2. Reshape and Activation
            cls_map = (
                cls_map.reshape(self.num_anchors, self.num_classes, fh, fw)
                .transpose(2, 3, 0, 1)
                .reshape(-1, self.num_classes)
            )
            reg_map = (
                reg_map.reshape(self.num_anchors, 4, fh, fw)
                .transpose(2, 3, 0, 1)
                .reshape(-1, 4)
            )

            # Sigmoid activation
            scores = 1 / (1 + np.exp(-np.clip(cls_map, -20, 20)))

            # FIXED: Do NOT slice scores. Index 0 is already 'person'.
            max_scores = np.max(scores, axis=1)
            effective_threshold = max(threshold, 0.51)
            mask = max_scores >= effective_threshold

            if not np.any(mask):
                continue

            # 3. Anchor Decoding
            anchors = self._generate_anchors(fh, fw, stride)
            boxes = self._decode(reg_map[mask], anchors[mask])

            # Scaling to original image
            fx1, fy1, fx2, fy2 = (
                boxes[:, 0] * w_orig / t_w,
                boxes[:, 1] * h_orig / t_h,
                boxes[:, 2] * w_orig / t_w,
                boxes[:, 3] * h_orig / t_h,
            )

            geo_mask = (fx2 > fx1 + 2.0) & (fy2 > fy1 + 2.0)
            if not np.any(geo_mask):
                continue

            all_boxes.append(
                np.stack(
                    [
                        fx1[geo_mask],
                        fy1[geo_mask],
                        fx2[geo_mask],
                        fy2[geo_mask],
                    ],
                    axis=1,
                )
            )
            all_scores.append(max_scores[mask][geo_mask])

            # FIXED: Map directly to self.classes. Since class[0] is background,
            # model index 0 (person) becomes 1.
            all_labels.append(np.argmax(scores[mask][geo_mask], axis=1) + 1)

        if not all_boxes:
            return self._get_empty_results()
        return (
            np.concatenate(all_labels).astype(np.int32),
            np.concatenate(all_scores).astype(np.float32),
            np.concatenate(all_boxes).astype(np.float32),
        )

    def _generate_anchors(self, fh, fw, stride):
        """Explicit anchor generation for 9 configurations."""
        y, x = np.mgrid[0:fh, 0:fw] * stride + stride // 2
        centers = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        base_size = stride * 4
        level_anchors = []
        for s in [2**0, 2 ** (1 / 3), 2 ** (2 / 3)]:
            for r in [0.5, 1.0, 2.0]:
                w, h = base_size * s * np.sqrt(r), base_size * s / np.sqrt(r)
                level_anchors.append(
                    np.hstack(
                        [
                            centers,
                            np.full((len(centers), 1), w),
                            np.full((len(centers), 1), h),
                        ]
                    )
                )
        return np.array(level_anchors).transpose(1, 0, 2).reshape(-1, 4)

    def _decode(self, deltas, anchors):
        """FIXED: Corrected dw/dh application for width and height."""
        dx, dy, dw, dh = (
            deltas[:, 0] * 0.1,
            deltas[:, 1] * 0.1,
            deltas[:, 2] * 0.2,
            deltas[:, 3] * 0.2,
        )
        ctx, cty = (
            dx * anchors[:, 2] + anchors[:, 0],
            dy * anchors[:, 3] + anchors[:, 1],
        )

        # FIXED: Use dw for width and dh for height
        w = np.exp(np.clip(dw, -10, 10)) * anchors[:, 2]
        h = np.exp(np.clip(dh, -10, 10)) * anchors[:, 3]
        return np.stack(
            [ctx - w / 2, cty - h / 2, ctx + w / 2, cty + h / 2], axis=1
        )
