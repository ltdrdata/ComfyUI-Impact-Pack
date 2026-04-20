import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Adapter for Tiny-YOLOv3-11.
    CONTRACT ENFORCEMENT:
    - Input: Fixed 416x416 NCHW + Auxiliary 'image_shape' tensor.
    - Output: 3-tuple [Boxes, Scores, Indices] from internal NMS.
    - Layout: [y1, x1, y2, x2] box coordinates.
    - Optimization: Internal graph handles decoding; Adapter handles projection.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "Tiny-YOLOv3"
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

    def get_score(self, model_metadata):
        score = 0.0
        out_names = [n.lower() for n in model_metadata.get("output_names", [])]
        in_names = [n.lower() for n in model_metadata.get("input_names", [])]
        if any("yolonms" in n for n in out_names) and "image_shape" in in_names:
            score += 1.3
        return score

    def preprocess(self, image_rgb, model_metadata):
        """Hard-coded 416x416 for the Tiny graph structure."""
        h, w = image_rgb.shape[:2]
        img_res = cv2.resize(
            image_rgb, (416, 416), interpolation=cv2.INTER_LINEAR
        )
        img_f = img_res.astype(np.float32)
        processed_img = np.transpose(img_f, (2, 0, 1))[None, ...]

        # Consistent image_shape for the NMS internal layer
        image_shape = np.array([[416, 416]], dtype=np.float32)

        return {"input_1": processed_img, "image_shape": image_shape}, {
            "orig_res": (h, w)
        }

    def postprocess(self, outputs, original_shape, threshold, run_params):
        h_orig, w_orig = original_shape

        boxes_raw = np.squeeze(outputs[0])  # [y1, x1, y2, x2]
        scores_raw = np.squeeze(outputs[1])  # [80, N]
        indices = np.squeeze(outputs[2])

        if indices.ndim != 2 or indices.shape[0] == 0:
            return self._get_empty_results()

        # FIXED: Remove the +1 offset. Index 0 is 'person'.
        labels = indices[:, 1].astype(np.int32)
        box_ids = indices[:, 2].astype(np.int32)

        m_boxes = boxes_raw[box_ids]
        m_scores = scores_raw[labels, box_ids]

        # Use 416 reference to scale back to original image
        r_h, r_w = h_orig / 416, w_orig / 416
        fy1, fx1, fy2, fx2 = (
            m_boxes[:, 0] * r_h,
            m_boxes[:, 1] * r_w,
            m_boxes[:, 2] * r_h,
            m_boxes[:, 3] * r_w,
        )

        # Iron-Grade Filter: Cleanup noise
        mask = (
            (m_scores >= threshold)
            & (fx2 - fx1 >= 1.0)
            & (fy2 - fy1 >= 1.0)
            & (fx1 >= 0)
            & (fy1 >= 0)
            & (fx2 <= w_orig)
            & (fy2 <= h_orig)
        )

        if not np.any(mask):
            return self._get_empty_results()

        return (
            labels[mask].astype(np.int32),
            m_scores[mask].astype(np.float32),
            np.stack(
                [fx1[mask], fy1[mask], fx2[mask], fy2[mask]], axis=1
            ).astype(np.float32),
        )
