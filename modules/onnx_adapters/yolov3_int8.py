import cv2
import numpy as np
from onnx_adapters.base import BaseAdapter


class Adapter(BaseAdapter):
    """
    Purified Adapter for YOLOv3-12-INT8.
    CONTRACT ENFORCEMENT:
    - Input: Strict 416x416 NCHW (Float32).
    - Context: Requires auxiliary 'image_shape' tensor [1, 2] for NMS.
    - Output: Direct 3-array translation [Boxes, Scores, Indices].
    - Optimization: Targeted for INT8 models with Opset 12 Integrated NMS.
    """

    def __init__(self):
        super().__init__()
        self.FAMILY = "YOLOv3-INT8"
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
        """Strictly targets INT8 YOLOv3 models."""
        name = model_metadata.get("name", "").lower()
        if "yolov3" in name and "int8" in name:
            return 1.3
        return 0.0

    def preprocess(self, image_rgb, model_metadata):
        """Strict 416x416 NCHW input with image_shape tensor."""
        h, w = image_rgb.shape[:2]
        # Resizing to 416x416 as required by this specific INT8 graph
        img_res = cv2.resize(
            image_rgb, (416, 416), interpolation=cv2.INTER_LINEAR
        )
        img_f = img_res.astype(np.float32)
        processed_img = np.transpose(img_f, (2, 0, 1))[None, ...]

        # image_shape informs the internal NMS layer of the grid size
        image_shape = np.array([[416, 416]], dtype=np.float32)

        return {"input_1": processed_img, "image_shape": image_shape}, {
            "orig_res": (h, w)
        }

    def postprocess(self, outputs, original_shape, threshold, run_params):
        """Direct 3-array translation for the Orchestrator."""
        h_orig, w_orig = original_shape

        # Extract NMS Outputs: Boxes [1, N, 4], Scores [1, 80, N], Indices [M, 3]
        boxes_raw = np.squeeze(outputs[0])  # [N, 4] -> [y1, x1, y2, x2]
        scores_raw = np.squeeze(outputs[1])  # [80, N]
        indices = outputs[2]  # [M, 3]

        if indices.shape[0] == 0:
            return self._get_empty_results()

        # Map labels and box IDs directly from indices tensor
        f_labels = indices[:, 1].astype(np.int32)
        box_ids = indices[:, 2].astype(np.int32)

        m_boxes = boxes_raw[box_ids]
        m_scores = scores_raw[f_labels, box_ids]

        # Initial confidence filter to reduce data volume passed to Orchestrator
        mask = m_scores >= threshold
        if not np.any(mask):
            return self._get_empty_results()

        # Proportional Scaling: 416 grid space to original pixel resolution
        # No clipping or area checks here; Orchestrator handles it
        r_h, r_w = h_orig / 416, w_orig / 416
        fy1, fx1, fy2, fx2 = (
            m_boxes[mask, 0] * r_h,
            m_boxes[mask, 1] * r_w,
            m_boxes[mask, 2] * r_h,
            m_boxes[mask, 3] * r_w,
        )

        return (
            f_labels[mask].astype(np.int32),
            m_scores[mask].astype(np.float32),
            np.stack([fx1, fy1, fx2, fy2], axis=1).astype(np.float32),
        )
