import io
import json
import math
import threading

import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from impact.core import sam_predict


sam_editor_lock = threading.Lock()


def read_request(image_bytes, prompts_json):
    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = np.array(image.convert("RGB"))
    prompts = json.loads(prompts_json)
    if not isinstance(prompts, dict):
        raise ValueError("SAM prompts must be an object.")
    threshold = prompts.get("threshold", 0.7)
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("SAM confidence must be between 0 and 1.")
    points, labels = [], []
    for key, label in (("positive_points", 1), ("negative_points", 0)):
        entries = prompts.get(key, [])
        if not isinstance(entries, list):
            raise ValueError("SAM points must be arrays of [x, y] coordinates.")
        for point in entries:
            if not isinstance(point, list) or len(point) != 2 or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in point
            ):
                raise ValueError("SAM points must contain finite [x, y] coordinates.")
            points.append(point)
            labels.append(label)
    if not points:
        raise ValueError("Add at least one SAM point.")
    return pixels, points, labels, threshold


def detect_mask(image, points, labels, threshold, model_path, device):
    with sam_editor_lock:
        model_kind = "vit_h" if "vit_h" in model_path else "vit_l" if "vit_l" in model_path else "vit_b"
        model = sam_model_registry[model_kind](checkpoint=model_path)
        try:
            model.to(device=device)
            predictor = SamPredictor(model)
            predictor.set_image(image, "RGB")
            masks = sam_predict(predictor, points, labels, None, threshold)
            if not masks:
                raise ValueError("SAM did not return a mask. Try different points.")
            mask = np.logical_or.reduce(masks).astype(np.uint8) * 255
            output = io.BytesIO()
            Image.fromarray(mask).save(output, format="PNG")
            return output.getvalue()
        finally:
            model.cpu()
