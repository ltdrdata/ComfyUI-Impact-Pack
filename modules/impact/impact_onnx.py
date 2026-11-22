import ast
import impact.additional_dependencies
import cv2
import numpy as np
import torch
import torchvision
from impact import utils

impact.additional_dependencies.ensure_onnx_package()

import onnxruntime

UNSUPPORTED = ("ONNXDetectorProvider cannot use this model. It takes a detection model exported by Ultralytics YOLO, "
               "and a RetinaNet model in the format of detector_v2_base_checkpoint.onnx.")

NOISE_SCORE = 0.001

RETINANET_OUTPUTS = {"labels": ("tensor(int32)", 2), "scores": ("tensor(float)", 2), "boxes": ("tensor(float)", 3)}


def yolo_inference(image, onnx_model):
    metadata = onnx_model.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata["names"]) if "names" in metadata else None
    model_input = onnx_model.get_inputs()[0]
    exported_size = ast.literal_eval(metadata.get("imgsz", "[640, 640]"))
    input_h, input_w = [size if isinstance(size, int) else exported_size[i] for i, size in enumerate(model_input.shape[2:])]

    h, w = image.shape[1:3]
    ratio = min(input_h / h, input_w / w)
    fit_w, fit_h = round(w * ratio), round(h * ratio)
    left, top = (input_w - fit_w) // 2, (input_h - fit_h) // 2
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.float32)
    canvas[top:top + fit_h, left:left + fit_w] = cv2.resize(np.array(utils.tensor2pil(image)), (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)

    output = onnx_model.run(None, {model_input.name: (canvas / 255).transpose(2, 0, 1)[None]})[0]
    if output.ndim != 3:
        raise ValueError(UNSUPPORTED)
    output = output[0]

    if 4 < output.shape[0] < output.shape[1]:
        # a row for each value and a column for each anchor: cx, cy, w, h, then a score for each class
        output = output.T
        class_scores = output[:, 4:4 + len(names)] if names else output[:, 4:]
        labels = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        boxes = np.concatenate([output[:, :2] - output[:, 2:4] / 2, output[:, :2] + output[:, 2:4] / 2], axis=1)

        candidates = scores > NOISE_SCORE
        labels, scores, boxes = labels[candidates], scores[candidates], boxes[candidates]
        kept = torchvision.ops.batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels), 0.7).numpy()
        labels, scores, boxes = labels[kept], scores[kept], boxes[kept]
    elif output.shape[1] == 6:
        # the model filters its own boxes: x1, y1, x2, y2, score, class
        kept = output[:, 4] > NOISE_SCORE
        labels, scores, boxes = output[kept, 5].astype(np.int64), output[kept, 4], output[kept, :4]
    else:
        raise ValueError(UNSUPPORTED)

    boxes = ((boxes - [left, top, left, top]) / ratio).clip(0, [w, h, w, h]).astype(np.uint32)

    if names:
        labels = [names[label] for label in labels]

    return labels, scores, boxes


def onnx_inference(image, onnx_model):
    onnx_model = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    input_shape = onnx_model.get_inputs()[0].shape
    if len(input_shape) != 4:
        raise ValueError(UNSUPPORTED)
    if input_shape[1] == 3:
        return yolo_inference(image, onnx_model)
    if input_shape[3] != 3 or sorted((output.type, len(output.shape)) for output in onnx_model.get_outputs()) != sorted(RETINANET_OUTPUTS.values()):
        raise ValueError(UNSUPPORTED)

    pil = utils.tensor2pil(image)
    image = np.ascontiguousarray(pil)
    image = image[:, :, ::-1]
    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68]  # 'caffe' mode image preprocessing

    outputs = onnx_model.run(
        [s_i.name for s_i in onnx_model.get_outputs()],
        {onnx_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    labels = [op for op in outputs if op.dtype == "int32"][0]
    scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
    boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

    used = labels[0] != -1

    labels = labels[0][used]
    scores = scores[0][used]
    boxes = boxes[0][used].astype(np.uint32)

    return labels, scores, boxes
