import os, sys, subprocess
from torchvision.datasets.utils import download_url

# ----- SETUP --------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.append('../ComfyUI')

def packages_pip():
    import sys, subprocess
    return [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

def packages_mim():
    import sys, subprocess
    return [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'mim', 'list']).split()]

# INSTALL
print("Loading: ComfyUI-Impact-Pack")
print("### ComfyUI-Impact-Pack: Check dependencies")
installed_pip = packages_pip()

if "openmim" not in installed_pip:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'])

installed_mim = packages_mim()

if "mmcv-full" not in installed_mim:
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv-full==1.7.0'])

if "mmdet" not in installed_mim:
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmdet==2.28.2'])

# Download model
print("### ComfyUI-Impact-Pack: Check basic models")

if os.path.realpath("..").endswith("custom_nodes"):
    # For user
    comfy_path = os.path.realpath("../..")
else:
    # For development
    comfy_path = os.path.realpath("../ComfyUI")

model_path = os.path.join(comfy_path, "models")
bbox_path = os.path.join(model_path, "mmdets", "bbox")
segm_path = os.path.join(model_path, "mmdets", "segm")

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.pth")):
    download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.py")):
    download_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)

if not os.path.exists(os.path.join(segm_path, "mmdet_dd-person_mask2former.pth")):
    download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)

if not os.path.exists(os.path.join(segm_path, "mmdet_dd-person_mask2former.py")):
    download_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/segm/mmdet_dd-person_mask2former.py", segm_path)


# ----- MAIN CODE --------------------------------------------------------------

# Core
import torch
import cv2
import mmcv
import numpy as np
from mmdet.core import get_classes
from mmdet.apis import (inference_detector,
                        init_detector)
from PIL import Image

import model_management

def load_mmdet(model_path):
    model_config = os.path.splitext(model_path)[0] + ".py"
    model = init_detector(model_config, model_path, device="cpu")
    return model

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def create_segmasks(results):
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        segmasks.append(segms[i].astype(np.float32))
    return segmasks

def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)

    # combined_mask = Image.fromarray(combined_cv2_mask)
    # return combined_mask
    mask = torch.from_numpy(combined_cv2_mask)
    return mask

def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = torch.from_numpy(cv2_mask)
    return mask

def dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks
    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = masks[i]
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        dilated_masks.append(dilated_mask)
    return dilated_masks

def subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = torch.from_numpy(cv2_mask) / 255.0
    return mask

def inference_segm(model, image, conf_threshold):
    image = image.numpy()[0] * 255
    mmdet_results = inference_detector(model, image)

    bbox_results, segm_results = mmdet_results
    label = "A"

    classes = get_classes("coco")
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n,m = bbox_results[0].shape
    if (n == 0):
        return [[],[],[]]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:,-1] > conf_threshold)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

    return mmdet_results

def inference_bbox(model, image, conf_threshold):
    image = image.numpy()[0] * 255
    label = "A"
    results = inference_detector(model, image)
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for (x0, y0, x1, y1, conf) in results[0]:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = results[0].shape
    if (n == 0):
        return [[], [], []]
    bboxes = np.vstack(results[0])
    filter_inds = np.where(bboxes[:, -1] > conf_threshold)[0]
    results = [[], [], []]
    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

# Nodes
import folder_paths
# folder_paths.supported_pt_extensions
folder_paths.folder_names_and_paths["mmdets_bbox"] = ([os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets_segm"] = ([os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets"] = ([os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)

class NO_BBOX_MODEL:
    ERROR = ""

class NO_SEGM_MODEL:
    ERROR = ""

class MMDetLoader:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = [ "bbox/"+x for x in folder_paths.get_filename_list("mmdets_bbox") ]
        segms = [ "segm/"+x for x in folder_paths.get_filename_list("mmdets_segm") ]
        return {"required": { "model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_MODEL", "SEGM_MODEL")
    FUNCTION = "load_mmdet"

    CATEGORY = "ImpactPack"

    def load_mmdet(self, model_name):
        mmdet_path = folder_paths.get_full_path("mmdets", model_name)
        model = load_mmdet(mmdet_path)

        if model_name.startswith("bbox"):
            return (model, NO_SEGM_MODEL())
        else:
            return (NO_BBOX_MODEL(), model)

class SegmDetector:
    input_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "segm_model": ("SEGM_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, segm_model, image, threshold, dilation):
        mmdet_results = inference_segm(segm_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)
        mask = combine_masks(segmasks)
        return (mask,)

class BboxDetector(SegmDetector):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "bbox_model": ("BBOX_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 4, "min": 0, "max": 255, "step": 1}),
                      }
                }


    def doit(self, bbox_model, image, threshold, dilation):
        mmdet_results = inference_bbox(bbox_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)
        mask = combine_masks(segmasks)
        return (mask,)

class BitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "mask1": ("MASK", ),
                        "mask2": ("MASK", ),
                      }
                }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, mask1, mask2):
        mask = bitwise_and_masks(mask1, mask2)
        return (mask,)
    
class SubtractMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "mask1": ("MASK", ),
                        "mask2": ("MASK", ),
                      }
                }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, mask1, mask2):
        mask = subtract_masks(mask1, mask2)
        return (mask,)


NODE_CLASS_MAPPINGS = {
    "MMDetLoader": MMDetLoader,
    "BboxDetector": BboxDetector,
    "SegmDetector": SegmDetector,
    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
}
