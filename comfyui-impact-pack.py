import os, sys, subprocess
from torchvision.datasets.utils import download_url
import platform
import configparser
import folder_paths


# INSTALL
print("### Loading: ComfyUI-Impact-Pack")

comfy_path = os.path.dirname(folder_paths.__file__)
config_path = os.path.join(comfy_path, "custom_nodes", "impact-pack.ini")

js_path = os.path.join(comfy_path, "web", "extensions", "core")
js_version = 2
js_url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Pack/Main/js/impact-pack.js"

def read_js_version():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return int(config['default']['js_version'])
    except:
        return 0

def write_js_version():
    config = configparser.ConfigParser()
    config['default'] = {
        'js_version': js_version
    }
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    

# Download js
if not os.path.exists(os.path.join(js_path, "impact-pack.js")):
    download_url(js_url, js_path)
    write_js_version()
else:
    # update js
    js = read_js_version()
    if js < js_version:
        download_url(js_url, js_path)
        write_js_version()


# ----- SETUP --------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.append('../ComfyUI')


print("### ComfyUI-Impact-Pack: Check dependencies")


def ensure_pip_packages():
    try:
        import segment_anything
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'segment-anything'])

    try:
        from skimage.measure import label, regionprops
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])

    try:
        import onnxruntime
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxruntime'])

    try:
        import pycocotools
    except Exception:
        if platform.system() not in ["Windows"] or platform.machine() not in ["AMD64", "x86_64"]:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools'])
        else:
            pycocotools = {
                (3, 8): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp38-cp38-win_amd64.whl",
                (3, 9): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp39-cp39-win_amd64.whl",
                (3, 10): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp310-cp310-win_amd64.whl",
                (3, 11): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp311-cp311-win_amd64.whl",
            }

            version = sys.version_info[:2]
            url = pycocotools[version]
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', url])


def ensure_mmdet_package():
    try:
        import mmcv
        import mmdet
        from mmdet.evaluation import get_classes
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'])
        subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv==2.0.0'])
        subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmdet==3.0.0'])
        subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmengine==0.7.2'])

ensure_pip_packages()
ensure_mmdet_package()

# Download model
print("### ComfyUI-Impact-Pack: Check basic models")

model_path = folder_paths.models_dir

bbox_path = os.path.join(model_path, "mmdets", "bbox")
#segm_path = os.path.join(model_path, "mmdets", "segm") -- deprecated
sam_path = os.path.join(model_path, "sams")
onnx_path = os.path.join(model_path, "onnx")

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.pth")):
    download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.py")):
    download_url("https://raw.githubusercontent.com/Bing-su/dddetailer/master/config/mmdet_anime-face_yolov3.py", bbox_path)

if not os.path.exists(os.path.join(sam_path, "sam_vit_b_01ec64.pth")):
    download_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_path)

if not os.path.exists(onnx_path):
    print(f"### ComfyUI-Impact-Pack: onnx model directory created ({onnx_path})")
    os.mkdir(onnx_path)

# ----- MAIN CODE --------------------------------------------------------------

# Core
import torch
import cv2
import mmcv
import numpy as np
from mmdet.apis import (inference_detector, init_detector)
import comfy.samplers
import comfy.sd
import nodes
import warnings
from PIL import Image, ImageFilter
from mmdet.evaluation import get_classes
from skimage.measure import label, regionprops
from collections import namedtuple

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label'],
                 defaults=[None])

def load_mmdet(model_path):
    model_config = os.path.splitext(model_path)[0] + ".py"
    model = init_detector(model_config, model_path, device="cpu")
    return model


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w/2, bbox[1] + h/2


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])
            combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def combine_masks2(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i]).astype(np.uint8)
            combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = torch.from_numpy(cv2_mask)
    return mask


def to_binary_mask(mask):
    mask = mask.clone()
    mask[mask != 0] = 1.
    return mask


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return mask

    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    return cv2.dilate(mask, kernel, iter)


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)
    return dilated_masks


def feather_mask(mask, thickness):
    pil_mask = Image.fromarray(np.uint8(mask * 255))

    # Create a feathered mask by applying a Gaussian blur to the mask
    blurred_mask = pil_mask.filter(ImageFilter.GaussianBlur(thickness))
    feathered_mask = Image.new("L", pil_mask.size, 0)
    feathered_mask.paste(blurred_mask, (0, 0), blurred_mask)
    return feathered_mask


def subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = torch.from_numpy(cv2_mask) / 255.0
    return mask


def inference_segm_old(model, image, conf_threshold):
    image = image.numpy()[0] * 255
    mmdet_results = inference_detector(model, image)

    bbox_results, segm_results = mmdet_results
    label = "A"

    classes = get_classes("coco")
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n, m = bbox_results[0].shape
    if n == 0:
        return [[], [], []]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:, -1] > conf_threshold)[0]
    results = [[], [], []]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results


def inference_segm(image, modelname, conf_thres, label):
    image = image.numpy()[0] * 255
    mmdet_results = inference_detector(modelname, image).pred_instances
    bboxes = mmdet_results.bboxes.numpy()
    segms = mmdet_results.masks.numpy()
    scores = mmdet_results.scores.numpy()

    classes = get_classes("coco")

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]
    labels = mmdet_results.labels
    filter_inds = np.where(mmdet_results.scores > conf_thres)[0]
    results = [[], [], [], []]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(scores[i])

    return results


def inference_bbox(modelname, image, conf_threshold):
    image = image.numpy()[0] * 255
    label = "A"
    output = inference_detector(modelname, image).pred_instances
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in output.bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = output.bboxes.shape
    if n == 0:
        return [[], [], [], []]

    bboxes = output.bboxes.numpy()
    scores = output.scores.numpy()
    filter_inds = np.where(scores > conf_threshold)[0]
    results = [[], [], [], []]
    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(scores[i])

    return results


def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    points = []
    plabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(mask.shape[0]/20))
    x_step = max(3, int(mask.shape[1]/20))
    
    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x+j, y+i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x+j, y+i))
                plabs.append(0)
    
    return points, plabs


def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(w/20))
    x_step = max(3, int(h/20))
    
    for i in range(10, h-10, y_step):
        for j in range(10, w-10, x_step):
            if not (x1-10 <= j and j <= x2+10 and y1-10 <= i and i <= y2+10):
                npoints.append((j,i))
                nplabs.append(0)
    
    return npoints, nplabs

# Nodes
# folder_paths.supported_pt_extensions
folder_paths.folder_names_and_paths["mmdets_bbox"] = ([os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets_segm"] = ([os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets"] = ([os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["sams"] = ([os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["onnx"] = ([os.path.join(model_path, "onnx")], set(['.onnx']))


class NO_BBOX_MODEL:
    ERROR = ""


class NO_SEGM_MODEL:
    ERROR = ""


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = limit - size
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2-x1
    bbox_h = y2-y1
    
    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w/2)
    new_y1 = int(kernel_y - crop_h/2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_image(image, crop_region):
    return crop_ndarray4(np.array(image), crop_region)


def to_latent_image(pixels, vae):
    x = (pixels.shape[1] // 64) * 64
    y = (pixels.shape[2] // 64) * 64
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def scale_tensor(w, h, image):
    image = tensor2pil(image)
    scaled_image = image.resize((w, h), resample=LANCZOS)
    return pil2tensor(scaled_image)


def scale_tensor_and_to_pil(w,h, image):
    image = tensor2pil(image)
    return image.resize((w, h), resample=LANCZOS)


def enhance_detail(image, model, vae, guide_size, guide_size_for, bbox, seed, steps, cfg, sampler_name, scheduler,
                   positive, negative, denoise, noise_mask, force_inpaint):

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3]-bbox[1]
    bbox_w = bbox[2]-bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if bbox_h >= guide_size and bbox_w >= guide_size:
        print(f"Detailer: segment skip")
        None

    if guide_size_for == "bbox":
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size/min(bbox_w,bbox_h)
    else:
        # for cropped_size
        upscale = guide_size/min(w,h)

    new_w = int(((w * upscale)//64) * 64)
    new_h = int(((h * upscale)//64) * 64)

    if not force_inpaint:
        if upscale <= 1.0:
            print(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None
        
        if new_w == 0 or new_h == 0:
            print(f"Detailer: segment skip [zero size={new_w,new_h}]")
            return None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            print(f"Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h
            
    print(f"Detailer: segment upscale for ({bbox_w,bbox_h}) | crop region {w,h} x {upscale} -> {new_w,new_h}")

    # upscale
    upscaled_image = scale_tensor(new_w, new_h, torch.from_numpy(image))

    # ksampler
    latent_image = to_latent_image(upscaled_image, vae)

    if noise_mask is not None:
        # upscale the mask tensor by a factor of 2 using bilinear interpolation
        noise_mask = torch.from_numpy(noise_mask)
        upscaled_mask = torch.nn.functional.interpolate(noise_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                                                   mode='bilinear', align_corners=False)

        # remove the extra dimensions added by unsqueeze
        upscaled_mask = upscaled_mask.squeeze().squeeze()
        latent_image['noise_mask'] = upscaled_mask

    sampler = nodes.KSampler()
    refined_latent = sampler.sample(model, seed, steps, cfg, sampler_name, scheduler,
                                    positive, negative, latent_image, denoise)
    refined_latent = refined_latent[0]

    # non-latent downscale - latent downscale cause bad quality
    refined_image = vae.decode(refined_latent['samples'])

    # downscale
    refined_image = scale_tensor_and_to_pil(w, h, refined_image)

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image


def composite_to(dest_latent, crop_region, src_latent):
    x1 = crop_region[0]
    y1 = crop_region[1]

    # composite to original latent
    lc = nodes.LatentComposite()

    # 현재 mask 를 고려한 composite 가 없음... 이거 처리 필요.

    orig_image = lc.composite(dest_latent, src_latent, x1, y1)

    return orig_image[0]


def onnx_inference(image, onnx_model):
    # prepare image
    pil = tensor2pil(image)
    image = np.ascontiguousarray(pil)
    image = image[:, :, ::-1]  # to BGR image
    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68]  # 'caffe' mode image preprocessing

    # do detection
    onnx_model = onnxruntime.InferenceSession(onnx_model)
    outputs = onnx_model.run(
        [s_i.name for s_i in onnx_model.get_outputs()],
        {onnx_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    labels = [op for op in outputs if op.dtype == "int32"][0]
    scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
    boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

    # filter-out useless item
    idx = np.where(labels[0] == -1)[0][0]

    labels = labels[0][:idx]
    scores = scores[0][:idx]
    boxes = boxes[0][:idx].astype(np.uint32)

    return labels, scores, boxes


class MMDetLoader:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("mmdets_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("mmdets_segm")]
        return {"required": {"model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_MODEL", "SEGM_MODEL")
    FUNCTION = "load_mmdet"

    CATEGORY = "ImpactPack"

    def load_mmdet(self, model_name):
        mmdet_path = folder_paths.get_full_path("mmdets", model_name)
        model = load_mmdet(mmdet_path)

        if model_name.startswith("bbox"):
            return model, NO_SEGM_MODEL()
        else:
            return NO_BBOX_MODEL(), model


from segment_anything import SamPredictor
from segment_anything import sam_model_registry
import onnxruntime


class SAMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("sams"), )}}

    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path("sams", model_name)
        sam = sam_model_registry["vit_b"](checkpoint=modelname)
        print(f"Loads SAM model: {modelname}")
        return (sam, )


class ONNXLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("onnx"), )}}

    RETURN_TYPES = ("ONNX_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path("onnx", model_name)
        print(f"Loads ONNX model: {modelname}")
        return (modelname, )


class ONNXDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "onnx_model": ("ONNX_MODEL",),
                    "image": ("IMAGE",),
                    "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "crop_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10, "step": 0.1}),
                    }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    OUTPUT_NODE = True

    def doit(self, onnx_model, image, threshold, crop_factor):
        h = image.shape[1]
        w = image.shape[2]

        labels, scores, boxes = onnx_inference(image, onnx_model)

        # collect feasible item
        result = []

        for i in range(len(labels)):
            if scores[i] > threshold:
                item_bbox = boxes[i]
                x1, y1, x2, y2 = item_bbox

                crop_region = make_crop_region(w, h, item_bbox, crop_factor)
                crop_x1, crop_y1, crop_x2, crop_y2, = crop_region

                # prepare cropped mask
                cropped_mask = np.zeros((crop_y2-crop_y1,crop_x2-crop_x1))
                inner_mask = np.ones((y2-y1,x2-x1))
                cropped_mask[y1-crop_y1:y2-crop_y1, x1-crop_x1:x2-crop_x1] = inner_mask

                # make items
                item = SEG(None, cropped_mask, scores[i], crop_region, item_bbox)
                result.append(item)
                
        return (result,)


class DetailerForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "model": ("MODEL",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 128, "max": nodes.MAX_RESOLUTION, "step": 64}),
                     "guide_size_for": (["bbox", "crop_region"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": (["enabled", "disabled"], ),
                     "force_inpaint": (["disabled", "enabled"], ),
                     },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    @staticmethod
    def do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, force_inpaint):
        
        image_pil = tensor2pil(image).convert('RGBA')

        for seg in segs:
            cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                              else crop_ndarray4(image.numpy(), seg.crop_region)

            mask_pil = feather_mask(seg.cropped_mask, feather)

            if noise_mask == "enabled":
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            enhanced_pil = enhance_detail(cropped_image, model, vae, guide_size, guide_size_for, seg.bbox,
                                          seed, steps, cfg, sampler_name, scheduler,
                                          positive, negative, denoise, cropped_mask, force_inpaint)

            if not (enhanced_pil is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image_pil.paste(enhanced_pil, (seg.crop_region[0], seg.crop_region[1]), mask_pil)

        image_tensor = pil2tensor(image_pil.convert('RGB'))

        if len(segs) > 0:
            enhanced_tensor = pil2tensor(enhanced_pil) if enhanced_pil is not None else None
            return image_tensor, torch.from_numpy(cropped_image), enhanced_tensor,
        else:
            return image_tensor, None, None,

    def doit(self, image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint):

        enhanced_img, cropped, cropped_enhanced = \
            DetailerForEach.do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint)

        return (enhanced_img, )


class DetailerForEachTest(DetailerForEach):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("image","cropped","cropped_refined")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint):

        enhanced_img, cropped, cropped_enhanced = \
            DetailerForEach.do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint)

        # set fallback image
        if cropped is None:
            cropped = enhanced_img

        if cropped_enhanced is None:
            cropped_enhanced = enhanced_img

        return enhanced_img, cropped, cropped_enhanced,


class EmptySEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self):
        return ([],)


class SegsMaskCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs, image):
        h = image.shape[1]
        w = image.shape[2]

        mask = np.zeros((h, w), dtype=np.uint8)

        for seg in segs:
            cropped_mask = seg.cropped_mask
            crop_region = seg.crop_region
            mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

        return (torch.from_numpy(mask.astype(np.float32) / 255.0), )


def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []

    selected = False
    max_score = 0
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected:
        total_masks.append(max_mask)

    return total_masks


class SAMDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "sam_model": ("SAM_MODEL", ),
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                        "detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "none"],),
                        "dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                        "threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                        "mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "mask_hint_use_negative": (["False", "Small", "Outter"], )
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, sam_model, segs, image, detection_hint, dilation,
             threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):

        predictor = SamPredictor(sam_model)
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        predictor.set_image(image, "RGB")

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(segs[i].bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2]-bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_predict(predictor, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])
                
                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]  # 1 = foreground point, 0 = background point

                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.append((x1 + gap, center[1]))
                    points.append((x1 + gap*2, center[1]))
                    plabs = [1, 1]

                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.append((center[0], y1 + gap))
                    points.append((center[0], y1 + gap*2))
                    plabs = [1, 1]

                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, center[1]))
                    points.append((x1 + x_gap*2, center[1]))
                    points.append((center[0], y1 + y_gap))
                    points.append((center[0], y1 + y_gap*2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, y1 + y_gap))
                    points.append((x1 + x_gap*2, y1 + y_gap))
                    points.append((x1 + x_gap, y1 + y_gap*2))
                    points.append((x1 + x_gap*2, y1 + y_gap*2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(segs[i].crop_region[0], segs[i].crop_region[1], segs[i].cropped_mask,
                                                                       mask_hint_threshold, use_small_negative)


                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1], 
                                                         segs[i].crop_region[0], segs[i].crop_region[1], segs[i].crop_region[2], segs[i].crop_region[3])
                    
                    points += npoints
                    plabs += nplabs
                    
                detected_masks = sam_predict(predictor, points, plabs, dilated_bbox, threshold)
                total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

        if mask is not None:
            mask = mask.float()
            mask = dilate_mask(mask.numpy(), dilation)
            mask = torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu") # empty mask

        return (mask, )


class BboxDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_model": ("BBOX_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                      }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, bbox_model, image, threshold, dilation, crop_factor):
        mmdet_results = inference_bbox(bbox_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]
        for x in segmasks:
            item_bbox = x[0]
            item_mask = x[1]

            crop_region = make_crop_region(w, h, item_bbox, crop_factor)
            cropped_image = crop_image(image, crop_region)
            cropped_mask = crop_ndarray2(item_mask, crop_region)
            confidence = x[2]
            # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

            item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox)
            items.append(item)

        return (items, )


class SegmDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_model": ("SEGM_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                      }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, segm_model, image, threshold, dilation, crop_factor):
        mmdet_results = inference_segm(segm_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]
        for x in segmasks:
            item_bbox = x[0]
            item_mask = x[1]

            crop_region = make_crop_region(w, h, item_bbox, crop_factor)
            cropped_image = crop_image(image, crop_region)
            cropped_mask = crop_ndarray2(item_mask, crop_region)
            confidence = x[2]

            item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox)
            items.append(item)

        return (items, )


class SegsBitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS",),
                        "mask": ("MASK",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs, mask):
        if mask is None:
            print("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
            return ([], )

        items = []

        mask = (mask.numpy() * 255).astype(np.uint8)

        for seg in segs:
            cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
            crop_region = seg.crop_region

            cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

            new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
            new_mask = new_mask.astype(np.float32) / 255.0

            item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label)
            items.append(item)

        return (items,)


class BitwiseAndMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "base_segs": ("SEGS",),
                "mask_segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, base_segs, mask_segs):

        result = []

        for bseg in base_segs:
            cropped_mask1 = bseg.cropped_mask.copy()
            crop_region1 = bseg.crop_region

            for mseg in mask_segs:
                cropped_mask2 = mseg.cropped_mask
                crop_region2 = mseg.crop_region

                # compute the intersection of the two crop regions
                intersect_region = (max(crop_region1[0], crop_region2[0]),
                                    max(crop_region1[1], crop_region2[1]),
                                    min(crop_region1[2], crop_region2[2]),
                                    min(crop_region1[3], crop_region2[3]))

                overlapped = False

                # set all pixels in cropped_mask1 to 0 except for those that overlap with cropped_mask2
                for i in range(intersect_region[0], intersect_region[2]):
                    for j in range(intersect_region[1], intersect_region[3]):
                        if cropped_mask1[j - crop_region1[1], i - crop_region1[0]] == 1 and \
                                cropped_mask2[j - crop_region2[1], i - crop_region2[0]] == 1:
                            # pixel overlaps with both masks, keep it as 1
                            overlapped = True
                            pass
                        else:
                            # pixel does not overlap with both masks, set it to 0
                            cropped_mask1[j - crop_region1[1], i - crop_region1[0]] = 0

                if overlapped:
                    item = SEG(bseg.cropped_image, cropped_mask1, bseg.confidence, bseg.crop_region, bseg.bbox, bseg.label)
                    result.append(item)

        return (result,)


class SubtractMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "base_segs": ("SEGS",),
                        "mask_segs": ("SEGS",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, base_segs, mask_segs):

        result = []

        for bseg in base_segs:
            cropped_mask1 = bseg.cropped_mask.copy()
            crop_region1 = bseg.crop_region

            for mseg in mask_segs:
                cropped_mask2 = mseg.cropped_mask
                crop_region2 = mseg.crop_region

                # compute the intersection of the two crop regions
                intersect_region = (max(crop_region1[0], crop_region2[0]),
                                    max(crop_region1[1], crop_region2[1]),
                                    min(crop_region1[2], crop_region2[2]),
                                    min(crop_region1[3], crop_region2[3]))

                changed = False

                # subtract operation
                for i in range(intersect_region[0], intersect_region[2]):
                    for j in range(intersect_region[1], intersect_region[3]):
                        if cropped_mask1[j - crop_region1[1], i - crop_region1[0]] == 1 and \
                                cropped_mask2[j - crop_region2[1], i - crop_region2[0]] == 1:
                            # pixel overlaps with both masks, set it as 0
                            changed = True
                            cropped_mask1[j - crop_region1[1], i - crop_region1[0]] = 0
                        else:
                            # pixel does not overlap with both masks, don't care
                            pass

                if changed:
                    item = SEG(bseg.cropped_image, cropped_mask1, bseg.confidence, bseg.crop_region, bseg.bbox, bseg.label)
                    result.append(item)
                else:
                    result.append(base_segs)

        return (result,)


class MaskToSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "mask": ("MASK",),
                                "combined": (["False", "True"], ),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask, combined, crop_factor):
        if mask is None:
            print("[MaskToSEGS] Cannot operate: MASK is empty.")
            return ([], )

        mask = mask.numpy()

        result = []
        if combined == "True":
            # Find the indices of the non-zero elements
            indices = np.nonzero(mask)

            if len(indices[0]) > 0 and len(indices[1]) > 0:
                # Determine the bounding box of the non-zero elements
                bbox = np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0])
                crop_region = make_crop_region(mask.shape[1], mask.shape[0], bbox, crop_factor)
                x1, y1, x2, y2 = crop_region

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask[y1:y2, x1:x2]
                    item = SEG(None, cropped_mask, 1.0, crop_region, bbox)
                    result.append(item)

        else:
            # label the connected components
            labelled_mask = label(mask)

            # get the region properties for each connected component
            regions = regionprops(labelled_mask)

            # iterate over the regions and print their bounding boxes
            for region in regions:
                y1, x1, y2, x2 = region.bbox
                bbox = x1, y1, x2, y2
                crop_region = make_crop_region(mask.shape[1], mask.shape[0], bbox, crop_factor)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
                    item = SEG(None, cropped_mask, 1.0, crop_region, bbox)
                    result.append(item)

        if not result:
            print(f"[MaskToSEGS] Empty mask.")

        print(f"# of Detected SEGS: {len(result)}")
        # for r in result:
        #     print(f"\tbbox={r.bbox}, crop={r.crop_region}, label={r.label}")

        return (result, )


class SegmDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_model": ("SEGM_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, segm_model, image, threshold, dilation):
        mmdet_results = inference_segm(segm_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        mask = combine_masks(segmasks)
        return (mask,)


class BboxDetectorCombined(SegmDetectorCombined):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
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


class ToBinaryMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                      "mask": ("MASK",),
                    }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask,):
        mask = to_binary_mask(mask)
        return (mask,)


class BitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask1": ("MASK",),
                        "mask2": ("MASK",),
                    }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask1, mask2):
        mask = bitwise_and_masks(mask1, mask2)
        return (mask,)


class SubtractMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask1": ("MASK", ),
                        "mask2": ("MASK", ),
                      }
                }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask1, mask2):
        mask = subtract_masks(mask1, mask2)
        return (mask,)



import nodes


class MaskPainter(nodes.PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ), },
                "hidden": {
                            "prompt": "PROMPT",
                            "extra_pnginfo": "EXTRA_PNGINFO",
                            },
                "optional": {"mask_image": ("IMAGE_PATH", ), },
                }
    
    RETURN_TYPES = ("MASK", )
    
    FUNCTION = "save_painted_images"

    CATEGORY = "ImpactPack"

    def load_mask(self, imagepath):
        if imagepath['type'] == "temp":
            input_dir = folder_paths.get_temp_directory()
        else:
            input_dir = folder_paths.get_input_directory()

        image_path = os.path.join(input_dir, imagepath['filename'])

        if os.path.exists(image_path):
            i = Image.open(image_path)
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (mask, )

    def save_painted_images(self, images, filename_prefix="impact-mask", 
                            prompt=None, extra_pnginfo=None, mask_image=None):
        res = self.save_images(images, filename_prefix, prompt, extra_pnginfo)

        if mask_image is not None:
            res['result'] = self.load_mask(mask_image)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            res['result'] = (mask, )

        return res


NODE_CLASS_MAPPINGS = {
    "MMDetLoader": MMDetLoader,
    "SAMLoader": SAMLoader,
    "ONNXLoader": ONNXLoader,

    "BboxDetectorForEach": BboxDetectorForEach,
    "SegmDetectorForEach": SegmDetectorForEach,
    "ONNXDetectorForEach": ONNXDetectorForEach,

    "BitwiseAndMaskForEach": BitwiseAndMaskForEach,

    "DetailerForEach": DetailerForEach,
    "DetailerForEachDebug": DetailerForEachTest,

    "BboxDetectorCombined": BboxDetectorCombined,
    "SegmDetectorCombined": SegmDetectorCombined,
    "SAMDetectorCombined": SAMDetectorCombined,

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "Segs & Mask": SegsBitwiseAndMask,
    "SegsMaskCombine": SegsMaskCombine,
    "EmptySegs": EmptySEGS,

    "MaskToSEGS": MaskToSEGS,
    "ToBinaryMask": ToBinaryMask,
    
    "MaskPainter": MaskPainter,
}

# Expand Server api

import server
from aiohttp import web


@server.PromptServer.instance.routes.post("/upload/temp")
async def upload_image(request):
    upload_dir = folder_paths.get_temp_directory()

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    post = await request.post()
    image = post.get("image")

    if image and image.file:
        filename = image.filename
        if not filename:
            return web.Response(status=400)

        split = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(upload_dir, filename)):
            filename = f"{split[0]} ({i}){split[1]}"
            i += 1

        filepath = os.path.join(upload_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image.file.read())
        
        return web.json_response({"name": filename})
    else:
        return web.Response(status=400)
