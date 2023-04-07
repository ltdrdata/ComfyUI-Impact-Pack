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

if "segment-anything" not in installed_pip:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'segment-anything'])

installed_mim = packages_mim()

if "mmcv" not in installed_mim:
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv==2.0.0'])

if "mmdet" not in installed_mim:
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmdet==3.0.0'])

if "mmengine" not in installed_mim:
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmengine==0.7.2'])

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
#segm_path = os.path.join(model_path, "mmdets", "segm") -- deprecated
sam_path = os.path.join(model_path, "sams")

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.pth")):
    download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)

if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.py")):
    download_url("https://raw.githubusercontent.com/Bing-su/dddetailer/master/config/mmdet_anime-face_yolov3.py", bbox_path)

if not os.path.exists(os.path.join(sam_path, "sam_vit_b_01ec64.pth")):
    download_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_path)

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
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def load_mmdet(model_path):
    model_config = os.path.splitext(model_path)[0] + ".py"
    model = init_detector(model_config, model_path, device="cpu")
    return model


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


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


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks
    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        item = (segmasks[i][0],dilated_mask,segmasks[i][2])
        dilated_masks.append(item)
    return dilated_masks


from PIL import Image, ImageFilter
from mmdet.evaluation import get_classes


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
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
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


# Nodes
import folder_paths
# folder_paths.supported_pt_extensions
folder_paths.folder_names_and_paths["mmdets_bbox"] = ([os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets_segm"] = ([os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets"] = ([os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["sams"] = ([os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)


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


from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry


class SAMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("sams"), )}}
    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    def load_model(selfself, model_name):
        modelname = folder_paths.get_full_path("sams", model_name)
        sam = sam_model_registry["vit_b"](checkpoint=modelname)
        print(f"Loads SAM model: {modelname}")
        return (sam, )


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = size
        new_startp = 0
    elif startp + size > limit:
        new_startp = limit - size
        new_endp = limit
    else:
        new_startp = startp
        new_endp = startp+size

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2-x1
    bbox_h = y2-y1

    # multiply and normalize size
    # multiply: contains enough around context
    # normalize: make sure size to be times of 64 and minimum size must be greater equal than 64
    crop_w = max(64, (bbox_w * crop_factor // 64) * 64)
    crop_h = max(64, (bbox_h * crop_factor // 64) * 64)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w/2)
    new_y1 = int(kernel_y - crop_h/2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1,new_y1,new_x2,new_y2]


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


def scale_tensor(w,h, image):
    image = tensor2pil(image)
    scaled_image = image.resize((w,h), resample=LANCZOS)
    return pil2tensor(scaled_image)


def scale_tensor_and_to_pil(w,h, image):
    image = tensor2pil(image)
    return image.resize((w,h), resample=LANCZOS)


def enhance_detail(image, model, vae, guide_size, bbox_size, seed, steps, cfg, sampler_name, scheduler,
                   positive, negative, denoise):

    h = image.shape[1]
    w = image.shape[2]

    # Skip processing if the detected bbox is already larger than the guide_size
    if bbox_size[0] >= guide_size and bbox_size[1] >= guide_size:
        print(f"Detailer: segment skip")
        None

    # Scale up based on the smaller dimension between width and height.
    upscale = guide_size/min(bbox_size[0],bbox_size[1])

    new_w = int(((w * upscale)//64) * 64)
    new_h = int(((h * upscale)//64) * 64)


    print(f"Detailer: segment upscale for ({bbox_size}) | crop region {w,h} x {upscale} -> {new_w,new_h}")

    # upscale
    upscaled_image = scale_tensor(new_w, new_h, torch.from_numpy(image))

    # ksampler
    latent_image = to_latent_image(upscaled_image, vae)

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


class DetailerForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "image":("IMAGE", ),
                     "segs": ("SEGS", ),
                     "model": ("MODEL",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 128, "max": nodes.MAX_RESOLUTION, "step": 64}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     }
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, image, segs, model, vae, guide_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather):

        image_pil = tensor2pil(image).convert('RGBA')

        for x in segs:
            cropped_image = x[0]
            mask_pil = feather_mask(x[1], feather)
            confidence = x[2]
            crop_region = x[3]
            bbox_size = x[4]

            enhanced_pil = enhance_detail(cropped_image, model, vae, guide_size, bbox_size,
                                          seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise)

            if not (enhanced_pil is None):
                # don't latent composite &
                # use image paste
                image_pil.paste(enhanced_pil, (crop_region[0], crop_region[1]), mask_pil)

        return (pil2tensor(image_pil.convert('RGB')), )


class DetailerForEachTest(DetailerForEach):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Test"

    def doit(self, image, segs, model, vae, guide_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather):

        image_pil = tensor2pil(image).convert('RGBA')

        for x in segs:
            cropped_image = x[0]
            mask_pil = feather_mask(x[1], feather)
            confidence = x[2]
            crop_region = x[3]
            bbox_size = x[4]

            enhanced_pil = enhance_detail(cropped_image, model, vae, guide_size, bbox_size,
                                          seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise)

            if not (enhanced_pil is None):
                # don't latent composite &
                # use image paste
                image_pil.paste(enhanced_pil, (crop_region[0], crop_region[1]), mask_pil)

        image_tensor = pil2tensor(image_pil.convert('RGB'))

        if len(segs) > 0:
            return image_tensor, torch.from_numpy(cropped_image), pil2tensor(enhanced_pil),
        else:
            return image_tensor, image_tensor, image_tensor,


class SegsMaskCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, segs, image):
        h = image.shape[1]
        w = image.shape[2]

        mask = np.zeros((h, w), dtype=np.uint8)

        for seg in segs:
            cropped_mask = seg[1]
            crop_region = seg[3]
            mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

        return (torch.from_numpy(mask.astype(np.float32) / 255.0), )


class SAMDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "sam_model": ("SAM_MODEL", ),
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                        "dilation": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                        "threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, sam_model, segs, image, dilation, threshold):
        predictor = SamPredictor(sam_model)
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        predictor.set_image(image,"RGB")

        total_masks = []
        for i in range(len(segs)):
            bbox = segs[i][4]
            w,h = bbox[2] - bbox[0], bbox[3] - bbox[1] / 2
            center = bbox[1] + h/2, bbox[0] + w/2

            x1 = max(segs[i][4][0] - dilation, 0)
            y1 = max(segs[i][4][1] - dilation, 0)
            x2 = max(segs[i][4][2] + dilation, image.shape[2])
            y2 = max(segs[i][4][3] + dilation, image.shape[1])

            dilated_bbox = [x1, y1, x2, y2]

            cur_masks, scores, _ = predictor.predict(point_coords=np.array([center]), point_labels=np.array([1]),
                                                     box=np.array([dilated_bbox]))

            selected = False
            max_score = 0
            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_mask = cur_masks[i]

                if scores[i] >= threshold:
                    selected = True
                    total_masks.append(cur_masks[i])
                else:
                    pass

            if not selected:
                total_masks.append(max_mask)

        mask = combine_masks2(total_masks)
        if mask is not None:
            mask = mask.float()

        return (mask, )


class BboxDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_model": ("BBOX_MODEL", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.5, "max": 10, "step": 1}),
                      }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

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

            item = (cropped_image, cropped_mask, confidence, crop_region, item_bbox)
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
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.5, "max": 10, "step": 1}),
                      }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

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
            bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

            item = (cropped_image, cropped_mask, confidence, crop_region, bbox_size)
            items.append(item)

        return (items, )


class SegsBitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "segs": ("SEGS",),
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, segs, mask):
        items = []

        mask = (mask.numpy() * 255).astype(np.uint8)

        for x in segs:
            cropped_mask = (x[1].copy() * 255).astype(np.uint8)
            crop_region = x[3]

            cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

            new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
            new_mask = new_mask.astype(np.float32) / 255.0

            item = (x[0], new_mask, x[2], x[3], x[4])
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

    CATEGORY = "ImpactPack"

    def doit(self, base_segs, mask_segs):

        result = []

        for x in base_segs:
            cropped_mask1 = x[1].copy()
            crop_region1 = x[3]

            for y in mask_segs:
                cropped_mask2 =y[1]
                crop_region2 = y[3]

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
                    item = x[0], cropped_mask1, x[2], x[3], x[4]
                    result.append(item)


        return (result,)


class SubtractMaskForEach:
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

    CATEGORY = "ImpactPack"

    def doit(self, base_segs, mask_segs):

        result = []

        for x in base_segs:
            cropped_mask1 = x[1].copy()
            crop_region1 = x[3]

            for y in mask_segs:
                cropped_mask2 = y[1]
                crop_region2 = y[3]

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
                    item = x[0], cropped_mask1, x[2], x[3], x[4]
                    result.append(item)
                else:
                    result.append(base_segs)

        return (result,)


class SegmDetectorCombined:
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


class BboxDetectorCombined(SegmDetectorCombined):
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
                "mask1": ("MASK",),
                "mask2": ("MASK",),
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
    "SAMLoader": SAMLoader,

    "BboxDetectorForEach": BboxDetectorForEach,
    "SegmDetectorForEach": SegmDetectorForEach,
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
}
