import folder_paths
import impact_core as core
from impact_utils import *
from impact_core import SEG
import nodes
import os

class NO_BBOX_MODEL:
    pass


class NO_SEGM_MODEL:
    pass


class MMDetLoader:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("mmdets_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("mmdets_segm")]
        return {"required": {"model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_MODEL", "SEGM_MODEL")
    FUNCTION = "load_mmdet"

    CATEGORY = "ImpactPack/Legacy"

    def load_mmdet(self, model_name):
        mmdet_path = folder_paths.get_full_path("mmdets", model_name)
        model = core.load_mmdet(mmdet_path)

        if model_name.startswith("bbox"):
            return model, NO_SEGM_MODEL()
        else:
            return NO_BBOX_MODEL(), model


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

    CATEGORY = "ImpactPack/Legacy"

    @staticmethod
    def detect(bbox_model, image, threshold, dilation, crop_factor):
        mmdet_results = core.inference_bbox(bbox_model, image, threshold)
        segmasks = core.create_segmasks(mmdet_results)

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

        shape = h, w
        return shape, items

    def doit(self, bbox_model, image, threshold, dilation, crop_factor):
        return (BboxDetectorForEach.detect(bbox_model, image, threshold, dilation, crop_factor), )


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

    CATEGORY = "ImpactPack/Legacy"

    def doit(self, segm_model, image, threshold, dilation):
        mmdet_results = core.inference_segm(image, segm_model, threshold)
        segmasks = core.create_segmasks(mmdet_results)
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
        mmdet_results = core.inference_bbox(bbox_model, image, threshold)
        segmasks = core.create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        mask = combine_masks(segmasks)
        return (mask,)


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

    CATEGORY = "ImpactPack/Legacy"

    def doit(self, segm_model, image, threshold, dilation, crop_factor):
        mmdet_results = core.inference_segm(image, segm_model, threshold)
        segmasks = core.create_segmasks(mmdet_results)

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

        shape = h,w
        return ((shape, items), )


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

    CATEGORY = "ImpactPack/Legacy"

    @staticmethod
    def combine(segs, image):
        h = image.shape[1]
        w = image.shape[2]

        mask = np.zeros((h, w), dtype=np.uint8)

        for seg in segs[1]:
            cropped_mask = seg.cropped_mask
            crop_region = seg.crop_region
            mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

        return torch.from_numpy(mask.astype(np.float32) / 255.0)

    def doit(self, segs, image):
        return (SegsMaskCombine.combine(segs, image), )


class MaskPainter(nodes.PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), },
                "hidden": {
                    "prompt": "PROMPT",
                    "extra_pnginfo": "EXTRA_PNGINFO",
                },
                "optional": {"mask_image": ("IMAGE_PATH",), },
                }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "save_painted_images"

    CATEGORY = "ImpactPack/Legacy"

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
                mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")
        else:
            mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")

        return (mask,)

    def save_painted_images(self, images, filename_prefix="impact-mask",
                            prompt=None, extra_pnginfo=None, mask_image=None):
        res = self.save_images(images, filename_prefix, prompt, extra_pnginfo)

        if mask_image is not None:
            res['result'] = self.load_mask(mask_image)
        else:
            mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")
            res['result'] = (mask,)

        return res