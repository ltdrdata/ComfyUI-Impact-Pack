import impact.core as core
from impact.config import MAX_RESOLUTION
import impact.segs_nodes as segs_nodes
import numpy as np
import impact.utils as utils

class SAMDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "sam_model": ("SAM_MODEL", ),
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                        "detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area",
                                            "mask-points", "mask-point-bbox", "none"],),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
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
        return (core.make_sam_mask(sam_model, segs, image, detection_hint, dilation,
                                   threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative), )


class SAMDetectorSegmented:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "sam_model": ("SAM_MODEL", ),
                        "segs": ("SEGS", ),
                        "image": ("IMAGE", ),
                        "detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area",
                                            "mask-points", "mask-point-bbox", "none"],),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                        "mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "mask_hint_use_negative": (["False", "Small", "Outter"], )
                      }
                }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("combined_mask", "batch_masks")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, sam_model, segs, image, detection_hint, dilation,
             threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
        combined_mask, batch_masks = core.make_sam_mask_segmented(sam_model, segs, image, detection_hint, dilation,
                                                                  threshold, bbox_expansion, mask_hint_threshold,
                                                                  mask_hint_use_negative)
        return (combined_mask, batch_masks, )


class BboxDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                        "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
                      },
                "optional": {"detailer_hook": ("DETAILER_HOOK",), }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, bbox_detector, image, threshold, dilation, crop_factor, drop_size, labels=None, detailer_hook=None):
        segs = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

        if labels is not None and labels != '':
            labels = labels.split(',')
            if len(labels) > 0:
                segs, _ = segs_nodes.SEGSLabelFilter.filter(segs, labels)

        return (segs, )


class SegmDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_detector": ("SEGM_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                        "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
                      },
                "optional": {"detailer_hook": ("DETAILER_HOOK",), }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, segm_detector, image, threshold, dilation, crop_factor, drop_size, labels=None, detailer_hook=None):
        segs = segm_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

        if labels is not None and labels != '':
            labels = labels.split(',')
            if len(labels) > 0:
                segs, _ = segs_nodes.SEGSLabelFilter.filter(segs, labels)

        return (segs, )


class SegmDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_detector": ("SEGM_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, segm_detector, image, threshold, dilation):
        mask = segm_detector.detect_combined(image, threshold, dilation)
        return (mask,)


class BboxDetectorCombined(SegmDetectorCombined):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                      }
                }

    def doit(self, bbox_detector, image, threshold, dilation):
        mask = bbox_detector.detect_combined(image, threshold, dilation)
        return (mask,)


class SimpleDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "sam_model_opt": ("SAM_MODEL", ),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                 }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion,
             sam_mask_hint_threshold, sam_model_opt=None, segm_detector_opt=None):
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size)

        if sam_model_opt is not None:
            mask = core.make_sam_mask(sam_model_opt, segs, image, "center-1", sub_dilation,
                                      sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
            segs = core.segs_bitwise_and_mask(segs, mask)
        elif segm_detector_opt is not None:
            segm_segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size)
            mask = core.segs_to_combined_mask(segm_segs)
            segs = core.segs_bitwise_and_mask(segs, mask)

        return (segs,)


    def doit(self, bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion,
             sam_mask_hint_threshold, sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForEach.detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                            sub_threshold, sub_dilation, sub_bbox_expansion,
                                            sam_mask_hint_threshold, sam_model_opt, segm_detector_opt)


class SimpleDetectorForEachPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "detailer_pipe": ("DETAILER_PIPE", ),
                        "image": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, detailer_pipe, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold):

        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe

        return SimpleDetectorForEach.detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                            sub_threshold, sub_dilation, sub_bbox_expansion,
                                            sam_mask_hint_threshold, sam_model_opt, segm_detector_opt)


class SimpleDetectorForAnimateDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image_frames": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "sam_model_opt": ("SAM_MODEL", ),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                 }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
               sub_threshold, sub_dilation, sub_bbox_expansion,
               sam_mask_hint_threshold, sam_model_opt=None, segm_detector_opt=None):

        # gather segs for all frames
        all_segs = []
        for image in image_frames:
            image = image.unsqueeze(0)
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size)

            if sam_model_opt is not None:
                mask = core.make_sam_mask(sam_model_opt, segs, image, "center-1", sub_dilation,
                                          sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
                segs = core.segs_bitwise_and_mask(segs, mask)
            elif segm_detector_opt is not None:
                segm_segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size)
                mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, mask)

            all_segs.append(segs)

        # create merged masks
        all_masks = []
        for segs in all_segs:
            all_masks += segs_nodes.SEGSToMaskList().doit(segs)[0]

        result_mask = all_masks[0]
        for mask in all_masks[1:]:
            result_mask += mask

        result_mask = utils.to_binary_mask(result_mask, 0.1)

        return segs_nodes.MaskToSEGS().doit(result_mask, False, crop_factor, False, drop_size)

    def doit(self, bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion,
             sam_mask_hint_threshold, sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForAnimateDiff.detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                                   sub_threshold, sub_dilation, sub_bbox_expansion,
                                                   sam_mask_hint_threshold, sam_model_opt, segm_detector_opt)
