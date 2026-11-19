import logging

import impact.core as core
from nodes import MAX_RESOLUTION
import impact.segs_nodes as segs_nodes
import impact.utils as utils
from impact.node_base import ImpactNode
from comfy_api.latest import io
import torch
from impact.core import SEG

SAM_MODEL_TOOLTIP = {"tooltip": "Segment Anything Model for Silhouette Detection.\nBe sure to use the SAM_MODEL loaded through the SAMLoader (Impact) node as input."}
SAM_MODEL_TOOLTIP_OPTIONAL = {"tooltip": "[OPTIONAL]\nSegment Anything Model for Silhouette Detection.\nBe sure to use the SAM_MODEL loaded through the SAMLoader (Impact) node as input.\nGiven this input, it refines the rectangular areas detected by BBOX_DETECTOR into silhouette shapes through SAM.\nsam_model_opt takes priority over segm_detector_opt."}

MASK_HINT_THRESHOLD_TOOLTIP = "When detection_hint is mask-area, points of the SEGS mask are given to SAM (Segment Anything) as hints.\nOnly the points where the mask value is greater than mask_hint_threshold are used."
MASK_HINT_USE_NEGATIVE_TOOLTIP = "Adds negative hints, which mark what SAM (Segment Anything) must leave out of the mask.\nFalse: No negative hint.\nSmall: With mask-points, a SEG narrower than 10 pixels becomes a negative point. With mask-area, the points where the SEGS mask is 0 become negative points.\nOutter: Points sampled at regular intervals outside the area of the SEG. Not used with mask-points."

DILATION_TOOLTIP = "The size in pixels of the kernel that dilates the result mask. A negative value erodes the mask."
DETECTION_HINT_TOOLTIP = {"tooltip": "It is recommended to use only center-1.\ncenter-1 gives SAM (Segment Anything) the rectangular area of each SEG and a single point at its exact center as hints. none gives only the rectangular area.\nThe other options were added during the experimental stage and do not work well."}

BBOX_EXPANSION_TOOLTIP = "The rectangular area of each SEG is expanded by this many pixels on every side before it is given to SAM (Segment Anything) as a hint."
SAM_THRESHOLD_TOOLTIP = "SAM (Segment Anything) proposes several masks for an area, each with a score. The masks that score at least this value are combined; if none does, the best one is used.\nA higher value gives a narrower mask. For a person, it may take only the clothes instead of the whole person."
IMAGE_TOOLTIP = "The image the SEGS were detected in. SEGS hold only the detected areas and not the image, so SAM (Segment Anything) reads this image."

class SAMDetectorCombined(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAMDetectorCombined",
            display_name="SAMDetector (combined)",
            category="ImpactPack/Detector",
            inputs=[
                io.Custom("SAM_MODEL").Input("sam_model", tooltip=SAM_MODEL_TOOLTIP["tooltip"]),
                io.Custom("SEGS").Input("segs", tooltip="The areas found by a detector.\nSAM (Segment Anything) makes a mask for each area of the SEGS, and the masks are combined into a single mask."),
                io.Image.Input("image", tooltip=IMAGE_TOOLTIP),
                io.Combo.Input("detection_hint", options=["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"], tooltip=DETECTION_HINT_TOOLTIP["tooltip"]),
                io.Int.Input("dilation", default=0, min=-512, max=512, step=1, tooltip=DILATION_TOOLTIP),
                io.Float.Input("threshold", default=0.93, min=0.0, max=1.0, step=0.01, tooltip=SAM_THRESHOLD_TOOLTIP),
                io.Int.Input("bbox_expansion", default=0, min=0, max=1000, step=1, tooltip=BBOX_EXPANSION_TOOLTIP),
                io.Float.Input("mask_hint_threshold", default=0.7, min=0.0, max=1.0, step=0.01, tooltip=MASK_HINT_THRESHOLD_TOOLTIP),
                io.Combo.Input("mask_hint_use_negative", options=["False", "Small", "Outter"], tooltip=MASK_HINT_USE_NEGATIVE_TOOLTIP),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, sam_model, segs, image, detection_hint, dilation,
                threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
        return io.NodeOutput(core.make_sam_mask(sam_model, segs, image, detection_hint, dilation,
                                                threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative))


class SAMDetectorSegmented(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAMDetectorSegmented",
            display_name="SAMDetector (segmented)",
            category="ImpactPack/Detector",
            inputs=[
                io.Custom("SAM_MODEL").Input("sam_model", tooltip=SAM_MODEL_TOOLTIP["tooltip"]),
                io.Custom("SEGS").Input("segs", tooltip="The areas found by a detector.\nSAM (Segment Anything) makes a mask for each area of the SEGS. The masks are returned combined into a single mask, and separately as a batch: one mask for each SEG, or with mask-points one mask for all of them."),
                io.Image.Input("image", tooltip=IMAGE_TOOLTIP),
                io.Combo.Input("detection_hint", options=["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"], tooltip=DETECTION_HINT_TOOLTIP["tooltip"]),
                io.Int.Input("dilation", default=0, min=-512, max=512, step=1, tooltip=DILATION_TOOLTIP + "\nOnly combined_mask is dilated."),
                io.Float.Input("threshold", default=0.93, min=0.0, max=1.0, step=0.01, tooltip=SAM_THRESHOLD_TOOLTIP),
                io.Int.Input("bbox_expansion", default=0, min=0, max=1000, step=1, tooltip=BBOX_EXPANSION_TOOLTIP),
                io.Float.Input("mask_hint_threshold", default=0.7, min=0.0, max=1.0, step=0.01, tooltip=MASK_HINT_THRESHOLD_TOOLTIP),
                io.Combo.Input("mask_hint_use_negative", options=["False", "Small", "Outter"], tooltip=MASK_HINT_USE_NEGATIVE_TOOLTIP),
            ],
            outputs=[io.Mask.Output(display_name="combined_mask"), io.Mask.Output(display_name="batch_masks")],
        )

    @classmethod
    def execute(cls, sam_model, segs, image, detection_hint, dilation,
                threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
        combined_mask, batch_masks = core.make_sam_mask_segmented(sam_model, segs, image, detection_hint, dilation,
                                                                  threshold, bbox_expansion, mask_hint_threshold,
                                                                  mask_hint_use_negative)
        return io.NodeOutput(combined_mask, batch_masks)


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
        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: BboxDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

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
        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SegmDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

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

        if mask is None:
            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

        return (mask.unsqueeze(0),)


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

        if mask is None:
            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

        return (mask.unsqueeze(0),)


class SimpleDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "post_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "sam_model_opt": ("SAM_MODEL", SAM_MODEL_TOOLTIP_OPTIONAL),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                      }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
               sub_threshold, sub_dilation, sub_bbox_expansion,
               sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None, segm_detector_opt=None,
               detailer_hook=None):
        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SimpleDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        if segm_detector_opt is not None and hasattr(segm_detector_opt, 'bbox_detector') and segm_detector_opt.bbox_detector == bbox_detector:
            # Better segm support for YOLO-World detector
            segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size, detailer_hook=detailer_hook)
        else:
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size, detailer_hook=detailer_hook)

            if sam_model_opt is not None:
                mask = core.make_sam_mask(sam_model_opt, segs, image, "center-1", sub_dilation,
                                          sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
                segs = core.segs_bitwise_and_mask(segs, mask)
            elif segm_detector_opt is not None:
                segm_segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size, detailer_hook=detailer_hook)
                mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, mask)

        segs = core.dilate_segs(segs, post_dilation)

        return (segs,)

    def doit(self, bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion,
             sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForEach.detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                            sub_threshold, sub_dilation, sub_bbox_expansion,
                                            sam_mask_hint_threshold, post_dilation=post_dilation,
                                            sam_model_opt=sam_model_opt, segm_detector_opt=segm_detector_opt)


class SimpleDetectorForEachPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "detailer_pipe": ("DETAILER_PIPE", ),
                        "image": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "post_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                      }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, detailer_pipe, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold, post_dilation=0):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SimpleDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, refiner_model, refiner_clip, refiner_positive, refiner_negative = detailer_pipe

        return SimpleDetectorForEach.detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                            sub_threshold, sub_dilation, sub_bbox_expansion,
                                            sam_mask_hint_threshold, post_dilation=post_dilation, sam_model_opt=sam_model_opt, segm_detector_opt=segm_detector_opt,
                                            detailer_hook=detailer_hook)

def track_video_segs(tracker, bbox_detector, image_frames, bbox_threshold, crop_factor, drop_size, cutoff):
    """SEGS of the objects that the detector finds in the first frame, each with its mask in every frame.
    A pixel is in a mask where the tracker gives at least `cutoff`."""
    # ---- Detect bboxes ----
    segs = bbox_detector.detect(image_frames[0].unsqueeze(0), bbox_threshold, 0, 0, drop_size)

      # ---- If no detections, try reversed frames before giving up ----
    if len(segs[1]) == 0:
        reversed_frames = torch.flip(image_frames, dims=[0])
        segs_rev = bbox_detector.detect(reversed_frames[0].unsqueeze(0), bbox_threshold, 0, 0, drop_size)

        if len(segs_rev[1]) == 0:
            # No Bboxes when reversed -> Give up
            h, w = image_frames.shape[1:3]
            return (h, w), []
            
        # ---- Predict masks in reversed mode ----
        segs_masks = tracker.predict_video_segs(reversed_frames, segs_rev)

        # segs_masks wieder umdrehen, damit sie mit Originalframes matchen
        for k in segs_masks.keys():
            segs_masks[k] = torch.flip(segs_masks[k], dims=[0])
    else:
        # ---- Predict masks if BBOXES were found in forward pass----
        segs_masks = tracker.predict_video_segs(image_frames, segs)

    def get_whole_merged_mask(all_masks):
        merged_mask = (all_masks[0] * 255).to(torch.uint8)
        for mask in all_masks[1:]:
            merged_mask |= (mask * 255).to(torch.uint8)

        merged_mask = (merged_mask / 255.0).to(torch.float32)
        merged_mask = utils.to_binary_mask(merged_mask, 0.1)[0]
        return merged_mask

    new_segs = []
    for k, v in segs_masks.items():
        v = v.squeeze(3)
        m = get_whole_merged_mask(v)
        seg = segs_nodes.MaskToSEGS.doit(m, False, crop_factor, False, drop_size, contour_fill=True)[0][1]

        if len(seg) == 0:
            continue

        seg = seg[0]

        x1, y1, x2, y2 = seg.crop_region
        masks = []
        for mask in v:
            masks.append(mask[y1:y2, x1:x2])
        cropped_mask = torch.stack(masks)
        cropped_mask = (cropped_mask >= cutoff).to(torch.uint8).cpu()

        new_seg = SEG(
            seg.cropped_image,
            cropped_mask,
            seg.confidence,
            seg.crop_region,
            seg.bbox,
            seg.label,
            seg.control_net_wrapper
        )
        new_segs.append(new_seg)

    return segs[0], new_segs


class SAM2VideoDetectorSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_frames": ("IMAGE", ),

            "bbox_detector": ("BBOX_DETECTOR", ),
            "sam2_model": ("SAM_MODEL", ),

            "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "sam2_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            }
        }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def doit(bbox_detector, sam2_model, image_frames, bbox_threshold, sam2_threshold, crop_factor, drop_size):
        # ---- Check SAM2 model ----
        if not isinstance(sam2_model, core.SAM2Wrapper):
            logging.error("[Impact Pack] To use the SAM2VideoDetectorSEGS node, a valid SAM2 model must be provided as input to `sam2_model`.")
            raise Exception("To use the SAM2VideoDetectorSEGS node, a SAM2 model must be provided as input to `sam2_model`.")

        return (track_video_segs(sam2_model, bbox_detector, image_frames, bbox_threshold, crop_factor, drop_size, sam2_threshold * 100 - 50), )


class SAM3VideoDetectorSEGS(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImpactSAM3VideoDetectorSEGS",
            display_name="SAM3 Video Detector (SEGS)",
            category="ImpactPack/Detector",
            description="Detects objects in the first frame and tracks them through the frames with a SAM3 model. Returns SEGS with a mask for every frame.",
            inputs=[
                io.Image.Input("image_frames", tooltip="The frames of the video, as a batch of images."),
                io.Custom("BBOX_DETECTOR").Input("bbox_detector", tooltip="Finds the objects to track in the first frame. When it finds none there, the last frame is used and the video is tracked backwards."),
                io.Custom("SAM_MODEL").Input("sam3_model", tooltip="A SAM3 model loaded through SAMLoader (Impact)."),
                io.Float.Input("bbox_threshold", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Objects that the detector scores below this value are not tracked."),
                io.Float.Input("crop_factor", default=3.0, min=1.0, max=100, step=0.1, tooltip="How much wider than the tracked object the crop region of each SEG is."),
                io.Int.Input("drop_size", default=10, min=1, max=MAX_RESOLUTION, step=1, tooltip="Objects smaller than this many pixels in width or height are dropped."),
            ],
            outputs=[io.Custom("SEGS").Output()],
        )

    @classmethod
    def execute(cls, image_frames, bbox_detector, sam3_model, bbox_threshold, crop_factor, drop_size) -> io.NodeOutput:
        tracker = core.sam_wrapper_of(sam3_model)
        if not isinstance(tracker, core.SAM3Wrapper):
            raise Exception("To use the SAM3 Video Detector (SEGS) node, a SAM3 model must be provided as input to `sam3_model`.")

        # SAM3Wrapper returns masks of 0 and 1, where SAM2Wrapper returns logits
        return io.NodeOutput(track_video_segs(tracker, bbox_detector, image_frames, bbox_threshold, crop_factor, drop_size, 0.5))


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
                        "masking_mode": (["Pivot SEGS", "Combine neighboring frames", "Don't combine"],),
                        "segs_pivot": (["Combined mask", "1st frame mask"],),
                        "sam_model_opt": ("SAM_MODEL", SAM_MODEL_TOOLTIP_OPTIONAL),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                 }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
               sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
               masking_mode="Pivot SEGS", segs_pivot="Combined mask", sam_model_opt=None, segm_detector_opt=None):

        h = image_frames.shape[1]
        w = image_frames.shape[2]

        # gather segs for all frames
        segs_by_frames = []
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

            segs_by_frames.append(segs)

        def get_masked_frames():
            masks_by_frame = []
            for i, segs in enumerate(segs_by_frames):
                masks_in_frame = segs_nodes.SEGSToMaskList().doit(segs)[0]
                current_frame_mask = (masks_in_frame[0] * 255).to(torch.uint8)

                for mask in masks_in_frame[1:]:
                    current_frame_mask |= (mask * 255).to(torch.uint8)

                current_frame_mask = (current_frame_mask/255.0).to(torch.float32)
                current_frame_mask = utils.to_binary_mask(current_frame_mask, 0.1)[0]

                masks_by_frame.append(current_frame_mask)

            return masks_by_frame

        def get_empty_mask():
            return torch.zeros((h, w), dtype=torch.float32, device="cpu")

        def get_neighboring_mask_at(i, masks_by_frame):
            prv = masks_by_frame[i-1] if i > 1 else get_empty_mask()
            cur = masks_by_frame[i]
            nxt = masks_by_frame[i-1] if i > 1 else get_empty_mask()

            prv = prv if prv is not None else get_empty_mask()
            cur = cur.clone() if cur is not None else get_empty_mask()
            nxt = nxt if nxt is not None else get_empty_mask()

            return prv, cur, nxt

        def get_merged_neighboring_mask(masks_by_frame):
            if len(masks_by_frame) <= 1:
                return masks_by_frame

            result = []
            for i in range(0, len(masks_by_frame)):
                prv, cur, nxt = get_neighboring_mask_at(i, masks_by_frame)
                cur = (cur * 255).to(torch.uint8)
                cur |= (prv * 255).to(torch.uint8)
                cur |= (nxt * 255).to(torch.uint8)
                cur = (cur / 255.0).to(torch.float32)
                cur = utils.to_binary_mask(cur, 0.1)[0]
                result.append(cur)

            return result

        def get_whole_merged_mask():
            all_masks = []
            for segs in segs_by_frames:
                all_masks += segs_nodes.SEGSToMaskList().doit(segs)[0]

            merged_mask = (all_masks[0] * 255).to(torch.uint8)
            for mask in all_masks[1:]:
                merged_mask |= (mask * 255).to(torch.uint8)

            merged_mask = (merged_mask / 255.0).to(torch.float32)
            merged_mask = utils.to_binary_mask(merged_mask, 0.1)[0]
            return merged_mask

        def get_pivot_segs():
            if segs_pivot == "1st frame mask":
                return segs_by_frames[0][1]
            else:
                merged_mask = get_whole_merged_mask()
                return segs_nodes.MaskToSEGS.doit(merged_mask, False, crop_factor, False, drop_size, contour_fill=True)[0]

        def get_segs(merged_neighboring=False):
            pivot_segs = get_pivot_segs()

            masks_by_frame = get_masked_frames()
            if merged_neighboring:
                masks_by_frame = get_merged_neighboring_mask(masks_by_frame)

            new_segs = []
            for seg in pivot_segs[1]:
                cropped_mask = torch.zeros(seg.cropped_mask.shape, dtype=torch.float32, device="cpu").unsqueeze(0)
                pivot_mask = torch.from_numpy(seg.cropped_mask)
                x1, y1, x2, y2 = seg.crop_region
                for mask in masks_by_frame:
                    cropped_mask_at_frame = (mask[y1:y2, x1:x2] * pivot_mask).unsqueeze(0)
                    cropped_mask = torch.cat((cropped_mask, cropped_mask_at_frame), dim=0)

                if len(cropped_mask) > 1:
                    cropped_mask = cropped_mask[1:]

                new_seg = SEG(seg.cropped_image, cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                new_segs.append(new_seg)

            return pivot_segs[0], new_segs

        # create result mask
        if masking_mode == "Pivot SEGS":
            return (get_pivot_segs(), )

        elif masking_mode == "Combine neighboring frames":
            return (get_segs(merged_neighboring=True), )

        else: # elif masking_mode == "Don't combine":
            return (get_segs(merged_neighboring=False), )

    def doit(self, bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
             masking_mode="Pivot SEGS", segs_pivot="Combined mask", sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForAnimateDiff.detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                                   sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
                                                   masking_mode, segs_pivot, sam_model_opt, segm_detector_opt)
