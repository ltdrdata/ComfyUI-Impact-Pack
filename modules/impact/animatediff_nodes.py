from nodes import MAX_RESOLUTION
from impact.utils import *
import impact.core as core
from impact.core import SEG
from impact.segs_nodes import SEGSPaste


class SEGSDetailerForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                     "image_frames": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (core.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "basic_pipe": ("BASIC_PIPE",),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0})
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     # TODO: "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     # TODO: "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                     }
                }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "cnet_images")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image_frames.shape)

        new_segs = []
        cnet_image_list = []

        for seg in segs[1]:
            cropped_image_frames = None

            for image in image_frames:
                image = image.unsqueeze(0)
                cropped_image = seg.cropped_image if seg.cropped_image is not None else crop_tensor4(image, seg.crop_region)
                cropped_image = to_tensor(cropped_image)
                if cropped_image_frames is None:
                    cropped_image_frames = cropped_image
                else:
                    cropped_image_frames = torch.concat((cropped_image_frames, cropped_image), dim=0)

            cropped_image_frames = cropped_image_frames.cpu().numpy()

            # It is assumed that AnimateDiff does not support conditioning masks based on test results, but it will be added for future consideration.
            cropped_positive = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in positive
            ]

            cropped_negative = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in negative
            ]

            enhanced_image_tensor, cnet_images = core.enhance_detail_for_animatediff(cropped_image_frames, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                                     seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                                     cropped_positive, cropped_negative, denoise, seg.cropped_mask,
                                                                                     refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                                     refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                                                                     refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                                                                                     inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)
            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(new_cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
            new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_image_list

    def doit(self, image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        segs, cnet_images = SEGSDetailerForAnimateDiff.do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                                 scheduler, denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt,
                                                                 inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

        if len(cnet_images) == 0:
            cnet_images = [empty_pil_tensor()]

        return (segs, cnet_images)


class DetailerForEachPipeForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "image_frames": ("IMAGE", ),
                      "segs": ("SEGS", ),
                      "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                      "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                      "scheduler": (core.SCHEDULERS,),
                      "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                      "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                      "basic_pipe": ("BASIC_PIPE", ),
                      "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     },
                "optional": {
                     "detailer_hook": ("DETAILER_HOOK",),
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     # "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     # "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "basic_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def doit(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, basic_pipe, refiner_ratio=None, detailer_hook=None, refiner_basic_pipe_opt=None,
             inpaint_model=False, noise_mask_feather=0):

        enhanced_segs = []
        cnet_image_list = []

        for sub_seg in segs[1]:
            single_seg = segs[0], [sub_seg]
            enhanced_seg, cnet_images = SEGSDetailerForAnimateDiff().do_detail(image_frames, single_seg, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                                                                               denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt, inpaint_model, noise_mask_feather)

            image_frames = SEGSPaste.doit(image_frames, enhanced_seg, feather, alpha=255)[0]

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if detailer_hook is not None:
                image_frames = detailer_hook.post_paste(image_frames)

            enhanced_segs += enhanced_seg[1]

        new_segs = segs[0], enhanced_segs
        return image_frames, new_segs, basic_pipe, cnet_image_list
