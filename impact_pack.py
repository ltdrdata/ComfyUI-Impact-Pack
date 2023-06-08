import os
import sys
import folder_paths
import comfy.samplers
import comfy.sd
import warnings
from segment_anything import sam_model_registry

from impact_utils import *
import impact_core as core
from impact_core import SEG, NO_BBOX_DETECTOR, NO_SEGM_DETECTOR
from impact_config import MAX_RESOLUTION

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

model_path = folder_paths.models_dir


# Nodes
# folder_paths.supported_pt_extensions
folder_paths.folder_names_and_paths["mmdets_bbox"] = ([os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets_segm"] = ([os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["mmdets"] = ([os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["sams"] = ([os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["onnx"] = ([os.path.join(model_path, "onnx")], {'.onnx'})


class ONNXDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("onnx"), )}}

    RETURN_TYPES = ("ONNX_DETECTOR", )
    FUNCTION = "load_onnx"

    CATEGORY = "ImpactPack"

    def load_onnx(self, model_name):
        model = folder_paths.get_full_path("onnx", model_name)
        return (core.ONNXDetector(model), )


class MMDetDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("mmdets_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("mmdets_segm")]
        return {"required": {"model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "load_mmdet"

    CATEGORY = "ImpactPack"

    def load_mmdet(self, model_name):
        mmdet_path = folder_paths.get_full_path("mmdets", model_name)
        model = core.load_mmdet(mmdet_path)

        if model_name.startswith("bbox"):
            return core.BBoxDetector(model), NO_SEGM_DETECTOR()
        else:
            return NO_BBOX_DETECTOR(), model


class CLIPSegDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text": ("STRING", {"multiline": False}),
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }

    RETURN_TYPES = ("BBOX_DETECTOR", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, text, blur, threshold, dilation_factor):
        try:
            import custom_nodes.clipseg
            return (core.BBoxDetectorBasedOnCLIPSeg(text, blur, threshold, dilation_factor), )
        except Exception as e:
            print("[ERROR] CLIPSegToBboxDetector: CLIPSeg custom node isn't installed. You must install biegert/ComfyUI-CLIPSeg extension to use this node.")
            print(f"\t{e}")
            pass


class SAMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("sams"), )}}

    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path("sams", model_name)

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        print(f"Loads SAM model: {modelname}")
        return (sam, )


class ONNXDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "onnx_detector": ("ONNX_DETECTOR",),
                    "image": ("IMAGE",),
                    "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                    "crop_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10, "step": 0.1}),
                    "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                    }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    OUTPUT_NODE = True

    def doit(self, onnx_detector, image, threshold, dilation, crop_factor, drop_size):
        segs = onnx_detector.detect(image, threshold, dilation, crop_factor, drop_size)
        return (segs, )


class DetailerForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "model": ("MODEL",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
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

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, force_inpaint):

        image_pil = tensor2pil(image).convert('RGBA')

        for seg in segs[1]:
            cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                              else crop_ndarray4(image.numpy(), seg.crop_region)

            mask_pil = feather_mask(seg.cropped_mask, feather)

            if noise_mask == "enabled":
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            enhanced_pil = core.enhance_detail(cropped_image, model, vae, guide_size, guide_size_for, seg.bbox,
                                          seed, steps, cfg, sampler_name, scheduler,
                                          positive, negative, denoise, cropped_mask, force_inpaint == "enabled")

            if not (enhanced_pil is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image_pil.paste(enhanced_pil, (seg.crop_region[0], seg.crop_region[1]), mask_pil)

        image_tensor = pil2tensor(image_pil.convert('RGB'))

        if len(segs[1]) > 0:
            enhanced_tensor = pil2tensor(enhanced_pil) if enhanced_pil is not None else image_tensor
            return image_tensor, torch.from_numpy(cropped_image), enhanced_tensor,
        else:
            return image_tensor, image_tensor, image_tensor,

    def doit(self, image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint):

        enhanced_img, cropped, cropped_enhanced = \
            DetailerForEach.do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint)

        return (enhanced_img, )


class DetailerForEachPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": (["bbox", "crop_region"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": (["enabled", "disabled"], ),
                     "force_inpaint": (["disabled", "enabled"], ),
                     "basic_pipe": ("BASIC_PIPE", )
                     },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, segs, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, noise_mask, force_inpaint, basic_pipe):

        model, _, vae, positive, negative = basic_pipe
        enhanced_img, cropped, cropped_enhanced = \
            DetailerForEach.do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint)

        return (enhanced_img, )


class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = core.KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise)
        return (sampler, )


class TwoSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "latent_image": ("LATENT", ),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", )
                     },
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, latent_image, base_sampler, mask_sampler, mask):
        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample(latent_image)[0]

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)[0]

        del new_latent_image['noise_mask']

        return (new_latent_image, )


class FaceDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "model": ("MODEL",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
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

                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     "sam_dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                     "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                     "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                     "bbox_detector": ("BBOX_DETECTOR", ),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", )
    RETURN_NAMES = ("image", "cropped_refined", "mask", "detailer_pipe")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Simple"

    @staticmethod
    def enhance_face(image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector, sam_model_opt=None):
        # make default prompt as 'face' if empty prompt for CLIPSeg
        bbox_detector.setAux('face')
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
        bbox_detector.setAux(None)

        # bbox + sam combination
        if sam_model_opt is not None:
            sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                     sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                     sam_mask_hint_use_negative, )
            segs = core.segs_bitwise_and_mask(segs, sam_mask)

        enhanced_img, _, cropped_enhanced = \
            DetailerForEach.do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint)

        # Mask Generator
        mask = core.segs_to_combined_mask(segs)

        return enhanced_img, cropped_enhanced, mask

    def doit(self, image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector, sam_model_opt=None):

        enhanced_img, cropped_enhanced, mask = FaceDetailer.enhance_face(
            image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
            sam_mask_hint_use_negative, drop_size, bbox_detector, sam_model_opt)

        pipe = (model, vae, positive, negative, bbox_detector, sam_model_opt)
        return enhanced_img, cropped_enhanced, mask, pipe


class LatentPixelScale:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "scale_method": (s.upscale_methods,),
                     "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10000, "step": 0.1}),
                     "vae": ("VAE", ),
                     "use_tiled_vae": (["disabled", "enabled"],),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, samples, scale_method, scale_factor, vae, use_tiled_vae, upscale_model_opt=None):
        use_tile = use_tiled_vae == "enabled"
        if upscale_model_opt is None:
            latent = core.latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=use_tile)
        else:
            latent = core.latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model_opt, scale_factor, vae, use_tile=use_tile)
        return (latent,)


class CfgScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "target_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, schedule_for_iteration, target_cfg):
        hook = None
        if schedule_for_iteration == "simple":
            hook = core.SimpleCfgScheduleHook(target_cfg)

        return (hook, )


class DenoiseScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "target_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 100.0}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, schedule_for_iteration, target_denoise):
        hook = None
        if schedule_for_iteration == "simple":
            hook = core.SimpleDenoiseScheduleHook(target_denoise)

        return (hook, )


class PixelKSampleHookCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "hook1": ("PK_HOOK",),
                     "hook2": ("PK_HOOK",),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, hook1, hook2):
        hook = core.PixelKSampleHookCombine(hook1, hook2)
        return (hook, )


class TiledKSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE", )
                    }}

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, seed, steps, cfg, sampler_name, scheduler, denoise,
             tile_width, tile_height, tiling_strategy, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = core.TiledKSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                            tile_width, tile_height, tiling_strategy)
        return (sampler, )


class PixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt=None, pk_hook_opt=None):
        try:
            import custom_nodes.ComfyUI_TiledKSampler.nodes
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt)
            return (upscaler, )
        except Exception as e:
            print("[ERROR] PixelTiledKSampleUpscalerProvider: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")
            print(f"\t{e}")
            pass


class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, tiling_strategy, basic_pipe, upscale_model_opt=None, pk_hook_opt=None):
        try:
            import custom_nodes.ComfyUI_TiledKSampler.nodes
            model, _, vae, positive, negative = basic_pipe
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt)
            return (upscaler, )
        except Exception as e:
            print("[ERROR] PixelTiledKSampleUpscalerProviderPipe: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")
            print(f"\t{e}")
            pass


class PixelKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": (["disabled", "enabled"],),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
             use_tiled_vae, upscale_model_opt=None, pk_hook_opt=None):
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae == "enabled", upscale_model_opt, pk_hook_opt)
        return (upscaler, )


class PixelKSampleUpscalerProviderPipe(PixelKSampleUpscalerProvider):
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": (["disabled", "enabled"],),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit_pipe"

    CATEGORY = "ImpactPack/Upscale"

    def doit_pipe(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise,
                  use_tiled_vae, basic_pipe, upscale_model_opt=None, pk_hook_opt=None):
        model, _, vae, positive, negative = basic_pipe
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae == "enabled", upscale_model_opt, pk_hook_opt)
        return (upscaler, )


class TwoSamplersForMaskUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "scale_method": (s.upscale_methods,),
                     "full_sample_schedule": (
                         ["none", "interleave1", "interleave2", "interleave3",
                          "last1", "last2",
                          "interleave1+last1", "interleave2+last1", "interleave3+last1",
                          ],),
                     "use_tiled_vae": (["disabled", "enabled"],),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", ),
                     "vae": ("VAE",),
                     },
                "optional": {
                        "full_sampler_opt": ("KSAMPLER",),
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_base_opt": ("PK_HOOK", ),
                        "pk_hook_mask_opt": ("PK_HOOK", ),
                        "pk_hook_full_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, full_sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
             full_sampler_opt=None, upscale_model_opt=None,
             pk_hook_base_opt=None, pk_hook_mask_opt=None, pk_hook_full_opt=None):
        upscaler = core.TwoSamplersForMaskUpscaler(scale_method, full_sample_schedule, use_tiled_vae == "enabled",
                                                   base_sampler, mask_sampler, mask, vae, full_sampler_opt, upscale_model_opt,
                                                   pk_hook_base_opt, pk_hook_mask_opt, pk_hook_full_opt)
        return (upscaler, )


class TwoSamplersForMaskUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "scale_method": (s.upscale_methods,),
                     "full_sample_schedule": (
                         ["none", "interleave1", "interleave2", "interleave3",
                          "last1", "last2",
                          "interleave1+last1", "interleave2+last1", "interleave3+last1",
                          ],),
                     "use_tiled_vae": (["disabled", "enabled"],),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", ),
                     "basic_pipe": ("BASIC_PIPE",),
                     },
                "optional": {
                        "full_sampler_opt": ("KSAMPLER",),
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_base_opt": ("PK_HOOK", ),
                        "pk_hook_mask_opt": ("PK_HOOK", ),
                        "pk_hook_full_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, full_sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, basic_pipe,
             full_sampler_opt=None, upscale_model_opt=None,
             pk_hook_base_opt=None, pk_hook_mask_opt=None, pk_hook_full_opt=None):
        _, _, vae, _, _ = basic_pipe
        upscaler = core.TwoSamplersForMaskUpscaler(scale_method, full_sample_schedule, use_tiled_vae == "enabled",
                                                   base_sampler, mask_sampler, mask, vae, full_sampler_opt, upscale_model_opt,
                                                   pk_hook_base_opt, pk_hook_mask_opt, pk_hook_full_opt)
        return (upscaler, )


class IterativeLatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",)
                },
                "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, samples, upscale_factor, steps, temp_prefix, upscaler, unique_id):
        w = samples['samples'].shape[3]*8  # image width
        h = samples['samples'].shape[2]*8  # image height

        if temp_prefix == "":
            temp_prefix = None

        upscale_factor_unit = max(0, (upscale_factor-1.0)/steps)
        current_latent = samples
        scale = 1

        for i in range(steps-1):
            scale += upscale_factor_unit
            new_w = w*scale
            new_h = h*scale
            core.update_node_status(unique_id, f"{i+1}/{steps} steps | x{scale:.2f}", (i+1)/steps)
            print(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
            step_info = i, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        if scale < upscale_factor:
            new_w = w*upscale_factor
            new_h = h*upscale_factor
            core.update_node_status(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            print(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        core.update_node_status(unique_id, "", None)

        return (current_latent, )


class IterativeImageUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "pixels": ("IMAGE", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "vae": ("VAE",),
                },
                "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, pixels, upscale_factor, steps, temp_prefix, upscaler, vae, unique_id):
        if temp_prefix == "":
            temp_prefix = None

        core.update_node_status(unique_id, "VAEEncode (first)", 0)
        if upscaler.is_tiled:
            latent = nodes.VAEEncodeTiled().encode(vae, pixels)[0]
        else:
            latent = nodes.VAEEncode().encode(vae, pixels)[0]

        refined_latent = IterativeLatentUpscale().doit(latent, upscale_factor, steps, temp_prefix, upscaler, unique_id)

        core.update_node_status(unique_id, "VAEDecode (final)", 1.0)
        if upscaler.is_tiled:
            pixels = nodes.VAEDecodeTiled().decode(vae, refined_latent[0])[0]
        else:
            pixels = nodes.VAEDecode().decode(vae, refined_latent[0])[0]

        core.update_node_status(unique_id, "", None)

        return (pixels, )


class FaceDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "detailer_pipe": ("DETAILER_PIPE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": (["bbox", "crop_region"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": (["enabled", "disabled"], ),
                     "force_inpaint": (["disabled", "enabled"], ),

                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     "sam_dilation": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                     "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                     "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                     },
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", )
    RETURN_NAMES = ("image", "cropped_refined", "mask", "detailer_pipe")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Simple"

    def doit(self, image, detailer_pipe, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
             sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size):

        model, vae, positive, negative, bbox_detector, sam_model_opt = detailer_pipe

        enhanced_img, cropped_enhanced, mask = FaceDetailer.enhance_face(
            image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
            sam_mask_hint_use_negative, drop_size, bbox_detector, sam_model_opt)

        return enhanced_img, cropped_enhanced, mask, detailer_pipe


class DetailerForEachTest(DetailerForEach):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("image", "cropped", "cropped_refined")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

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


class DetailerForEachTestPipe(DetailerForEachPipe):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("image", "cropped", "cropped_refined")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, segs, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, noise_mask, force_inpaint, basic_pipe):

        model, _, vae, positive, negative = basic_pipe
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
        return {"required": {},}
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self):
        shape = 0, 0
        return ((shape, []),)


class SegsToCombinedMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs):
        return (core.segs_to_combined_mask(segs), )


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
        return (core.segs_bitwise_and_mask(segs, mask), )


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

        for bseg in base_segs[1]:
            cropped_mask1 = bseg.cropped_mask.copy()
            crop_region1 = bseg.crop_region

            for mseg in mask_segs[1]:
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

        return ((base_segs[0], result),)


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

        for bseg in base_segs[1]:
            cropped_mask1 = bseg.cropped_mask.copy()
            crop_region1 = bseg.crop_region

            for mseg in mask_segs[1]:
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

        return ((base_segs[0], result),)


class MaskToSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "mask": ("MASK",),
                                "combined": (["False", "True"], ),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                                "bbox_fill": (["disabled", "enabled"], ),
                                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask, combined, crop_factor, bbox_fill, drop_size):
        result = core.mask_to_segs(mask, combined, crop_factor, bbox_fill == "enabled", drop_size)
        return (result, )


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


class PreviewBridge(nodes.PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", },
                "optional": {"image": (["#placeholder"], )},
                }

    RETURN_TYPES = ("IMAGE", "MASK", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, images, image, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        if image == "#placeholder" or image['image_hash'] != id(images) or ('0' in image and 'forward_filename' not in image[0]):
            # new input image
            res = self.save_images(images, filename_prefix, prompt, extra_pnginfo)

            item = res['ui']['images'][0]

            if not item['filename'].endswith(']'):
                filepath = f"{item['filename']} [{item['type']}]"
            else:
                filepath = item['filename']

            image, mask = nodes.LoadImage().load_image(filepath)

            res['ui']['aux'] = [id(images), res['ui']['images']]
            res['result'] = (image, mask, )

            return res

        else:
            # new mask
            if '0' in image:  # fallback
                image = image['0']

            forward = {'filename': image['forward_filename'],
                       'subfolder': image['forward_subfolder'],
                       'type': image['forward_type'], }

            res = {'ui': {'images': [forward]}}

            imgpath = ""
            if 'subfolder' in image and image['subfolder'] != "":
                imgpath = image['subfolder'] + "/"

            imgpath += f"{image['filename']}"

            if 'type' in image and image['type'] != "":
                imgpath += f" [{image['type']}]"

            res['ui']['aux'] = [id(images), [forward]]
            res['result'] = nodes.LoadImage().load_image(imgpath)

            return res


class ImageReceiver(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {
                    "image": (sorted(files), ),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}), },
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, image, link_id):
        return nodes.LoadImage().load_image(image)

    @classmethod
    def VALIDATE_INPUTS(s, image, link_id):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


from server import PromptServer

class ImageSender(nodes.PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "ImgSender"}),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, images, filename_prefix="ImgSender", link_id=0, prompt=None, extra_pnginfo=None):
        result = nodes.PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": result['ui']['images']})
        return result


class ImageMaskSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                    "images1": ("IMAGE", ),
                    },
            
                "optional": {
                    "mask1_opt": ("MASK",),
                    "images2_opt": ("IMAGE",),
                    "mask2_opt": ("MASK",),
                    "images3_opt": ("IMAGE",),
                    "mask3_opt": ("MASK",),
                    "images4_opt": ("IMAGE",),
                    "mask4_opt": ("MASK",),
                    },
                }

    RETURN_TYPES = ("IMAGE", "MASK", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, images1, mask1_opt=None, images2_opt=None, mask2_opt=None, images3_opt=None, mask3_opt=None, images4_opt=None, mask4_opt=None):
        if select == 1:
            return images1, mask1_opt,
        elif select == 2:
            return images2_opt, mask2_opt,
        elif select == 3:
            return images3_opt, mask3_opt,
        else:
            return images4_opt, mask4_opt,


class LatentSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                    "latent1": ("IMAGE",),
                    },

                "optional": {
                        "latent2_opt": ("IMAGE",),
                        "latent3_opt": ("IMAGE",),
                        "latent4_opt": ("IMAGE",),
                    },
                }

    RETURN_TYPES = ("LATENT", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, latent1, latent2_opt=None, latent3_opt=None, latent4_opt=None):
        if select == 1:
            return (latent1,)
        elif select == 2:
            return (latent2_opt,)
        elif select == 3:
            return (latent3_opt,)
        else:
            return (latent4_opt,)