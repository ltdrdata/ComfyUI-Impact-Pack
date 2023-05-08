import os
import folder_paths
import comfy.samplers
import comfy.sd
import warnings
from segment_anything import sam_model_registry

from impact_utils import *
import impact_core as core
from impact_core import SEG, NO_BBOX_DETECTOR, NO_SEGM_DETECTOR

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
            print("[ERROR] CLIPSegToBboxDetector: CLIPSeg custom node isn't installed. You must install ComfyUI-CLIPSeg extension to use this node.")
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
        sam = sam_model_registry["vit_b"](checkpoint=modelname)
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
                    }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    OUTPUT_NODE = True

    def doit(self, onnx_detector, image, threshold, dilation, crop_factor):
        segs = onnx_detector.detect(image, threshold, dilation, crop_factor)
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
                                          positive, negative, denoise, cropped_mask, force_inpaint)

            if not (enhanced_pil is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image_pil.paste(enhanced_pil, (seg.crop_region[0], seg.crop_region[1]), mask_pil)

        image_tensor = pil2tensor(image_pil.convert('RGB'))

        if len(segs[1]) > 0:
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
                     sam_mask_hint_use_negative,
                     bbox_detector, sam_model_opt=None):
        # make default prompt as 'face' if empty prompt for CLIPSeg
        bbox_detector.setAux('face')
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor)
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
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative,
             bbox_detector, sam_model_opt=None):

        enhanced_img, cropped_enhanced, mask = FaceDetailer.enhance_face(
            image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
            sam_mask_hint_use_negative,
            bbox_detector, sam_model_opt)

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
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, samples, scale_method, scale_factor, vae, upscale_model_opt=None):
        if upscale_model_opt is None:
            latent = core.latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae)
        else:
            latent = core.latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model_opt, scale_factor, vae)
        return (latent,)


MAX_RESOLUTION=8192

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
                    "concurrent_tiles": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, concurrent_tiles, upscale_model_opt=None):
        try:
            import custom_nodes.ComfyUI_TiledKSampler.nodes
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, concurrent_tiles, upscale_model_opt)
            return (upscaler, )
        except Exception as e:
            print("[ERROR] PixelTiledKSampleUpscalerProvider: BlenderNeko/ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")
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
                    "concurrent_tiles": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                    "basic_pipe": ("BASIC_PIPE",) },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, concurrent_tiles, basic_pipe, upscale_model_opt=None):
        try:
            import custom_nodes.ComfyUI_TiledKSampler.nodes
            model, _, vae, positive, negative = basic_pipe
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, concurrent_tiles, upscale_model_opt)
            return (upscaler, )
        except Exception as e:
            print("[ERROR] PixelTiledKSampleUpscalerProvider: BlenderNeko/ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")
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
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, upscale_model_opt=None):
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, upscale_model_opt)
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
                    "basic_pipe": ("BASIC_PIPE",) },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, upscale_model_opt=None):
        model, _, vae, positive, negative = basic_pipe
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, upscale_model_opt)
        return (upscaler, )


class IterativeLatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "upscaler": ("UPSCALER",),
                }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, samples, upscale_factor, steps, upscaler):
        w = samples['samples'].shape[3]*8  # image width
        h = samples['samples'].shape[2]*8  # image height

        upscale_factor_unit = max(0, (upscale_factor-1.0)/steps)
        current_latent = samples
        scale = 1
        for i in range(steps-1):
            scale += upscale_factor_unit
            new_w = (w*scale//8)*8
            new_h = (h*scale//8)*8
            print(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w}x{new_h} (scale:{scale:.2f}) ")
            current_latent = upscaler.upscale_shape(current_latent, new_w, new_h)

        if scale < upscale_factor:
            new_w = (w*upscale_factor//8)*8
            new_h = (h*upscale_factor//8)*8
            print(f"IterativeLatentUpscale[Final]: {new_w}x{new_h} (scale:{upscale_factor:.2f}) ")
            current_latent = upscaler.upscale_shape(current_latent, new_w, new_h)

        return (current_latent, )


class IterativeImageUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "pixels": ("IMAGE", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "upscaler": ("UPSCALER",),
                     "vae": ("VAE",),
                }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, pixels, upscale_factor, steps, upscaler, vae):
        latent = nodes.VAEEncode().encode(vae, pixels)[0]
        refined_latent = IterativeLatentUpscale().doit(latent, upscale_factor, steps, upscaler)
        pixels = nodes.VAEDecode().decode(vae, refined_latent[0])[0]
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
                     },
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", )
    RETURN_NAMES = ("image", "cropped_refined", "mask", "detailer_pipe")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Simple"

    def doit(self, image, detailer_pipe, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative):

        model, vae, positive, negative, bbox_detector, sam_model_opt = detailer_pipe

        enhanced_img, cropped_enhanced, mask = FaceDetailer.enhance_face(
            image, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
            sam_mask_hint_use_negative,
            bbox_detector, sam_model_opt)

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
        return {}
    
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
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask, combined, crop_factor, bbox_fill):
        result = core.mask_to_segs(mask, combined, crop_factor, bbox_fill == "enabled")
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

    CATEGORY = "ImpactPack/Util"

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

        return (mask, )

    def save_painted_images(self, images, filename_prefix="impact-mask", 
                            prompt=None, extra_pnginfo=None, mask_image=None):
        res = self.save_images(images, filename_prefix, prompt, extra_pnginfo)

        if mask_image is not None:
            res['result'] = self.load_mask(mask_image)
        else:
            mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")
            res['result'] = (mask, )

        return res


class DetailerForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "segs": ("SEGS",),
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
            "noise_mask": (["enabled", "disabled"],),
            "force_inpaint": (["disabled", "enabled"],),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, model, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, force_inpaint):

        image_pil = tensor2pil(image).convert('RGBA')

        # shape = segs[0]
        segs = segs[1]
        for seg in segs:
            cropped_image = seg.cropped_image if seg.cropped_image is not None \
                else crop_ndarray4(image.numpy(), seg.crop_region)

            mask_pil = feather_mask(seg.cropped_mask, feather)

            if noise_mask == "enabled":
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            enhanced_pil = core.enhance_detail(cropped_image, model, vae, guide_size, guide_size_for, seg.bbox,
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

        return (enhanced_img,)

