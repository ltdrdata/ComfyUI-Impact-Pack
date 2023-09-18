import comfy
import math
import impact.core as core
from impact.utils import *
from nodes import MAX_RESOLUTION
import nodes

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
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
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


class KSamplerAdvancedProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                }

    RETURN_TYPES = ("KSAMPLER_ADVANCED",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, cfg, sampler_name, scheduler, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = core.KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative)
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
        new_latent_image = base_sampler.sample(latent_image)

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)

        del new_latent_image['noise_mask']

        return (new_latent_image, )


class TwoAdvancedSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "samples": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "mask_sampler": ("KSAMPLER_ADVANCED", ),
                     "mask": ("MASK", ),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000})
                     },
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def mask_erosion(samples, mask, grow_mask_by):
        mask = mask.clone()

        w = samples['samples'].shape[3]
        h = samples['samples'].shape[2]

        mask2 = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(w, h), mode="bilinear")
        if grow_mask_by == 0:
            mask_erosion = mask2
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask2.round(), kernel_tensor, padding=padding), 0, 1)

        return mask_erosion[:, :, :w, :h].round()

    def doit(self, seed, steps, denoise, samples, base_sampler, mask_sampler, mask, overlap_factor):

        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        adv_steps = int(steps / denoise)
        start_at_step = adv_steps - steps

        new_latent_image = samples.copy()

        mask_erosion = TwoAdvancedSamplersForMask.mask_erosion(samples, mask, overlap_factor)

        for i in range(start_at_step, adv_steps):
            add_noise = "enable" if i == start_at_step else "disable"
            return_with_leftover_noise = "enable" if i+1 != adv_steps else "disable"

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(add_noise, seed, adv_steps, new_latent_image, i, i + 1, "enable")

            new_latent_image['noise_mask'] = mask_erosion
            new_latent_image = mask_sampler.sample_advanced("disable", seed, adv_steps, new_latent_image, i, i + 1, return_with_leftover_noise)

        del new_latent_image['noise_mask']

        return (new_latent_image, )


class RegionalPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "mask": ("MASK", ),
                     "advanced_sampler": ("KSAMPLER_ADVANCED", ),
                     },
                }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    def doit(self, mask, advanced_sampler):
        regional_prompt = core.REGIONAL_PROMPT(mask, advanced_sampler)
        return ([regional_prompt], )


class CombineRegionalPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "regional_prompts1": ("REGIONAL_PROMPTS", ),
                     },
                }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    def doit(self, **kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )


class RegionalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "samples": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "regional_prompts": ("REGIONAL_PROMPTS", ),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000})
                     },
                 "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def mask_erosion(samples, mask, grow_mask_by):
        mask = mask.clone()

        w = samples['samples'].shape[3]
        h = samples['samples'].shape[2]

        mask2 = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(w, h), mode="bilinear")
        if grow_mask_by == 0:
            mask_erosion = mask2
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask2.round(), kernel_tensor, padding=padding), 0, 1)

        return mask_erosion[:, :, :w, :h].round()

    def doit(self, seed, steps, denoise, samples, base_sampler, regional_prompts, overlap_factor, unique_id):

        masks = [regional_prompt.mask.numpy() for regional_prompt in regional_prompts]
        masks = [np.ceil(mask).astype(np.int32) for mask in masks]
        combined_mask = torch.from_numpy(np.bitwise_or.reduce(masks))

        inv_mask = torch.where(combined_mask == 0, torch.tensor(1.0), torch.tensor(0.0))

        adv_steps = int(steps / denoise)
        start_at_step = adv_steps - steps

        region_len = len(regional_prompts)
        total = steps*region_len

        new_latent_image = samples.copy()

        for i in range(start_at_step, adv_steps):
            core.update_node_status(unique_id, f"{i}/{steps} steps  |         ", (i*region_len)/total)

            add_noise = "enable" if i == start_at_step else "disable"
            return_with_leftover_noise = "enable" if i+1 != adv_steps else "disable"

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(add_noise, seed, adv_steps, new_latent_image, i, i + 1, "enable")

            j = 1
            for regional_prompt in regional_prompts:
                core.update_node_status(unique_id, f"{i}/{steps} steps  |  {j}/{region_len}", (i*region_len + j)/total)
                new_latent_image['noise_mask'] = regional_prompt.get_mask_erosion(overlap_factor)
                new_latent_image = regional_prompt.sampler.sample_advanced("disable", seed, adv_steps, new_latent_image,
                                                                           i, i + 1, return_with_leftover_noise)
                j += 1

        core.update_node_status(unique_id, f"{steps}/{steps} steps", total)
        core.update_node_status(unique_id, "", None)

        del new_latent_image['noise_mask']

        return (new_latent_image, )

class KSamplerBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise=1.0):
        model, clip, vae, positive, negative = basic_pipe
        latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
        return (basic_pipe, latent, vae)


class KSamplerAdvancedBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     }
                }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        model, clip, vae, positive, negative = basic_pipe

        if add_noise:
            add_noise = "enabled"
        else:
            add_noise = "disabled"

        if return_with_leftover_noise:
            return_with_leftover_noise = "enabled"
        else:
            return_with_leftover_noise = "disabled"

        latent = nodes.KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0]
        return (basic_pipe, latent, vae)
