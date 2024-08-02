import math
import impact.core as core
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from impact.utils import *
from nodes import MAX_RESOLUTION
import nodes
from impact.impact_sampling import KSamplerWrapper, KSamplerAdvancedWrapper, separated_sample, impact_sample


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

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "tile_width": "Sets the width of the tile to be used in TiledKSampler.",
            "tile_height": "Sets the height of the tile to be used in TiledKSampler.",
            "tiling_strategy": "Sets the tiling strategy for TiledKSampler.",
            "basic_pipe": "basic_pipe input for sampling",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise,
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
                                "scheduler": (core.SCHEDULERS, ),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "basic_pipe": "basic_pipe input for sampling",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=scheduler_func_opt)
        return (sampler, )


class KSamplerAdvancedProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (core.SCHEDULERS, ),
                                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                "optional": {
                                "sampler_opt": ("SAMPLER", ),
                                "scheduler_func_opt": ("SCHEDULER_FUNC",),
                            }
                }

    TOOLTIPS = {
        "input": {
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "sigma_factor": "Multiplier of noise schedule",
            "basic_pipe": "basic_pipe input for sampling",
            "sampler_opt": "[OPTIONAL] Uses the passed sampler instead of internal impact_sampler.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER_ADVANCED",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=1.0, sampler_opt=None, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative, sampler_opt=sampler_opt, sigma_factor=sigma_factor, scheduler_func=scheduler_func_opt)
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

    TOOLTIPS = {
        "input": {
            "latent_image": "input latent image",
            "base_sampler": "Sampler to apply to the region outside the mask.",
            "mask_sampler": "Sampler to apply to the masked region.",
            "mask": "region mask",
        },
        "output": ("result latent", )
    }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(latent_image, base_sampler, mask_sampler, mask):
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

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "samples": "input latent image",
            "base_sampler": "Sampler to apply to the region outside the mask.",
            "mask_sampler": "Sampler to apply to the masked region.",
            "mask": "region mask",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask by the overlap_factor amount to overlap with other regions.",
        },
        "output": ("result latent", )
    }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, denoise, samples, base_sampler, mask_sampler, mask, overlap_factor):
        regional_prompts = RegionalPrompt().doit(mask=mask, advanced_sampler=mask_sampler)[0]

        return RegionalSampler().doit(seed=seed, seed_2nd=0, seed_2nd_mode="ignore", steps=steps, base_only_steps=1,
                                      denoise=denoise, samples=samples, base_sampler=base_sampler,
                                      regional_prompts=regional_prompts, overlap_factor=overlap_factor,
                                      restore_latent=True, additional_mode="ratio between",
                                      additional_sampler="AUTO", additional_sigma_ratio=0.3)


class RegionalPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mask": ("MASK", ),
                    "advanced_sampler": ("KSAMPLER_ADVANCED", ),
                    },
                "optional": {
                    "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "variation_method": (["linear", "slerp"],),
                    }
                }

    TOOLTIPS = {
        "input": {
            "mask": "region mask",
            "advanced_sampler": "sampler for specified region",
        },
        "output": ("regional prompts. (Can be used in the RegionalSampler.)", )
    }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(mask, advanced_sampler, variation_seed=0, variation_strength=0.0, variation_method="linear"):
        regional_prompt = core.REGIONAL_PROMPT(mask, advanced_sampler, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)
        return ([regional_prompt], )


class CombineRegionalPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "regional_prompts1": ("REGIONAL_PROMPTS", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "regional_prompts1": "input regional_prompts. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Combined REGIONAL_PROMPTS", )
    }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(**kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )


class CombineConditionings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "conditioning1": "input conditionings. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Combined conditioning", )
    }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(**kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )
    

class ConcatConditionings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "conditioning1": "input conditionings. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Concatenated conditioning", )
    }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(**kwargs):
        conditioning_to = list(kwargs.values())[0]

        for k, conditioning_from in list(kwargs.items())[1:]:
            out = []
            if len(conditioning_from) > 1:
                print("Warning: ConcatConditionings {k} contains more than 1 cond, only the first one will actually be applied to conditioning1.")

            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from), 1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)

            conditioning_to = out

        return (out, )
    
    
class RegionalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "seed_2nd": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "seed_2nd_mode": (["ignore", "fixed", "seed+seed_2nd", "seed-seed_2nd", "increment", "decrement", "randomize"], ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "base_only_steps": ("INT", {"default": 2, "min": 0, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "samples": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "regional_prompts": ("REGIONAL_PROMPTS", ),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000}),
                     "restore_latent": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "additional_mode": (["DISABLE", "ratio additional", "ratio between"], {"default": "ratio between"}),
                     "additional_sampler": (["AUTO", "euler", "heun", "heunpp2", "dpm_2", "dpm_fast", "dpmpp_2m", "ddpm"],),
                     "additional_sigma_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "seed_2nd": "Additional noise seed. The behavior is determined by seed_2nd_mode.",
            "seed_2nd_mode": "application method of seed_2nd. 1) ignore: Do not use seed_2nd. In the base only sampling stage, the seed is applied as a noise seed, and in the regional sampling stage, denoising is performed as it is without additional noise. 2) Others: In the base only sampling stage, the seed is applied as a noise seed, and once it is closed so that there is no leftover noise, new noise is added with seed_2nd and the regional samping stage is performed. a) fixed: Use seed_2nd as it is as an additional noise seed. b) seed+seed_2nd: Apply the value of seed+seed_2nd as an additional noise seed. c) seed-seed_2nd: Apply the value of seed-seed_2nd as an additional noise seed. d) increment: Not implemented yet. Same with fixed. e) decrement: Not implemented yet. Same with fixed. f) randomize: Not implemented yet. Same with fixed.",
            "steps": "total sampling steps",
            "base_only_steps": "total sampling steps",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "samples": "input latent image",
            "base_sampler": "The sampler applied outside the area set by the regional_prompt.",
            "regional_prompts": "The prompt applied to each region",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask set in regional_prompts by the overlap_factor amount to overlap with other regions.",
            "restore_latent": "At each step, restore the noise outside the mask area to its original state, as per the principle of inpainting. This option is provided for backward compatibility, and it is recommended to always set it to true.",
            "additional_mode": "..._sde or uni_pc and other special samplers are used, the region is not properly denoised, and it causes a phenomenon that destroys the overall harmony. To compensate for this, a recovery operation is performed using another sampler. This requires a longer time for sampling because a second sampling is performed at each step in each region using a special sampler. 1) DISABLE: Disable this feature. 2) ratio additional: After performing the denoise amount to be performed in the step with the sampler set in the region, the recovery sampler is additionally applied by the additional_sigma_ratio. If you use this option, the total denoise amount increases by additional_sigma_ratio. 3) ratio between: The denoise amount to be performed in the step with the sampler set in the region and the denoise amount to be applied to the recovery sampler are divided by additional_sigma_ratio, and denoise is performed for each denoise amount. If you use this option, the total denoise amount does not change.",
            "additional_sampler": "1) AUTO: Automatically set the recovery sampler. If the sampler is uni_pc, uni_pc_bh2, dpmpp_sde, dpmpp_sde_gpu, the dpm_fast sampler is selected If the sampler is dpmpp_2m_sde, dpmpp_2m_sde_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu, the dpmpp_2m sampler is selected. 2) Others: Manually set the recovery sampler.",
            "additional_sigma_ratio": "Multiplier of noise schedule to be applied according to additional_mode.",
        },
        "output": ("result latent", )
    }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def separated_sample(*args, **kwargs):
        return separated_sample(*args, **kwargs)

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

    @staticmethod
    def doit(seed, seed_2nd, seed_2nd_mode, steps, base_only_steps, denoise, samples, base_sampler, regional_prompts, overlap_factor, restore_latent,
             additional_mode, additional_sampler, additional_sigma_ratio, unique_id=None):
        if restore_latent:
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
        else:
            latent_compositor = None

        masks = [regional_prompt.mask.numpy() for regional_prompt in regional_prompts]
        masks = [np.ceil(mask).astype(np.int32) for mask in masks]
        combined_mask = torch.from_numpy(np.bitwise_or.reduce(masks))

        inv_mask = torch.where(combined_mask == 0, torch.tensor(1.0), torch.tensor(0.0))

        adv_steps = int(steps / denoise)
        start_at_step = adv_steps - steps

        region_len = len(regional_prompts)
        total = steps*region_len

        leftover_noise = False
        if base_only_steps > 0:
            if seed_2nd_mode == 'ignore':
                leftover_noise = True

            noise = Noise_RandomNoise(seed).generate_noise(samples)

            for rp in regional_prompts:
                noise = rp.touch_noise(noise)

            samples = base_sampler.sample_advanced(True, seed, adv_steps, samples, start_at_step, start_at_step + base_only_steps, leftover_noise, recovery_mode="DISABLE", noise=noise)

        if seed_2nd_mode == "seed+seed_2nd":
            seed += seed_2nd
            if seed > 1125899906842624:
                seed = seed - 1125899906842624
        elif seed_2nd_mode == "seed-seed_2nd":
            seed -= seed_2nd
            if seed < 0:
                seed += 1125899906842624
        elif seed_2nd_mode != 'ignore':
            seed = seed_2nd

        new_latent_image = samples.copy()
        base_latent_image = None

        if not leftover_noise:
            add_noise = True
            noise = Noise_RandomNoise(seed).generate_noise(samples)

            for rp in regional_prompts:
                noise = rp.touch_noise(noise)
        else:
            add_noise = False
            noise = None

        for i in range(start_at_step+base_only_steps, adv_steps):
            core.update_node_status(unique_id, f"{i}/{steps} steps  |         ", ((i-start_at_step)*region_len)/total)

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(add_noise, seed, adv_steps, new_latent_image,
                                                            start_at_step=i, end_at_step=i + 1, return_with_leftover_noise=True,
                                                            recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio, noise=noise)

            if restore_latent:
                if 'noise_mask' in new_latent_image:
                    del new_latent_image['noise_mask']
                base_latent_image = new_latent_image.copy()

            j = 1
            for regional_prompt in regional_prompts:
                if restore_latent:
                    new_latent_image = base_latent_image.copy()

                core.update_node_status(unique_id, f"{i}/{steps} steps  |  {j}/{region_len}", ((i-start_at_step)*region_len + j)/total)

                region_mask = regional_prompt.get_mask_erosion(overlap_factor).squeeze(0).squeeze(0)

                new_latent_image['noise_mask'] = region_mask
                new_latent_image = regional_prompt.sampler.sample_advanced(False, seed, adv_steps, new_latent_image, i, i + 1, True,
                                                                           recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

                if restore_latent:
                    del new_latent_image['noise_mask']
                    base_latent_image = latent_compositor.composite(base_latent_image, new_latent_image, 0, 0, False, region_mask)[0]
                    new_latent_image = base_latent_image

                j += 1

            add_noise = False

        # finalize
        core.update_node_status(unique_id, f"finalize")
        if base_latent_image is not None:
            new_latent_image = base_latent_image
        else:
            base_latent_image = new_latent_image

        new_latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample_advanced(False, seed, adv_steps, new_latent_image, adv_steps, adv_steps+1, False,
                                                        recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

        core.update_node_status(unique_id, f"{steps}/{steps} steps", total)
        core.update_node_status(unique_id, "", None)

        if restore_latent:
            new_latent_image = base_latent_image

        if 'noise_mask' in new_latent_image:
            del new_latent_image['noise_mask']

        return (new_latent_image, )


class RegionalSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000}),
                     "restore_latent": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "latent_image": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "regional_prompts": ("REGIONAL_PROMPTS", ),
                     "additional_mode": (["DISABLE", "ratio additional", "ratio between"], {"default": "ratio between"}),
                     "additional_sampler": (["AUTO", "euler", "heun", "heunpp2", "dpm_2", "dpm_fast", "dpmpp_2m", "ddpm"],),
                     "additional_sigma_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                 "hidden": {"unique_id": "UNIQUE_ID"},
                }

    TOOLTIPS = {
        "input": {
            "add_noise": "Whether to add noise",
            "noise_seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "start_at_step": "The starting step of the sampling to be applied at this node within the range of 'steps'.",
            "end_at_step": "The step at which sampling applied at this node will stop within the range of steps (if greater than steps, sampling will continue only up to steps).",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask set in regional_prompts by the overlap_factor amount to overlap with other regions.",
            "restore_latent": "At each step, restore the noise outside the mask area to its original state, as per the principle of inpainting. This option is provided for backward compatibility, and it is recommended to always set it to true.",
            "return_with_leftover_noise": "Whether to return the latent with noise remaining if the noise has not been completely removed according to the noise schedule, or to completely remove the noise before returning it.",
            "latent_image": "input latent image",
            "base_sampler": "The sampler applied outside the area set by the regional_prompt.",
            "regional_prompts": "The prompt applied to each region",
            "additional_mode": "..._sde or uni_pc and other special samplers are used, the region is not properly denoised, and it causes a phenomenon that destroys the overall harmony. To compensate for this, a recovery operation is performed using another sampler. This requires a longer time for sampling because a second sampling is performed at each step in each region using a special sampler. 1) DISABLE: Disable this feature. 2) ratio additional: After performing the denoise amount to be performed in the step with the sampler set in the region, the recovery sampler is additionally applied by the additional_sigma_ratio. If you use this option, the total denoise amount increases by additional_sigma_ratio. 3) ratio between: The denoise amount to be performed in the step with the sampler set in the region and the denoise amount to be applied to the recovery sampler are divided by additional_sigma_ratio, and denoise is performed for each denoise amount. If you use this option, the total denoise amount does not change.",
            "additional_sampler": "1) AUTO: Automatically set the recovery sampler. If the sampler is uni_pc, uni_pc_bh2, dpmpp_sde, dpmpp_sde_gpu, the dpm_fast sampler is selected If the sampler is dpmpp_2m_sde, dpmpp_2m_sde_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu, the dpmpp_2m sampler is selected. 2) Others: Manually set the recovery sampler.",
            "additional_sigma_ratio": "Multiplier of noise schedule to be applied according to additional_mode.",
        },
        "output": ("result latent", )
    }


    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(add_noise, noise_seed, steps, start_at_step, end_at_step, overlap_factor, restore_latent, return_with_leftover_noise, latent_image, base_sampler, regional_prompts,
             additional_mode, additional_sampler, additional_sigma_ratio, unique_id):

        if restore_latent:
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
        else:
            latent_compositor = None

        masks = [regional_prompt.mask.numpy() for regional_prompt in regional_prompts]
        masks = [np.ceil(mask).astype(np.int32) for mask in masks]
        combined_mask = torch.from_numpy(np.bitwise_or.reduce(masks))

        inv_mask = torch.where(combined_mask == 0, torch.tensor(1.0), torch.tensor(0.0))

        region_len = len(regional_prompts)
        end_at_step = min(steps, end_at_step)
        total = (end_at_step - start_at_step) * region_len

        new_latent_image = latent_image.copy()
        base_latent_image = None
        region_masks = {}

        for i in range(start_at_step, end_at_step-1):
            core.update_node_status(unique_id, f"{start_at_step+i}/{end_at_step} steps  |         ", ((i-start_at_step)*region_len)/total)

            cur_add_noise = True if i == start_at_step and add_noise else False

            if cur_add_noise:
                noise = Noise_RandomNoise(noise_seed).generate_noise(new_latent_image)
                for rp in regional_prompts:
                    noise = rp.touch_noise(noise)
            else:
                noise = None

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(cur_add_noise, noise_seed, steps, new_latent_image, i, i + 1, True,
                                                            recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio, noise=noise)

            if restore_latent:
                del new_latent_image['noise_mask']
                base_latent_image = new_latent_image.copy()

            j = 1
            for regional_prompt in regional_prompts:
                if restore_latent:
                    new_latent_image = base_latent_image.copy()

                core.update_node_status(unique_id, f"{start_at_step+i}/{end_at_step} steps  |  {j}/{region_len}", ((i-start_at_step)*region_len + j)/total)

                if j not in region_masks:
                    region_mask = regional_prompt.get_mask_erosion(overlap_factor).squeeze(0).squeeze(0)
                    region_masks[j] = region_mask
                else:
                    region_mask = region_masks[j]

                new_latent_image['noise_mask'] = region_mask
                new_latent_image = regional_prompt.sampler.sample_advanced(False, noise_seed, steps, new_latent_image, i, i + 1, True,
                                                                           recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

                if restore_latent:
                    del new_latent_image['noise_mask']
                    base_latent_image = latent_compositor.composite(base_latent_image, new_latent_image, 0, 0, False, region_mask)[0]
                    new_latent_image = base_latent_image

                j += 1

        # finalize
        core.update_node_status(unique_id, f"finalize")
        if base_latent_image is not None:
            new_latent_image = base_latent_image
        else:
            base_latent_image = new_latent_image

        new_latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample_advanced(False, noise_seed, steps, new_latent_image, end_at_step-1, end_at_step, return_with_leftover_noise,
                                                        recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

        core.update_node_status(unique_id, f"{end_at_step}/{end_at_step} steps", total)
        core.update_node_status(unique_id, "", None)

        if restore_latent:
            new_latent_image = base_latent_image

        if 'noise_mask' in new_latent_image:
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
                     "scheduler": (core.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC", ),
                    }
                }

    TOOLTIPS = {
        "input": {
            "basic_pipe": "basic_pipe input for sampling",
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "latent_image": "input latent image",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("passthrough input basic_pipe", "result latent", "VAE in basic_pipe")
    }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "ImpactPack/sampling"

    @staticmethod
    def sample(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise=1.0, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=scheduler_func_opt)
        return basic_pipe, latent, vae


class KSamplerAdvancedBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (core.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC", ),
                    }
                }

    TOOLTIPS = {
        "input": {
            "basic_pipe": "basic_pipe input for sampling",
            "add_noise": "Whether to add noise",
            "noise_seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "latent_image": "input latent image",
            "start_at_step": "The starting step of the sampling to be applied at this node within the range of 'steps'.",
            "end_at_step": "The step at which sampling applied at this node will stop within the range of steps (if greater than steps, sampling will continue only up to steps).",
            "return_with_leftover_noise": "Whether to return the latent with noise remaining if the noise has not been completely removed according to the noise schedule, or to completely remove the noise before returning it.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("passthrough input basic_pipe", "result latent", "VAE in basic_pipe")
    }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "ImpactPack/sampling"

    @staticmethod
    def sample(basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe

        latent = separated_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, scheduler_func=scheduler_func_opt)
        return basic_pipe, latent, vae


class GITSSchedulerFuncProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "coeff": ("FLOAT", {"default": 1.20, "min": 0.80, "max": 1.50, "step": 0.05}),
                        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    TOOLTIPS = {
        "input": {
            "coeff": "coeff factor of GITS Scheduler",
            "denoise": "denoise amount for noise schedule",
        },
        "output": ("Returns a function that generates a noise schedule using GITSScheduler. This can be used in place of a predetermined noise schedule to dynamically generate a noise schedule based on the steps.",)
    }

    RETURN_TYPES = ("SCHEDULER_FUNC",)
    CATEGORY = "ImpactPack/sampling"

    FUNCTION = "doit"

    @staticmethod
    def doit(coeff, denoise):
        def f(model, sampler, steps):
            if 'GITSScheduler' not in nodes.NODE_CLASS_MAPPINGS:
                raise Exception("[Impact Pack] ComfyUI is an outdated version. Cannot use GITSScheduler.")

            scheduler = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']()
            return scheduler.get_sigmas(coeff, steps, denoise)[0]

        return (f, )


class NegativeConditioningPlaceholder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    TOOLTIPS = {
        "output": ("This is a Placeholder for the FLUX model that does not use Negative Conditioning.",)
    }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "ImpactPack/sampling"

    FUNCTION = "doit"

    @staticmethod
    def doit():
        return ("NegativePlaceholder", )
