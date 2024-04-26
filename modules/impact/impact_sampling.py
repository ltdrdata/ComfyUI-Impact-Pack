import nodes
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
from comfy_extras import nodes_custom_sampler

import torch
import math


def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if hasattr(samplers, 'calculate_sigmas'):
        if scheduler.startswith('AYS'):
            sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps)[0]
        else:
            sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)
    else:
        print(f"[Impact Pack] calculate_sigmas: ComfyUI is an outdated version.")
        sigmas = samplers.calculate_sigmas_scheduler(model.model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde
    else:
        return samplers.ksampler(sampler_name, extra_options, inpaint_options)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None):
    if sampler_opt is None:
        total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
    else:
        total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas[start_at_step:end_at_step+1] * sigma_ratio
    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image
    
    res = nodes_custom_sampler.SamplerCustom().sample(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor=1.0):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        # Use separated_sample instead of KSampler for `AYS scheduler`
        # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]

        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + steps

        refined_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, False, sigma_ratio=sigma_factor)
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True, sigma_ratio=sigma_factor)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False, sigma_ratio=sigma_factor)

    return refined_latent


class KSamplerAdvancedWrapper:
    params = None

    def __init__(self, model, cfg, sampler_name, scheduler, positive, negative, sampler_opt=None, sigma_factor=1.0):
        self.params = model, cfg, sampler_name, scheduler, positive, negative, sigma_factor
        self.sampler_opt = sampler_opt

    def clone_with_conditionings(self, positive, negative):
        model, cfg, sampler_name, scheduler, _, _, _ = self.params
        return KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative, self.sampler_opt)

    def sample_advanced(self, add_noise, seed, steps, latent_image, start_at_step, end_at_step, return_with_leftover_noise, hook=None,
                        recovery_mode="ratio additional", recovery_sampler="AUTO", recovery_sigma_ratio=1.0):

        model, cfg, sampler_name, scheduler, positive, negative, sigma_factor = self.params
        # steps, start_at_step, end_at_step = self.compensate_denoise(steps, start_at_step, end_at_step)

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent = hook.pre_ksample_advanced(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                                                                                              positive, negative, latent_image, start_at_step, end_at_step,
                                                                                                                              return_with_leftover_noise)

        if recovery_mode != 'DISABLE' and sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']:
            base_image = latent_image.copy()
            if recovery_mode == "ratio between":
                sigma_ratio = 1.0 - recovery_sigma_ratio
            else:
                sigma_ratio = 1.0
        else:
            base_image = None
            sigma_ratio = 1.0

        try:
            if sigma_ratio > 0:
                latent_image = separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                positive, negative, latent_image, start_at_step, end_at_step,
                                                return_with_leftover_noise, sigma_ratio=sigma_ratio * sigma_factor, sampler_opt=self.sampler_opt)
        except ValueError as e:
            if str(e) == 'sigma_min and sigma_max must not be 0':
                print(f"\nWARN: sampling skipped - sigma_min and sigma_max are 0")
                return latent_image

        if (recovery_sigma_ratio > 0 and recovery_mode != 'DISABLE' and
                sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']):
            compensate = 0 if sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu'] else 2
            if recovery_sampler == "AUTO":
                recovery_sampler = 'dpm_fast' if sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu'] else 'dpmpp_2m'

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()

            noise_mask = latent_image['noise_mask']

            if len(noise_mask.shape) == 4:
                noise_mask = noise_mask.squeeze(0).squeeze(0)

            latent_image = latent_compositor.composite(base_image, latent_image, 0, 0, False, noise_mask)[0]

            try:
                latent_image = separated_sample(model, add_noise, seed, steps, cfg, recovery_sampler, scheduler,
                                                positive, negative, latent_image, start_at_step-compensate, end_at_step,
                                                return_with_leftover_noise, sigma_ratio=recovery_sigma_ratio * sigma_factor, sampler_opt=self.sampler_opt)
            except ValueError as e:
                if str(e) == 'sigma_min and sigma_max must not be 0':
                    print(f"\nWARN: sampling skipped - sigma_min and sigma_max are 0")

        return latent_image


class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
