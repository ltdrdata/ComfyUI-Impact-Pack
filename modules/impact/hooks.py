import copy
import nodes

from impact import utils
from . import segs_nodes
from thirdparty import noise_nodes
from server import PromptServer
import asyncio
import folder_paths
import os

class PixelKSampleHook:
    cur_step = 0
    total_step = 0

    def __init__(self):
        pass

    def set_steps(self, info):
        self.cur_step, self.total_step = info

    def post_decode(self, pixels):
        return pixels

    def post_upscale(self, pixels):
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise

    def post_crop_region(self, w, h, item_bbox, crop_region):
        return crop_region

    def touch_scaled_size(self, w, h):
        return w, h


class PixelKSampleHookCombine(PixelKSampleHook):
    hook1 = None
    hook2 = None

    def __init__(self, hook1, hook2):
        super().__init__()
        self.hook1 = hook1
        self.hook2 = hook2

    def set_steps(self, info):
        self.hook1.set_steps(info)
        self.hook2.set_steps(info)

    def pre_decode(self, samples):
        return self.hook2.pre_decode(self.hook1.pre_decode(samples))

    def post_decode(self, pixels):
        return self.hook2.post_decode(self.hook1.post_decode(pixels))

    def post_upscale(self, pixels):
        return self.hook2.post_upscale(self.hook1.post_upscale(pixels))

    def post_encode(self, samples):
        return self.hook2.post_encode(self.hook1.post_encode(samples))

    def post_crop_region(self, w, h, item_bbox, crop_region):
        crop_region = self.hook1.post_crop_region(w, h, item_bbox, crop_region)
        return self.hook2.post_crop_region(w, h, item_bbox, crop_region)

    def touch_scaled_size(self, w, h):
        w, h = self.hook1.touch_scaled_size(w, h)
        return self.hook2.touch_scaled_size(w, h)

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
            self.hook1.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                   upscaled_latent, denoise)

        return self.hook2.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)


class DetailerHookCombine(PixelKSampleHookCombine):
    def cycle_latent(self, latent):
        latent = self.hook1.cycle_latent(latent)
        latent = self.hook2.cycle_latent(latent)
        return latent

    def post_detection(self, segs):
        segs = self.hook1.post_detection(segs)
        segs = self.hook2.post_detection(segs)
        return segs

    def post_paste(self, image):
        image = self.hook1.post_paste(image)
        image = self.hook2.post_paste(image)
        return image


class SimpleCfgScheduleHook(PixelKSampleHook):
    target_cfg = 0

    def __init__(self, target_cfg):
        super().__init__()
        self.target_cfg = target_cfg

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_cfg - cfg
            current_cfg = int(cfg + gap * progress)
        else:
            current_cfg = self.target_cfg

        return model, seed, steps, current_cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class SimpleDenoiseScheduleHook(PixelKSampleHook):
    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_denoise - denoise
            current_denoise = denoise + gap * progress
        else:
            current_denoise = self.target_denoise

        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, current_denoise


class SimpleStepsScheduleHook(PixelKSampleHook):
    def __init__(self, target_steps):
        super().__init__()
        self.target_steps = target_steps

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_steps - steps
            current_steps = int(steps + gap * progress)
        else:
            current_steps = self.target_steps

        return model, seed, current_steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class DetailerHook(PixelKSampleHook):
    def cycle_latent(self, latent):
        return latent

    def post_detection(self, segs):
        return segs

    def post_paste(self, image):
        return image


class SimpleDetailerDenoiseSchedulerHook(DetailerHook):
    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_denoise - denoise
            current_denoise = denoise + gap * progress
        else:
            # ignore hook if total cycle <= 1
            current_denoise = denoise

        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, current_denoise


class CoreMLHook(DetailerHook):
    def __init__(self, mode):
        super().__init__()
        resolution = mode.split('x')

        self.w = int(resolution[0])
        self.h = int(resolution[1])

        self.override_bbox_by_segm = False

    def pre_decode(self, samples):
        new_samples = copy.deepcopy(samples)
        new_samples['samples'] = samples['samples'][0].unsqueeze(0)
        return new_samples

    def post_encode(self, samples):
        new_samples = copy.deepcopy(samples)
        new_samples['samples'] = samples['samples'].repeat(2, 1, 1, 1)
        return new_samples

    def post_crop_region(self, w, h, item_bbox, crop_region):
        x1, y1, x2, y2 = crop_region
        bx1, by1, bx2, by2 = item_bbox
        crop_w = x2-x1
        crop_h = y2-y1

        crop_ratio = crop_w/crop_h
        target_ratio = self.w/self.h
        if crop_ratio < target_ratio:
            # shrink height
            top_gap = by1 - y1
            bottom_gap = y2 - by2

            gap_ratio = top_gap / bottom_gap

            target_height = 1/target_ratio*crop_w
            delta_height = crop_h - target_height

            new_y1 = int(y1 + delta_height*gap_ratio)
            new_y2 = int(new_y1 + target_height)
            crop_region = x1, new_y1, x2, new_y2

        elif crop_ratio > target_ratio:
            # shrink width
            left_gap = bx1 - x1
            right_gap = x2 - bx2

            gap_ratio = left_gap / right_gap

            target_width = target_ratio*crop_h
            delta_width = crop_w - target_width

            new_x1 = int(x1 + delta_width*gap_ratio)
            new_x2 = int(new_x1 + target_width)
            crop_region = new_x1, y1, new_x2, y2

        return crop_region

    def touch_scaled_size(self, w, h):
        return self.w, self.h


# REQUIREMENTS: BlenderNeko/ComfyUI Noise
class InjectNoiseHook(PixelKSampleHook):
    def __init__(self, source, seed, start_strength, end_strength):
        super().__init__()
        self.source = source
        self.seed = seed
        self.start_strength = start_strength
        self.end_strength = end_strength

    def post_encode(self, samples):
        cur_step = self.cur_step

        size = samples['samples'].shape
        seed = cur_step + self.seed + cur_step

        if "BNK_NoisyLatentImage" in nodes.NODE_CLASS_MAPPINGS and "BNK_InjectNoise" in nodes.NODE_CLASS_MAPPINGS:
            NoisyLatentImage = nodes.NODE_CLASS_MAPPINGS["BNK_NoisyLatentImage"]
            InjectNoise = nodes.NODE_CLASS_MAPPINGS["BNK_InjectNoise"]
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_Noise',
                                          "To use 'NoiseInjectionHookProvider', 'ComfyUI Noise' extension is required.")
            raise Exception("'BNK_NoisyLatentImage', 'BNK_InjectNoise' nodes are not installed.")

        noise = NoisyLatentImage().create_noisy_latents(self.source, seed, size[3] * 8, size[2] * 8, size[0])[0]

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        strength = self.start_strength + (self.end_strength - self.start_strength) * cur_step / self.total_step
        samples = InjectNoise().inject_noise(samples, strength, noise, mask)[0]
        print(f"[Impact Pack] InjectNoiseHook: strength = {strength}")

        if mask is not None:
            samples['noise_mask'] = mask

        return samples


class UnsamplerHook(PixelKSampleHook):
    def __init__(self, model, steps, start_end_at_step, end_end_at_step, cfg, sampler_name,
                 scheduler, normalize, positive, negative):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.steps = steps
        self.start_end_at_step = start_end_at_step
        self.end_end_at_step = end_end_at_step
        self.scheduler = scheduler
        self.normalize = normalize
        self.positive = positive
        self.negative = negative

    def post_encode(self, samples):
        cur_step = self.cur_step

        try:
            Unsampler = noise_nodes.Unsampler
        except:
            if 'BNK_Unsampler' not in nodes.NODE_CLASS_MAPPINGS:
                print("[Impact Pack] ERROR: ComfyUI version is outdated and the BNK_Unsampler node is not installed, so this feature cannot be used.")
                raise Exception("ERROR: ComfyUI version is outdated and the BNK_Unsampler node is not installed, so this feature cannot be used.")

            Unsampler = nodes.NODE_CLASS_MAPPINGS['BNK_Unsampler']

        end_at_step = self.start_end_at_step + (self.end_end_at_step - self.start_end_at_step) * cur_step / self.total_step
        end_at_step = int(end_at_step)

        print(f"[Impact Pack] UnsamplerHook: end_at_step = {end_at_step}")

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        samples = Unsampler().unsampler(self.model, self.cfg, self.sampler_name, self.steps, end_at_step,
                                        self.scheduler, self.normalize, self.positive, self.negative, samples)[0]

        if mask is not None:
            samples['noise_mask'] = mask

        return samples


class InjectNoiseHookForDetailer(DetailerHook):
    def __init__(self, source, seed, start_strength, end_strength, from_start=False):
        super().__init__()
        self.source = source
        self.seed = seed
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.from_start = from_start

    def inject_noise(self, samples):
        cur_step = self.cur_step if self.from_start else self.cur_step - 1
        total_step = self.total_step if self.from_start else self.total_step - 1

        size = samples['samples'].shape
        seed = cur_step + self.seed + cur_step

        if "BNK_NoisyLatentImage" in nodes.NODE_CLASS_MAPPINGS and "BNK_InjectNoise" in nodes.NODE_CLASS_MAPPINGS:
            NoisyLatentImage = nodes.NODE_CLASS_MAPPINGS["BNK_NoisyLatentImage"]
            InjectNoise = nodes.NODE_CLASS_MAPPINGS["BNK_InjectNoise"]
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_Noise',
                                          "To use 'NoiseInjectionDetailerHookProvider', 'ComfyUI Noise' extension is required.")
            raise Exception("'BNK_NoisyLatentImage', 'BNK_InjectNoise' nodes are not installed.")

        noise = NoisyLatentImage().create_noisy_latents(self.source, seed, size[3] * 8, size[2] * 8, size[0])[0]

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        strength = self.start_strength + (self.end_strength - self.start_strength) * cur_step / total_step
        samples = InjectNoise().inject_noise(samples, strength, noise, mask)[0]

        if mask is not None:
            samples['noise_mask'] = mask

        return samples

    def cycle_latent(self, latent):
        if self.cur_step == 0 and not self.from_start:
            return latent
        else:
            return self.inject_noise(latent)


class UnsamplerDetailerHook(DetailerHook):
    def __init__(self, model, steps, start_end_at_step, end_end_at_step, cfg, sampler_name,
                 scheduler, normalize, positive, negative, from_start=False):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.steps = steps
        self.start_end_at_step = start_end_at_step
        self.end_end_at_step = end_end_at_step
        self.scheduler = scheduler
        self.normalize = normalize
        self.positive = positive
        self.negative = negative
        self.from_start = from_start

    def unsample(self, samples):
        cur_step = self.cur_step if self.from_start else self.cur_step - 1
        total_step = self.total_step if self.from_start else self.total_step - 1

        Unsampler = noise_nodes.Unsampler

        end_at_step = self.start_end_at_step + (self.end_end_at_step - self.start_end_at_step) * cur_step / total_step
        end_at_step = int(end_at_step)

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        samples = Unsampler().unsampler(self.model, self.cfg, self.sampler_name, self.steps, end_at_step,
                                        self.scheduler, self.normalize, self.positive, self.negative, samples)[0]

        if mask is not None:
            samples['noise_mask'] = mask

        return samples

    def cycle_latent(self, latent):
        if self.cur_step == 0 and not self.from_start:
            return latent
        else:
            return self.unsample(latent)


class SEGSOrderedFilterDetailerHook(DetailerHook):
    def __init__(self, target, order, take_start, take_count):
        super().__init__()
        self.target = target
        self.order = order
        self.take_start = take_start
        self.take_count = take_count

    def post_detection(self, segs):
        return segs_nodes.SEGSOrderedFilter().doit(segs, self.target, self.order, self.take_start, self.take_count)[0]


class SEGSRangeFilterDetailerHook(DetailerHook):
    def __init__(self, target, mode, min_value, max_value):
        super().__init__()
        self.target = target
        self.mode = mode
        self.min_value = min_value
        self.max_value = max_value

    def post_detection(self, segs):
        return segs_nodes.SEGSRangeFilter().doit(segs, self.target, self.mode, self.min_value, self.max_value)[0]


class SEGSLabelFilterDetailerHook(DetailerHook):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def post_detection(self, segs):
        return segs_nodes.SEGSLabelFilter().doit(segs, "", self.labels)[0]


class PreviewDetailerHook(DetailerHook):
    def __init__(self, node_id, quality):
        super().__init__()
        self.node_id = node_id
        self.quality = quality

    async def send(self, image):
        if len(image) > 0:
            image = image[0].unsqueeze(0)
        img = utils.tensor2pil(image)

        temp_path = os.path.join(folder_paths.get_temp_directory(), 'pvhook')

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        fullpath = os.path.join(temp_path, f"{self.node_id}.webp")
        img.save(fullpath, quality=self.quality)

        item = {
                "filename": f"{self.node_id}.webp",
                "subfolder": 'pvhook',
                "type": 'temp'
                }

        PromptServer.instance.send_sync("impact-preview", {'node_id': self.node_id, 'item': item})

    def post_paste(self, image):
        asyncio.run(self.send(image))
        return image
