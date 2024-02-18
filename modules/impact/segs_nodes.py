import os
import sys

import impact.impact_server
from nodes import MAX_RESOLUTION

from impact.utils import *
import impact.core as core
from impact.core import SEG
import impact.utils as utils
from . import defs


class SEGSDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "basic_pipe": ("BASIC_PIPE",),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                     }
                }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "cnet_images")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
                  refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image.shape)

        new_segs = []
        cnet_pil_list = []

        for i in range(batch_size):
            seed += 1
            for seg in segs[1]:
                cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                                  else crop_ndarray4(image.numpy(), seg.crop_region)
                cropped_image = to_tensor(cropped_image)

                is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
                if is_mask_all_zeros:
                    print(f"Detailer: segment skip [empty mask]")
                    new_segs.append(seg)
                    continue

                if noise_mask:
                    cropped_mask = seg.cropped_mask
                else:
                    cropped_mask = None

                enhanced_image, cnet_pils = core.enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                positive, negative, denoise, cropped_mask, force_inpaint,
                                                                refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                                                control_net_wrapper=seg.control_net_wrapper, cycle=cycle,
                                                                inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

                if cnet_pils is not None:
                    cnet_pil_list.extend(cnet_pils)

                if enhanced_image is None:
                    new_cropped_image = cropped_image
                else:
                    new_cropped_image = enhanced_image

                new_seg = SEG(to_numpy(new_cropped_image), seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
                new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_pil_list

    def doit(self, image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
             refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SEGSDetailer does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        segs, cnet_pil_list = SEGSDetailer.do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                     scheduler, denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio, batch_size, cycle=cycle,
                                                     refiner_basic_pipe_opt=refiner_basic_pipe_opt,
                                                     inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

        # set fallback image
        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return (segs, cnet_pil_list)


class SEGSPaste:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                     },
                "optional": {"ref_image_opt": ("IMAGE", ), }
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def doit(image, segs, feather, alpha=255, ref_image_opt=None):

        segs = core.segs_scale_match(segs, image.shape)

        result = None
        for i, single_image in enumerate(image):
            image_i = single_image.unsqueeze(0).clone()

            for seg in segs[1]:
                ref_image = None
                if ref_image_opt is None and seg.cropped_image is not None:
                    cropped_image = seg.cropped_image
                    if isinstance(cropped_image, np.ndarray):
                        cropped_image = torch.from_numpy(cropped_image)
                    ref_image = cropped_image[i].unsqueeze(0)
                elif ref_image_opt is not None:
                    ref_tensor = ref_image_opt[i].unsqueeze(0)
                    ref_image = crop_image(ref_tensor, seg.crop_region)
                if ref_image is not None:
                    if seg.cropped_mask.ndim == 3 and len(seg.cropped_mask) == len(image):
                        mask = seg.cropped_mask[i]
                    elif seg.cropped_mask.ndim == 3 and len(seg.cropped_mask) > 1:
                        print(f"[Impact Pack] WARN: SEGSPaste - The number of the mask batch({len(seg.cropped_mask)}) and the image batch({len(image)}) are different. Combine the mask frames and apply.")
                        combined_mask = (seg.cropped_mask[0] * 255).to(torch.uint8)

                        for frame_mask in seg.cropped_mask[1:]:
                            combined_mask |= (frame_mask * 255).to(torch.uint8)

                        combined_mask = (combined_mask/255.0).to(torch.float32)
                        mask = utils.to_binary_mask(combined_mask, 0.1)
                    else:  # ndim == 2
                        mask = seg.cropped_mask

                    mask = tensor_gaussian_blur_mask(mask, feather) * (alpha/255)
                    x, y, *_ = seg.crop_region
                    tensor_paste(image_i, ref_image, (x, y), mask)

            if result is None:
                result = image_i
            else:
                result = torch.concat((result, image_i), dim=0)

        return (result, )


class SEGSPreviewCNet:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS", ),}, }

    RETURN_TYPES = ("IMAGE", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    OUTPUT_NODE = True

    def doit(self, segs):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path("impact_seg_preview", self.output_dir, segs[0][1], segs[0][0])

        results = list()
        result_image_list = []

        for seg in segs[1]:
            file = f"{filename}_{counter:05}_.webp"

            if seg.control_net_wrapper is not None and seg.control_net_wrapper.control_image is not None:
                cnet_image = seg.control_net_wrapper.control_image
                result_image_list.append(cnet_image)
            else:
                cnet_image = empty_pil_tensor(64, 64)

            cnet_pil = utils.tensor2pil(cnet_image)
            cnet_pil.save(os.path.join(full_output_folder, file))

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

            counter += 1

        return {"ui": {"images": results}, "result": (result_image_list,)}


class SEGSPreview:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     "alpha_mode": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "min_alpha": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional": {
                     "fallback_image_opt": ("IMAGE", ),
                    }
                }

    RETURN_TYPES = ("IMAGE", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    OUTPUT_NODE = True

    def doit(self, segs, alpha_mode=True, min_alpha=0.0, fallback_image_opt=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path("impact_seg_preview", self.output_dir, segs[0][1], segs[0][0])

        results = list()
        result_image_list = []

        if fallback_image_opt is not None:
            segs = core.segs_scale_match(segs, fallback_image_opt.shape)

        if min_alpha != 0:
            min_alpha = int(255 * min_alpha)

        if len(segs[1]) > 0:
            if segs[1][0].cropped_image is not None:
                batch_count = len(segs[1][0].cropped_image)
            elif fallback_image_opt is not None:
                batch_count = len(fallback_image_opt)
            else:
                return {"ui": {"images": results}}

            for seg in segs[1]:
                result_image_batch = None
                cached_mask = None

                def get_combined_mask():
                    nonlocal cached_mask

                    if cached_mask is not None:
                        return cached_mask
                    else:
                        if isinstance(seg.cropped_mask, np.ndarray):
                            masks = torch.tensor(seg.cropped_mask)
                        else:
                            masks = seg.cropped_mask

                        cached_mask = (masks[0] * 255).to(torch.uint8)
                        for x in masks[1:]:
                            cached_mask |= (x * 255).to(torch.uint8)
                        cached_mask = (cached_mask/255.0).to(torch.float32)
                        cached_mask = utils.to_binary_mask(cached_mask, 0.1)
                        cached_mask = cached_mask.numpy()

                        return cached_mask

                def stack_image(image, mask=None):
                    nonlocal result_image_batch

                    if isinstance(image, np.ndarray):
                        image = torch.from_numpy(image)

                    if mask is not None:
                        image *= torch.tensor(mask)[None, ..., None]

                    if result_image_batch is None:
                        result_image_batch = image
                    else:
                        result_image_batch = torch.concat((result_image_batch, image), dim=0)

                for i in range(batch_count):
                    cropped_image = None

                    if seg.cropped_image is not None:
                        cropped_image = seg.cropped_image[i, None]
                    elif fallback_image_opt is not None:
                        # take from original image
                        ref_image = fallback_image_opt[i].unsqueeze(0)
                        cropped_image = crop_image(ref_image, seg.crop_region)

                    if cropped_image is not None:
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)

                        cropped_image = cropped_image.clone()
                        cropped_pil = to_pil(cropped_image)

                        if alpha_mode:
                            if isinstance(seg.cropped_mask, np.ndarray):
                                cropped_mask = seg.cropped_mask
                            else:
                                if seg.cropped_image is not None and len(seg.cropped_image) != len(seg.cropped_mask):
                                    cropped_mask = get_combined_mask()
                                else:
                                    cropped_mask = seg.cropped_mask[i].numpy()

                            mask_array = (cropped_mask * 255).astype(np.uint8)

                            if min_alpha != 0:
                                mask_array[mask_array < min_alpha] = min_alpha

                            mask_pil = Image.fromarray(mask_array, mode='L').resize(cropped_pil.size)
                            cropped_pil.putalpha(mask_pil)
                            stack_image(cropped_image, cropped_mask)
                        else:
                            stack_image(cropped_image)

                        file = f"{filename}_{counter:05}_.webp"
                        cropped_pil.save(os.path.join(full_output_folder, file))
                        results.append({
                            "filename": file,
                            "subfolder": subfolder,
                            "type": self.type
                        })

                        counter += 1

                if result_image_batch is not None:
                    result_image_list.append(result_image_batch)

        return {"ui": {"images": results}, "result": (result_image_list,) }


class SEGSLabelFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "preset": (['all'] + defs.detection_labels, ),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the types of segments to be allowed, separated by commas"}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def filter(segs, labels):
        labels = set([label.strip() for label in labels])

        if 'all' in labels:
            return (segs, (segs[0], []), )
        else:
            res_segs = []
            remained_segs = []

            for x in segs[1]:
                if x.label in labels:
                    res_segs.append(x)
                elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                    res_segs.append(x)
                elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                    res_segs.append(x)
                elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                    res_segs.append(x)
                else:
                    remained_segs.append(x)

        return ((segs[0], res_segs), (segs[0], remained_segs), )

    def doit(self, segs, preset, labels):
        labels = labels.split(',')
        return SEGSLabelFilter.filter(segs, labels)


class SEGSLabelAssign:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the label to be assigned in order of segs, separated by commas"}),
                     },
                }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def assign(segs, labels):
        labels = [label.strip() for label in labels]

        if len(labels) != len(segs[1]):
            print(f'Warning (SEGSLabelAssign): length of labels ({len(labels)}) != length of segs ({len(segs[1])})')

        labeled_segs = []

        idx = 0
        for x in segs[1]:
            if len(labels) > idx:
                x = x._replace(label=labels[idx])
            labeled_segs.append(x)
            idx += 1

        return ((segs[0], labeled_segs), )

    def doit(self, segs, labels):
        labels = labels.split(',')
        return SEGSLabelAssign.assign(segs, labels)


class SEGSOrderedFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2"],),
                        "order": ("BOOLEAN", {"default": True, "label_on": "descending", "label_off": "ascending"}),
                        "take_start": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "take_count": ("INT", {"default": 1, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, target, order, take_start, take_count):
        segs_with_order = []

        for seg in segs[1]:
            x1 = seg.crop_region[0]
            y1 = seg.crop_region[1]
            x2 = seg.crop_region[2]
            y2 = seg.crop_region[3]

            if target == "area(=w*h)":
                value = (y2 - y1) * (x2 - x1)
            elif target == "width":
                value = x2 - x1
            elif target == "height":
                value = y2 - y1
            elif target == "x1":
                value = x1
            elif target == "x2":
                value = x2
            elif target == "y1":
                value = y1
            else:
                value = y2

            segs_with_order.append((value, seg))

        if order:
            sorted_list = sorted(segs_with_order, key=lambda x: x[0], reverse=True)
        else:
            sorted_list = sorted(segs_with_order, key=lambda x: x[0], reverse=False)

        result_list = []
        remained_list = []

        for i, item in enumerate(sorted_list):
            if take_start <= i < take_start + take_count:
                result_list.append(item[1])
            else:
                remained_list.append(item[1])

        return ((segs[0], result_list), (segs[0], remained_list), )


class SEGSRangeFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent"],),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "inside", "label_off": "outside"}),
                        "min_value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "max_value": ("INT", {"default": 67108864, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, target, mode, min_value, max_value):
        new_segs = []
        remained_segs = []

        for seg in segs[1]:
            x1 = seg.crop_region[0]
            y1 = seg.crop_region[1]
            x2 = seg.crop_region[2]
            y2 = seg.crop_region[3]

            if target == "area(=w*h)":
                value = (y2 - y1) * (x2 - x1)
            elif target == "length_percent":
                h = y2 - y1
                w = x2 - x1
                value = max(h/w, w/h)*100
                print(f"value={value}")
            elif target == "width":
                value = x2 - x1
            elif target == "height":
                value = y2 - y1
            elif target == "x1":
                value = x1
            elif target == "x2":
                value = x2
            elif target == "y1":
                value = y1
            else:
                value = y2

            if mode and min_value <= value <= max_value:
                print(f"[in] value={value} / {mode}, {min_value}, {max_value}")
                new_segs.append(seg)
            elif not mode and (value < min_value or value > max_value):
                print(f"[out] value={value} / {mode}, {min_value}, {max_value}")
                new_segs.append(seg)
            else:
                remained_segs.append(seg)
                print(f"[filter] value={value} / {mode}, {min_value}, {max_value}")

        return ((segs[0], new_segs), (segs[0], remained_segs), )


class SEGSToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     },
                "optional": {
                     "fallback_image_opt": ("IMAGE", ),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, fallback_image_opt=None):
        results = list()

        if fallback_image_opt is not None:
            segs = core.segs_scale_match(segs, fallback_image_opt.shape)

        for seg in segs[1]:
            if seg.cropped_image is not None:
                cropped_image = to_tensor(seg.cropped_image)
            elif fallback_image_opt is not None:
                # take from original image
                cropped_image = to_tensor(crop_image(fallback_image_opt, seg.crop_region))
            else:
                cropped_image = empty_pil_tensor()

            results.append(cropped_image)

        if len(results) == 0:
            results.append(empty_pil_tensor())

        return (results,)


class SEGSToMaskList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     },
                }

    RETURN_TYPES = ("MASK",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        masks = core.segs_to_masklist(segs)
        if len(masks) == 0:
            empty_mask = torch.zeros(segs[0], dtype=torch.float32, device="cpu")
            masks = [empty_mask]
        masks = [utils.make_3d_mask(mask) for mask in masks]
        return (masks,)


class SEGSToMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     },
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        masks = core.segs_to_masklist(segs)
        masks = [utils.make_3d_mask(mask) for mask in masks]
        mask_batch = torch.concat(masks)
        return (mask_batch,)


class SEGSConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs1": ("SEGS", ),
                     },
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        dim = None
        res = None

        for k, v in list(kwargs.items()):
            if v[0] == (0, 0) or len(v[1]) == 0:
                continue

            if dim is None:
                dim = v[0]
                res = v[1]
            else:
                if v[0] == dim:
                    res = res + v[1]
                else:
                    print(f"ERROR: source shape of 'segs1'{dim} and '{k}'{v[0]} are different. '{k}' will be ignored")

        if dim is None:
            empty_segs = ((0, 0), [])
            return (empty_segs, )
        else:
            return ((dim, res), )


class DecomposeSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     },
                }

    RETURN_TYPES = ("SEGS_HEADER", "SEG_ELT",)
    OUTPUT_IS_LIST = (False, True, )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        return segs


class AssembleSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seg_header": ("SEGS_HEADER", ),
                     "seg_elt": ("SEG_ELT", ),
                     },
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("SEGS", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, seg_header, seg_elt):
        return ((seg_header[0], seg_elt), )


class From_SEG_ELT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seg_elt": ("SEG_ELT", ),
                     },
                }

    RETURN_TYPES = ("SEG_ELT", "IMAGE", "MASK", "SEG_ELT_crop_region", "SEG_ELT_bbox", "SEG_ELT_control_net_wrapper", "FLOAT", "STRING")
    RETURN_NAMES = ("seg_elt", "cropped_image", "cropped_mask", "crop_region", "bbox", "control_net_wrapper", "confidence", "label")

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, seg_elt):
        cropped_image = to_tensor(seg_elt.cropped_image) if seg_elt.cropped_image is not None else None
        return (seg_elt, cropped_image, to_tensor(seg_elt.cropped_mask), seg_elt.crop_region, seg_elt.bbox, seg_elt.control_net_wrapper, seg_elt.confidence, seg_elt.label,)


class Edit_SEG_ELT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seg_elt": ("SEG_ELT", ),
                     },
                "optional": {
                     "cropped_image_opt": ("IMAGE", ),
                     "cropped_mask_opt": ("MASK", ),
                     "crop_region_opt": ("SEG_ELT_crop_region", ),
                     "bbox_opt": ("SEG_ELT_bbox", ),
                     "control_net_wrapper_opt": ("SEG_ELT_control_net_wrapper", ),
                     "confidence_opt": ("FLOAT", {"min": 0, "max": 1.0, "step": 0.1, "forceInput": True}),
                     "label_opt": ("STRING", {"multiline": False, "forceInput": True}),
                    }
                }

    RETURN_TYPES = ("SEG_ELT", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, seg_elt, cropped_image_opt=None, cropped_mask_opt=None, confidence_opt=None, crop_region_opt=None,
             bbox_opt=None, label_opt=None, control_net_wrapper_opt=None):

        cropped_image = seg_elt.cropped_image if cropped_image_opt is None else cropped_image_opt
        cropped_mask = seg_elt.cropped_mask if cropped_mask_opt is None else cropped_mask_opt
        confidence = seg_elt.confidence if confidence_opt is None else confidence_opt
        crop_region = seg_elt.crop_region if crop_region_opt is None else crop_region_opt
        bbox = seg_elt.bbox if bbox_opt is None else bbox_opt
        label = seg_elt.label if label_opt is None else label_opt
        control_net_wrapper = seg_elt.control_net_wrapper if control_net_wrapper_opt is None else control_net_wrapper_opt

        cropped_image = cropped_image.numpy() if cropped_image is not None else None

        if isinstance(cropped_mask, torch.Tensor):
            if len(cropped_mask.shape) == 3:
                cropped_mask = cropped_mask.squeeze(0)

            cropped_mask = cropped_mask.numpy()

        seg = SEG(cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper)

        return (seg,)


class DilateMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "mask": ("MASK", ),
                     "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                }}

    RETURN_TYPES = ("MASK", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, mask, dilation):
        mask = core.dilate_mask(mask.numpy(), dilation)
        mask = torch.from_numpy(mask)
        mask = utils.make_3d_mask(mask)
        return (mask, )


class GaussianBlurMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "mask": ("MASK", ),
                     "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                     "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                }}

    RETURN_TYPES = ("MASK", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, mask, kernel_size, sigma):
        # Some custom nodes use abnormal 4-dimensional masks in the format of b, c, h, w. In the impact pack, internal 4-dimensional masks are required in the format of b, h, w, c. Therefore, normalization is performed using the normal mask format, which is 3-dimensional, before proceeding with the operation.
        mask = make_3d_mask(mask)
        mask = torch.unsqueeze(mask, dim=-1)
        mask = utils.tensor_gaussian_blur_mask(mask, kernel_size, sigma)
        mask = torch.squeeze(mask, dim=-1)
        return (mask, )


class DilateMaskInSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                }}

    RETURN_TYPES = ("SEGS", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, dilation):
        new_segs = []
        for seg in segs[1]:
            mask = core.dilate_mask(seg.cropped_mask, dilation)
            seg = SEG(seg.cropped_image, mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(seg)

        return ((segs[0], new_segs), )


class GaussianBlurMaskInSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                     "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                }}

    RETURN_TYPES = ("SEGS", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, kernel_size, sigma):
        new_segs = []
        for seg in segs[1]:
            mask = utils.tensor_gaussian_blur_mask(seg.cropped_mask, kernel_size, sigma)
            mask = torch.squeeze(mask, dim=-1).squeeze(0).numpy()
            seg = SEG(seg.cropped_image, mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(seg)

        return ((segs[0], new_segs), )


class Dilate_SEG_ELT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seg_elt": ("SEG_ELT", ),
                     "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                }}

    RETURN_TYPES = ("SEG_ELT", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, seg, dilation):
        mask = core.dilate_mask(seg.cropped_mask, dilation)
        seg = SEG(seg.cropped_image, mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
        return (seg,)


class SEG_ELT_BBOX_ScaleBy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seg": ("SEG_ELT", ),
                     "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}), }
                }

    RETURN_TYPES = ("SEG_ELT", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def fill_zero_outside_bbox(mask, crop_region, bbox):
        cx1, cy1, _, _ = crop_region
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1-cx1, y1-cy1, x2-cx1, y2-cy1
        h, w = mask.shape

        x1 = min(w-1, max(0, x1))
        x2 = min(w-1, max(0, x2))
        y1 = min(h-1, max(0, y1))
        y2 = min(h-1, max(0, y2))

        mask_cropped = mask.copy()
        mask_cropped[:, :x1] = 0  # zero fill left side
        mask_cropped[:, x2:] = 0  # zero fill right side
        mask_cropped[:y1, :] = 0  # zero fill top side
        mask_cropped[y2:, :] = 0  # zero fill bottom side
        return mask_cropped

    def doit(self, seg, scale_by):
        x1, y1, x2, y2 = seg.bbox
        w = x2-x1
        h = y2-y1

        dw = int((w * scale_by - w)/2)
        dh = int((h * scale_by - h)/2)

        bbox = (x1-dw, y1-dh, x2+dw, y2+dh)

        cropped_mask = SEG_ELT_BBOX_ScaleBy.fill_zero_outside_bbox(seg.cropped_mask, seg.crop_region, bbox)
        seg = SEG(seg.cropped_image, cropped_mask, seg.confidence, seg.crop_region, bbox, seg.label, seg.control_net_wrapper)
        return (seg,)


class EmptySEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self):
        shape = 0, 0
        return ((shape, []),)


class SegsToCombinedMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS",), }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs):
        mask = core.segs_to_combined_mask(segs)
        mask = utils.make_3d_mask(mask)
        return (mask,)


class MediaPipeFaceMeshToSEGS:
    @classmethod
    def INPUT_TYPES(s):
        bool_true_widget = ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"})
        bool_false_widget = ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"})
        return {"required": {
                                "image": ("IMAGE",),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                                "crop_min_size": ("INT", {"min": 10, "max": MAX_RESOLUTION, "step": 1, "default": 50}),
                                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 1}),
                                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                                "face": bool_true_widget,
                                "mouth": bool_false_widget,
                                "left_eyebrow": bool_false_widget,
                                "left_eye": bool_false_widget,
                                "left_pupil": bool_false_widget,
                                "right_eyebrow": bool_false_widget,
                                "right_eye": bool_false_widget,
                                "right_pupil": bool_false_widget,
                             },
                # "optional": {"reference_image_opt": ("IMAGE", ), }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, image, crop_factor, bbox_fill, crop_min_size, drop_size, dilation, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil):
        # padding is obsolete now
        # https://github.com/Fannovel16/comfyui_controlnet_aux/blob/1ec41fceff1ee99596445a0c73392fd91df407dc/utils.py#L33
        # def calc_pad(h_raw, w_raw):
        #     resolution = normalize_size_base_64(h_raw, w_raw)
        #
        #     def pad64(x):
        #         return int(np.ceil(float(x) / 64.0) * 64 - x)
        #
        #     k = float(resolution) / float(min(h_raw, w_raw))
        #     h_target = int(np.round(float(h_raw) * k))
        #     w_target = int(np.round(float(w_raw) * k))
        #
        #     return pad64(h_target), pad64(w_target)

        # if reference_image_opt is not None:
        #     if image.shape[1:] != reference_image_opt.shape[1:]:
        #         scale_by1 = reference_image_opt.shape[1] / image.shape[1]
        #         scale_by2 = reference_image_opt.shape[2] / image.shape[2]
        #         scale_by = min(scale_by1, scale_by2)
        #
        #         # padding is obsolete now
        #         # h_pad, w_pad = calc_pad(reference_image_opt.shape[1], reference_image_opt.shape[2])
        #         # if h_pad != 0:
        #         #     # height padded
        #         #     image = image[:, :-h_pad, :, :]
        #         # elif w_pad != 0:
        #         #     # width padded
        #         #     image = image[:, :, :-w_pad, :]
        #
        #         image = nodes.ImageScaleBy().upscale(image, "bilinear", scale_by)[0]

        result = core.mediapipe_facemesh_to_segs(image, crop_factor, bbox_fill, crop_min_size, drop_size, dilation, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil)
        return (result, )


class MaskToSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "mask": ("MASK",),
                                "combined": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                                "contour_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask, combined, crop_factor, bbox_fill, drop_size, contour_fill=False):
        mask = make_2d_mask(mask)

        result = core.mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
        return (result, )


class MaskToSEGS_for_AnimateDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "mask": ("MASK",),
                                "combined": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                                "contour_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask, combined, crop_factor, bbox_fill, drop_size, contour_fill=False):
        mask = make_2d_mask(mask)

        segs = core.mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)

        all_masks = SEGSToMaskList().doit(segs)[0]

        result_mask = (all_masks[0] * 255).to(torch.uint8)
        for mask in all_masks[1:]:
            result_mask |= (mask * 255).to(torch.uint8)

        result_mask = (result_mask/255.0).to(torch.float32)
        result_mask = utils.to_binary_mask(result_mask, 0.1)[0]

        return MaskToSEGS().doit(result_mask, False, crop_factor, False, drop_size, contour_fill)


class ControlNetApplySEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS",),
                    "control_net": ("CONTROL_NET",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    },
                "optional": {
                    "segs_preprocessor": ("SEGS_PREPROCESSOR",),
                    "control_image": ("IMAGE",)
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, control_net, strength, segs_preprocessor=None, control_image=None):
        new_segs = []

        for seg in segs[1]:
            control_net_wrapper = core.ControlNetWrapper(control_net, strength, segs_preprocessor, seg.control_net_wrapper,
                                                         original_size=segs[0], crop_region=seg.crop_region, control_image=control_image)
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, control_net_wrapper)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )


class ControlNetApplyAdvancedSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS",),
                    "control_net": ("CONTROL_NET",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                    },
                "optional": {
                    "segs_preprocessor": ("SEGS_PREPROCESSOR",),
                    "control_image": ("IMAGE",)
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, control_net, strength, start_percent, end_percent, segs_preprocessor=None, control_image=None):
        new_segs = []

        for seg in segs[1]:
            control_net_wrapper = core.ControlNetAdvancedWrapper(control_net, strength, start_percent, end_percent, segs_preprocessor,
                                                                 seg.control_net_wrapper, original_size=segs[0], crop_region=seg.crop_region,
                                                                 control_image=control_image)
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, control_net_wrapper)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )


class ControlNetClearSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS",), }, }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        new_segs = []

        for seg in segs[1]:
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )


class SEGSSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
                    "segs1": ("SEGS",),
                    },
                }

    RETURN_TYPES = ("SEGS", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, *args, **kwargs):
        input_name = f"segs{int(kwargs['select'])}"

        if input_name in kwargs:
            return (kwargs[input_name],)
        else:
            print(f"SEGSSwitch: invalid select index ('segs1' is selected)")
            return (kwargs['segs1'],)


class SEGSPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "picks": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                    "segs": ("SEGS",),
                    },
                "optional": {
                     "fallback_image_opt": ("IMAGE", ),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("SEGS", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, picks, segs, fallback_image_opt=None, unique_id=None):
        if fallback_image_opt is not None:
            segs = core.segs_scale_match(segs, fallback_image_opt.shape)

        # generate candidates image
        cands = []
        for seg in segs[1]:
            if seg.cropped_image is not None:
                cropped_image = seg.cropped_image
            elif fallback_image_opt is not None:
                # take from original image
                cropped_image = crop_image(fallback_image_opt, seg.crop_region)
            else:
                cropped_image = empty_pil_tensor()

            mask_array = seg.cropped_mask
            mask_array[mask_array < 0.3] = 0.3
            mask_array = mask_array[None, ..., None]
            cropped_image = cropped_image * mask_array

            cands.append(cropped_image)

        impact.impact_server.segs_picker_map[unique_id] = cands

        # pass only selected
        pick_ids = set()

        for pick in picks.split(","):
            try:
                pick_ids.add(int(pick)-1)
            except Exception:
                pass

        new_segs = []
        for i in pick_ids:
            if 0 <= i < len(segs[1]):
                new_segs.append(segs[1][i])

        return ((segs[0], new_segs),)


class DefaultImageForSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS", ),
                    "image": ("IMAGE", ),
                    "override": ("BOOLEAN", {"default": True}),
                }}

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, image, override):
        results = []

        segs = core.segs_scale_match(segs, image.shape)

        if len(segs[1]) > 0:
            if segs[1][0].cropped_image is not None:
                batch_count = len(segs[1][0].cropped_image)
            else:
                batch_count = len(image)

            for seg in segs[1]:
                if seg.cropped_image is not None and not override:
                    cropped_image = seg.cropped_image
                else:
                    cropped_image = None
                    for i in range(0, batch_count):
                        # take from original image
                        ref_image = image[i].unsqueeze(0)
                        cropped_image2 = crop_image(ref_image, seg.crop_region)

                        if cropped_image is None:
                            cropped_image = cropped_image2
                        else:
                            cropped_image = torch.cat((cropped_image, cropped_image2), dim=0)

                new_seg = SEG(cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                results.append(new_seg)

            return ((segs[0], results), )
        else:
            return (segs, )


class RemoveImageFromSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS", ), }}

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        results = []

        if len(segs[1]) > 0:
            for seg in segs[1]:
                new_seg = SEG(None, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                results.append(new_seg)

            return ((segs[0], results), )
        else:
            return (segs, )


class MakeTileSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "images": ("IMAGE", ),
                     "bbox_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                     "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                     "min_overlap": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
                     "filter_segs_dilation": ("INT", {"default": 20, "min": -255, "max": 255, "step": 1}),
                     "mask_irregularity": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                     "irregular_mask_mode": (["Reuse fast", "Reuse quality", "All random fast", "All random quality"],)
                    },
                "optional": {
                    "filter_in_segs_opt": ("SEGS", ),
                    "filter_out_segs_opt": ("SEGS", ),
                    }
                }

    RETURN_TYPES = ("SEGS",)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/__for_testing"

    def doit(self, images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, mask_irregularity=0, irregular_mask_mode="Reuse fast", filter_in_segs_opt=None, filter_out_segs_opt=None):
        if bbox_size <= 2*min_overlap:
            new_min_overlap = 2 / bbox_size
            print(f"[MakeTileSEGS] min_overlap should be greater than bbox_size. (value changed: {min_overlap} => {new_min_overlap})")
            min_overlap = new_min_overlap

        _, ih, iw, _ = images.size()

        mask_cache = None
        mask_quality = 512
        if mask_irregularity > 0:
            if irregular_mask_mode == "Reuse fast":
                mask_quality = 128
                mask_cache = np.zeros((128, 128)).astype(np.float32)
                core.random_mask(mask_cache, (0, 0, 128, 128), factor=mask_irregularity, size=mask_quality)
            elif irregular_mask_mode == "Reuse quality":
                mask_quality = 512
                mask_cache = np.zeros((512, 512)).astype(np.float32)
                core.random_mask(mask_cache, (0, 0, 512, 512), factor=mask_irregularity, size=mask_quality)
            elif irregular_mask_mode == "All random fast":
                mask_quality = 512

        # create exclusion mask
        if filter_out_segs_opt is not None:
            exclusion_mask = core.segs_to_combined_mask(filter_out_segs_opt)
            exclusion_mask = utils.make_3d_mask(exclusion_mask)
            exclusion_mask = utils.resize_mask(exclusion_mask, (ih, iw))
            exclusion_mask = dilate_mask(exclusion_mask.cpu().numpy(), filter_segs_dilation)
        else:
            exclusion_mask = None

        if filter_in_segs_opt is not None:
            and_mask = core.segs_to_combined_mask(filter_in_segs_opt)
            and_mask = utils.make_3d_mask(and_mask)
            and_mask = utils.resize_mask(and_mask, (ih, iw))
            and_mask = dilate_mask(and_mask.cpu().numpy(), filter_segs_dilation)

            a, b = core.mask_to_segs(and_mask, True, 1.0, False, 0)
            if len(b) == 0:
                return a, b

            start_x, start_y, c, d = b[0].crop_region
            w = c - start_x
            h = d - start_y
        else:
            start_x = 0
            start_y = 0
            h, w = ih, iw
            and_mask = None

        # calculate tile factors
        if bbox_size > h or bbox_size > w:
            new_bbox_size = min(bbox_size, min(w, h))
            print(f"[MaskTileSEGS] bbox_size is greater than resolution (value changed: {bbox_size} => {new_bbox_size}")
            bbox_size = new_bbox_size

        n_horizontal = int(w / (bbox_size - min_overlap))
        n_vertical = int(h / (bbox_size - min_overlap))

        w_overlap_sum = (bbox_size * n_horizontal) - w
        if w_overlap_sum < 0:
            n_horizontal += 1
            w_overlap_sum = (bbox_size * n_horizontal) - w

        w_overlap_size = 0 if n_horizontal == 1 else int(w_overlap_sum/(n_horizontal-1))

        h_overlap_sum = (bbox_size * n_vertical) - h
        if h_overlap_sum < 0:
            n_vertical += 1
            h_overlap_sum = (bbox_size * n_vertical) - h

        h_overlap_size = 0 if n_vertical == 1 else int(h_overlap_sum/(n_vertical-1))

        new_segs = []

        y = start_y
        for j in range(0, n_vertical):
            x = start_x
            for i in range(0, n_horizontal):
                x1 = x
                y1 = y

                if x+bbox_size < iw-1:
                    x2 = x+bbox_size
                else:
                    x2 = iw
                    x1 = iw-bbox_size

                if y+bbox_size < ih-1:
                    y2 = y+bbox_size
                else:
                    y2 = ih
                    y1 = ih-bbox_size

                bbox = x1, y1, x2, y2
                crop_region = make_crop_region(iw, ih, bbox, crop_factor)
                cx1, cy1, cx2, cy2 = crop_region

                mask = np.zeros((cy2 - cy1, cx2 - cx1)).astype(np.float32)

                rel_left = x1 - cx1
                rel_top = y1 - cy1
                rel_right = x2 - cx1
                rel_bot = y2 - cy1

                if mask_irregularity > 0:
                    if mask_cache is not None:
                        core.adaptive_mask_paste(mask, mask_cache, (rel_left, rel_top, rel_right, rel_bot))
                    else:
                        core.random_mask(mask, (rel_left, rel_top, rel_right, rel_bot), factor=mask_irregularity, size=mask_quality)

                    # corner filling
                    if rel_left == 0:
                        pad = int((x2 - x1) / 8)
                        mask[rel_top:rel_bot, :pad] = 1.0

                    if rel_top == 0:
                        pad = int((y2 - y1) / 8)
                        mask[:pad, rel_left:rel_right] = 1.0

                    if rel_right == mask.shape[1]:
                        pad = int((x2 - x1) / 8)
                        mask[rel_top:rel_bot, -pad:] = 1.0

                    if rel_bot == mask.shape[0]:
                        pad = int((y2 - y1) / 8)
                        mask[-pad:, rel_left:rel_right] = 1.0
                else:
                    mask[rel_top:rel_bot, rel_left:rel_right] = 1.0

                mask = torch.tensor(mask)

                if exclusion_mask is not None:
                    exclusion_mask_cropped = exclusion_mask[cy1:cy2, cx1:cx2]
                    mask[exclusion_mask_cropped != 0] = 0.0

                if and_mask is not None:
                    and_mask_cropped = and_mask[cy1:cy2, cx1:cx2]
                    mask[and_mask_cropped == 0] = 0.0

                is_mask_zero = torch.all(mask == 0.0).item()

                if not is_mask_zero:
                    item = SEG(None, mask.numpy(), 1.0, crop_region, bbox, "", None)
                    new_segs.append(item)

                x += bbox_size - w_overlap_size
            y += bbox_size - h_overlap_size

        res = (ih, iw), new_segs  # segs
        return (res,)
