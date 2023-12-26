import copy
import os

import numpy
import torch
from segment_anything import SamPredictor
import torch.nn.functional as F

from impact.utils import *
from collections import namedtuple
import numpy as np
from skimage.measure import label, regionprops

import nodes
import comfy_extras.nodes_upscale_model as model_upscale
from server import PromptServer
import comfy
import impact.wildcards as wildcards
import math
import cv2
import time
from impact import utils

from thirdparty import noise_nodes

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

pb_id_cnt = time.time()
preview_bridge_image_id_map = {}
preview_bridge_image_name_map = {}
preview_bridge_cache = {}


def set_previewbridge_image(node_id, file, item):
    global pb_id_cnt

    if file in preview_bridge_image_name_map:
        pb_id = preview_bridge_image_name_map[node_id, file]
        if pb_id.startswith(f"${node_id}"):
            return pb_id

    pb_id = f"${node_id}-{pb_id_cnt}"
    preview_bridge_image_id_map[pb_id] = (file, item)
    preview_bridge_image_name_map[node_id, file] = (pb_id, item)
    pb_id_cnt += 1

    return pb_id


def erosion_mask(mask, grow_mask_by):
    mask = make_2d_mask(mask)

    w = mask.shape[1]
    h = mask.shape[0]

    device = comfy.model_management.get_torch_device()
    mask = mask.clone().to(device)
    mask2 = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(w, h), mode="bilinear").to(device)
    if grow_mask_by == 0:
        mask_erosion = mask2
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by)).to(device)
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask2.round(), kernel_tensor, padding=padding), 0, 1)

    return mask_erosion[:, :, :w, :h].round().cpu()


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                     refiner_negative=None):
    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        refined_latent = \
            nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise)[0]
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = \
            nodes.KSamplerAdvanced().sample(model, "enable", seed, advanced_steps, cfg, sampler_name, scheduler,
                                            positive, negative, latent_image, start_at_step, end_at_step,
                                            "enable")[0]

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     nodes.KSamplerAdvanced().sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                     scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                     end_at_step, "enable")[0]

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = \
                latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = \
            nodes.KSamplerAdvanced().sample(refiner_model, "disable", seed, advanced_steps, cfg, sampler_name, scheduler,
                                            refiner_positive, refiner_negative, temp_latent, end_at_step,
                                            advanced_steps + 1,
                                            "disable")[0]

    return refined_latent


class REGIONAL_PROMPT:
    def __init__(self, mask, sampler):
        mask = make_2d_mask(mask)

        self.mask = mask
        self.sampler = sampler
        self.mask_erosion = None
        self.erosion_factor = None

    def get_mask_erosion(self, factor):
        if self.mask_erosion is None or self.erosion_factor != factor:
            self.mask_erosion = erosion_mask(self.mask, factor)
            self.erosion_factor = factor

        return self.mask_erosion


class NO_BBOX_DETECTOR:
    pass


class NO_SEGM_DETECTOR:
    pass


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    mask = make_2d_mask(mask)

    points = []
    plabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(mask.shape[0] / 20))
    x_step = max(3, int(mask.shape[1] / 20))

    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x + j, y + i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x + j, y + i))
                plabs.append(0)

    return points, plabs


def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(w / 20))
    x_step = max(3, int(h / 20))

    for i in range(10, h - 10, y_step):
        for j in range(10, w - 10, x_step):
            if not (x1 - 10 <= j and j <= x2 + 10 and y1 - 10 <= i and i <= y2 + 10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs


def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name,
                   scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None,
                   detailer_hook=None,
                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                   refiner_negative=None, control_net_wrapper=None, cycle=1):
    if noise_mask is not None and len(noise_mask.shape) == 3:
        noise_mask = noise_mask.squeeze(0)

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        print(f"Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    # safeguard
    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            print(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None

        if new_w == 0 or new_h == 0:
            print(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            print(f"Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale
    upscaled_image = tensor_resize(image, new_w, new_h)

    # ksampler
    latent_image = to_latent_image(upscaled_image, vae)

    upscaled_mask = None
    if noise_mask is not None:
        # upscale the mask tensor by a factor of 2 using bilinear interpolation
        noise_mask = torch.from_numpy(noise_mask)
        upscaled_mask = torch.nn.functional.interpolate(noise_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)

        # remove the extra dimensions added by unsqueeze
        upscaled_mask = upscaled_mask.squeeze(0).squeeze(0)
        latent_image['noise_mask'] = upscaled_mask

    if detailer_hook is not None:
        latent_image = detailer_hook.post_encode(latent_image)

    cnet_pil = None
    if control_net_wrapper is not None:
        positive, cnet_pil = control_net_wrapper.apply(positive, upscaled_image, upscaled_mask)

    refined_latent = latent_image

    for i in range(0, cycle):
        if detailer_hook is not None:
            if detailer_hook is not None:
                detailer_hook.set_steps((i, cycle))

            refined_latent = detailer_hook.cycle_latent(refined_latent)

            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                detailer_hook.pre_ksample(model, seed+i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
        else:
            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise

        refined_latent = ksampler_wrapper(model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2,
                                          refined_latent, denoise2,
                                          refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative)

    if detailer_hook is not None:
        refined_latent = detailer_hook.pre_decode(refined_latent)

    # non-latent downscale - latent downscale cause bad quality
    refined_image = vae.decode(refined_latent['samples'])

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    # downscale
    refined_image = tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pil


def enhance_detail_for_animatediff(image_frames, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                                   sampler_name,
                                   scheduler, positive, negative, denoise, noise_mask,
                                   wildcard_opt=None, wildcard_opt_concat_mode=None,
                                   detailer_hook=None,
                                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                                   refiner_negative=None):
    if noise_mask is not None and len(noise_mask.shape) == 3:
        noise_mask = noise_mask.squeeze(0)

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive

    h = image_frames.shape[1]
    w = image_frames.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    # safeguard
    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        print(f"Detailer: force inpaint")
        upscale = 1.0
        new_w = w
        new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale the mask tensor by a factor of 2 using bilinear interpolation
    if isinstance(noise_mask, numpy.ndarray):
        noise_mask = torch.from_numpy(noise_mask)

    if len(noise_mask.shape) == 2:
        noise_mask = noise_mask.unsqueeze(0)
    else:  # == 3
        noise_mask = noise_mask

    upscaled_mask = None

    for single_mask in noise_mask:
        single_mask = single_mask.unsqueeze(0).unsqueeze(0)
        upscaled_single_mask = torch.nn.functional.interpolate(single_mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
        upscaled_single_mask = upscaled_single_mask.squeeze(0)

        if upscaled_mask is None:
            upscaled_mask = upscaled_single_mask
        else:
            upscaled_mask = torch.cat((upscaled_mask, upscaled_single_mask), dim=0)

    latent_frames = None
    for image in image_frames:
        image = torch.from_numpy(image).unsqueeze(0)

        # upscale
        upscaled_image = tensor_resize(image, new_w, new_h)

        # ksampler
        samples = to_latent_image(upscaled_image, vae)['samples']

        if latent_frames is None:
            latent_frames = samples
        else:
            latent_frames = torch.concat((latent_frames, samples), dim=0)

    if len(upscaled_mask) != len(image_frames) and len(upscaled_mask) > 1:
        print(f"[Impact Pack] WARN: DetailerForAnimateDiff - The number of the mask frames({len(upscaled_mask)}) and the image frames({len(image_frames)}) are different. Combine the mask frames and apply.")
        combined_mask = upscaled_mask[0].to(torch.uint8)

        for frame_mask in upscaled_mask[1:]:
            combined_mask |= (frame_mask * 255).to(torch.uint8)

        combined_mask = (combined_mask/255.0).to(torch.float32)

        upscaled_mask = combined_mask.expand(len(image_frames), -1, -1)
        upscaled_mask = utils.to_binary_mask(upscaled_mask, 0.1)

    latent = {
        'noise_mask': upscaled_mask,
        'samples': latent_frames
    }

    if detailer_hook is not None:
        latent = detailer_hook.post_encode(latent)

    refined_latent = ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent, denoise,
                                      refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative)

    if detailer_hook is not None:
        refined_latent = detailer_hook.pre_decode(refined_latent)

    refined_image_frames = None
    for refined_sample in refined_latent['samples']:
        refined_sample = refined_sample.unsqueeze(0)

        # non-latent downscale - latent downscale cause bad quality
        refined_image = vae.decode(refined_sample)

        if refined_image_frames is None:
            refined_image_frames = refined_image
        else:
            refined_image_frames = torch.concat((refined_image_frames, refined_image), dim=0)

    if detailer_hook is not None:
        refined_image_frames = detailer_hook.post_decode(refined_image_frames)

    refined_image_frames = nodes.ImageScale().upscale(image=refined_image_frames, upscale_method='lanczos', width=w, height=h, crop='disabled')[0]

    return refined_image_frames


def composite_to(dest_latent, crop_region, src_latent):
    x1 = crop_region[0]
    y1 = crop_region[1]

    # composite to original latent
    lc = nodes.LatentComposite()
    orig_image = lc.composite(dest_latent, src_latent, x1, y1)

    return orig_image[0]


def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []

    selected = False
    max_score = 0
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected:
        total_masks.append(max_mask)

    return total_masks


def make_sam_mask(sam_model, segs, image, detection_hint, dilation,
                  threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
    if sam_model.is_auto_mode:
        device = comfy.model_management.get_torch_device()
        sam_model.to(device=device)

    try:
        predictor = SamPredictor(sam_model)
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        predictor.set_image(image, "RGB")

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(segs[i].bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_predict(predictor, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]  # 1 = foreground point, 0 = background point

                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.append((x1 + gap, center[1]))
                    points.append((x1 + gap * 2, center[1]))
                    plabs = [1, 1]

                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.append((center[0], y1 + gap))
                    points.append((center[0], y1 + gap * 2))
                    plabs = [1, 1]

                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, center[1]))
                    points.append((x1 + x_gap * 2, center[1]))
                    points.append((center[0], y1 + y_gap))
                    points.append((center[0], y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, y1 + y_gap))
                    points.append((x1 + x_gap * 2, y1 + y_gap))
                    points.append((x1 + x_gap, y1 + y_gap * 2))
                    points.append((x1 + x_gap * 2, y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "mask-point-bbox":
                    center = center_of_bbox(segs[i].bbox)
                    points.append(center)
                    plabs = [1]

                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(segs[i].crop_region[0], segs[i].crop_region[1],
                                                                       segs[i].cropped_mask,
                                                                       mask_hint_threshold, use_small_negative)

                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1],
                                                         segs[i].crop_region[0], segs[i].crop_region[1],
                                                         segs[i].crop_region[2], segs[i].crop_region[3])

                    points += npoints
                    plabs += nplabs

                detected_masks = sam_predict(predictor, points, plabs, dilated_bbox, threshold)
                total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        if sam_model.is_auto_mode:
            print(f"semd to {device}")
            sam_model.to(device="cpu")

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
    else:
        mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")  # empty mask

    return mask


def generate_detection_hints(image, seg, center, detection_hint, dilated_bbox, mask_hint_threshold, use_small_negative,
                             mask_hint_use_negative):
    [x1, y1, x2, y2] = dilated_bbox

    points = []
    plabs = []
    if detection_hint == "center-1":
        points.append(center)
        plabs = [1]  # 1 = foreground point, 0 = background point

    elif detection_hint == "horizontal-2":
        gap = (x2 - x1) / 3
        points.append((x1 + gap, center[1]))
        points.append((x1 + gap * 2, center[1]))
        plabs = [1, 1]

    elif detection_hint == "vertical-2":
        gap = (y2 - y1) / 3
        points.append((center[0], y1 + gap))
        points.append((center[0], y1 + gap * 2))
        plabs = [1, 1]

    elif detection_hint == "rect-4":
        x_gap = (x2 - x1) / 3
        y_gap = (y2 - y1) / 3
        points.append((x1 + x_gap, center[1]))
        points.append((x1 + x_gap * 2, center[1]))
        points.append((center[0], y1 + y_gap))
        points.append((center[0], y1 + y_gap * 2))
        plabs = [1, 1, 1, 1]

    elif detection_hint == "diamond-4":
        x_gap = (x2 - x1) / 3
        y_gap = (y2 - y1) / 3
        points.append((x1 + x_gap, y1 + y_gap))
        points.append((x1 + x_gap * 2, y1 + y_gap))
        points.append((x1 + x_gap, y1 + y_gap * 2))
        points.append((x1 + x_gap * 2, y1 + y_gap * 2))
        plabs = [1, 1, 1, 1]

    elif detection_hint == "mask-point-bbox":
        center = center_of_bbox(seg.bbox)
        points.append(center)
        plabs = [1]

    elif detection_hint == "mask-area":
        points, plabs = gen_detection_hints_from_mask_area(seg.crop_region[0], seg.crop_region[1],
                                                           seg.cropped_mask,
                                                           mask_hint_threshold, use_small_negative)

    if mask_hint_use_negative == "Outter":
        npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1],
                                             seg.crop_region[0], seg.crop_region[1],
                                             seg.crop_region[2], seg.crop_region[3])

        points += npoints
        plabs += nplabs

    return points, plabs


def convert_and_stack_masks(masks):
    if len(masks) == 0:
        return None

    mask_tensors = []
    for mask in masks:
        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensors.append(mask_tensor)

    stacked_masks = torch.stack(mask_tensors, dim=0)
    stacked_masks = stacked_masks.unsqueeze(1)

    return stacked_masks


def merge_and_stack_masks(stacked_masks, group_size):
    if stacked_masks is None:
        return None

    num_masks = stacked_masks.size(0)
    merged_masks = []

    for i in range(0, num_masks, group_size):
        subset_masks = stacked_masks[i:i + group_size]
        merged_mask = torch.any(subset_masks, dim=0)
        merged_masks.append(merged_mask)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks, dim=0)

    return merged_masks


def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        cropped_mask = torch.from_numpy(cropped_mask)
        cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = tensor_resize(torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return ((th, tw), new_segs)


# Used Python's slicing feature. stacked_masks[2::3] means starting from index 2, selecting every third tensor with a step size of 3.
# This allows for quickly obtaining the last tensor of every three tensors in stacked_masks.
def every_three_pick_last(stacked_masks):
    selected_masks = stacked_masks[2::3]
    return selected_masks


def make_sam_mask_segmented(sam_model, segs, image, detection_hint, dilation,
                            threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
    if sam_model.is_auto_mode:
        device = comfy.model_management.get_torch_device()
        sam_model.to(device=device)

    try:
        predictor = SamPredictor(sam_model)
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        predictor.set_image(image, "RGB")

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_predict(predictor, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)
                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points, plabs = generate_detection_hints(image, segs[i], center, detection_hint, dilated_bbox,
                                                         mask_hint_threshold, use_small_negative,
                                                         mask_hint_use_negative)

                detected_masks = sam_predict(predictor, points, plabs, dilated_bbox, threshold)

                total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        # Temporarily disabling the switch back to CPU after inference.
        # Rationale: After multiple tests and comparisons, it's concluded that not only does it fail to conserve GPU memory, 
        # but it also introduces additional IO overhead from transferring the model between devices.

        # if sam_model.is_auto_mode:
        #     sam_model.to(device=torch.device("cpu"))

        pass

    mask_working_device = torch.device("cpu")

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
        mask = mask.to(device=mask_working_device)
    else:
        # Extracting batch, height and width
        height, width, _ = image.shape
        mask = torch.zeros(
            (height, width), dtype=torch.float32, device=mask_working_device
        )  # empty mask

    stacked_masks = convert_and_stack_masks(total_masks)

    return (mask, merge_and_stack_masks(stacked_masks, group_size=3))
    # return every_three_pick_last(stacked_masks)


def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)

    if mask is None:
        print("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return ([],)

    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items


def apply_mask_to_each_seg(segs, masks):
    if masks is None:
        print("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return (segs[0], [],)

    items = []

    masks = masks.squeeze(1)

    for seg, mask in zip(segs[1], masks):
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = (mask.cpu().numpy() * 255).astype(np.uint8)
        cropped_mask2 = cropped_mask2[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items


def dilate_segs(segs, factor):
    if factor == 0:
        return segs

    new_segs = []
    for seg in segs[1]:
        new_mask = dilate_mask(seg.cropped_mask, factor)
        new_seg = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (segs[0], new_segs)


class ONNXDetector:
    onnx_model = None

    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        try:
            import impact.onnx as onnx

            h = image.shape[1]
            w = image.shape[2]

            labels, scores, boxes = onnx.onnx_inference(image, self.onnx_model)

            # collect feasible item
            result = []

            for i in range(len(labels)):
                if scores[i] > threshold:
                    item_bbox = boxes[i]
                    x1, y1, x2, y2 = item_bbox

                    if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                        crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                        if detailer_hook is not None:
                            crop_region = item_bbox.post_crop_region(w, h, item_bbox, crop_region)

                        crop_x1, crop_y1, crop_x2, crop_y2, = crop_region

                        # prepare cropped mask
                        cropped_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1))
                        cropped_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = 1
                        cropped_mask = dilate_mask(cropped_mask, dilation)

                        # make items. just convert the integer label to a string
                        item = SEG(None, cropped_mask, scores[i], crop_region, item_bbox, str(labels[i]), None)
                        result.append(item)

            shape = h, w
            return shape, result
        except Exception as e:
            print(f"ONNXDetector: unable to execute.\n{e}")
            pass

    def detect_combined(self, image, threshold, dilation):
        return segs_to_combined_mask(self.detect(image, threshold, dilation, 1))

    def setAux(self, x):
        pass


def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None, is_contour=True):
    drop_size = max(drop_size, 1)
    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    if isinstance(mask, np.ndarray):
        pass  # `mask` is already a NumPy array
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            print("[mask_to_segs] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)

    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    for i in range(mask.shape[0]):
        mask_i = mask[i]

        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (
                    np.min(indices[1]),
                    np.min(indices[0]),
                    np.max(indices[1]),
                    np.max(indices[0]),
                )
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor
                )
                x1, y1, x2, y2 = crop_region

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]

                    if cropped_mask is not None:
                        item = SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None)
                        result.append(item)

        else:
            mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
            contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                hierarchy = ctree[0][j]
                if hierarchy[3] != -1:
                    continue

                separated_mask = np.zeros_like(mask_i_uint8)
                cv2.drawContours(separated_mask, [contour], 0, 255, -1)
                separated_mask = np.array(separated_mask / 255.0).astype(np.float32)

                x, y, w, h = cv2.boundingRect(contour)
                bbox = x, y, x + w, y + h
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor, crop_min_size
                )

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if w > drop_size and h > drop_size:
                    if is_contour:
                        mask_src = separated_mask
                    else:
                        mask_src = mask_i * separated_mask

                    cropped_mask = np.array(
                        mask_src[
                            crop_region[1]: crop_region[3],
                            crop_region[0]: crop_region[2],
                        ]
                    )

                    if bbox_fill:
                        cx1, cy1, _, _ = crop_region
                        bx1 = x - cx1
                        bx2 = x+w - cx1
                        by1 = y - cy1
                        by2 = y+h - cy1
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        cropped_mask = utils.to_binary_mask(torch.from_numpy(cropped_mask), 0.1)[0]
                        item = SEG(None, cropped_mask.numpy(), 1.0, crop_region, bbox, label, None)
                        result.append(item)

    if not result:
        print(f"[mask_to_segs] Empty mask.")

    print(f"# of Detected SEGS: {len(result)}")
    # for r in result:
    #     print(f"\tbbox={r.bbox}, crop={r.crop_region}, label={r.label}")

    # shape: (b,h,w) -> (h,w)
    return (mask.shape[1], mask.shape[2]), result


def mediapipe_facemesh_to_segs(image, crop_factor, bbox_fill, crop_min_size, drop_size, dilation, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil):
    parts = {
        "face": np.array([0x0A, 0xC8, 0x0A]),
        "mouth": np.array([0x0A, 0xB4, 0x0A]),
        "left_eyebrow": np.array([0xB4, 0xDC, 0x0A]),
        "left_eye": np.array([0xB4, 0xC8, 0x0A]),
        "left_pupil": np.array([0xFA, 0xC8, 0x0A]),
        "right_eyebrow": np.array([0x0A, 0xDC, 0xB4]),
        "right_eye": np.array([0x0A, 0xC8, 0xB4]),
        "right_pupil": np.array([0x0A, 0xC8, 0xFA]),
    }

    def create_segments(image, color):
        image = (image * 255).to(torch.uint8)
        image = image.squeeze(0).numpy()
        mask = cv2.inRange(image, color, color)

        contours, ctree = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_list = []
        for i, contour in enumerate(contours):
            hierarchy = ctree[0][i]
            if hierarchy[3] == -1:
                convex_hull = cv2.convexHull(contour)
                convex_segment = np.zeros_like(image)
                cv2.fillPoly(convex_segment, [convex_hull], (255, 255, 255))

                convex_segment = np.expand_dims(convex_segment, axis=0).astype(np.float32) / 255.0
                tensor = torch.from_numpy(convex_segment)
                mask_tensor = torch.any(tensor != 0, dim=-1).float()
                mask_tensor = mask_tensor.squeeze(0)
                mask_tensor = torch.from_numpy(dilate_mask(mask_tensor.numpy(), dilation))
                mask_list.append(mask_tensor.unsqueeze(0))

        return mask_list

    segs = []

    def create_seg(label):
        mask_list = create_segments(image, parts[label])
        for mask in mask_list:
            seg = mask_to_segs(mask, False, crop_factor, bbox_fill, drop_size=drop_size, label=label, crop_min_size=crop_min_size)
            if len(seg[1]) > 0:
                segs.extend(seg[1])

    if face:
        create_seg('face')

    if mouth:
        create_seg('mouth')

    if left_eyebrow:
        create_seg('left_eyebrow')

    if left_eye:
        create_seg('left_eye')

    if left_pupil:
        create_seg('left_pupil')

    if right_eyebrow:
        create_seg('right_eyebrow')

    if right_eye:
        create_seg('right_eye')

    if right_pupil:
        create_seg('right_pupil')

    return (image.shape[1], image.shape[2]), segs


def segs_to_combined_mask(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        crop_region = seg.crop_region
        mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

    return torch.from_numpy(mask.astype(np.float32) / 255.0)


def segs_to_masklist(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    masks = []
    for seg in segs[1]:
        if isinstance(seg.cropped_mask, numpy.ndarray):
            cropped_mask = torch.from_numpy(seg.cropped_mask)
        else:
            cropped_mask = seg.cropped_mask

        if cropped_mask.ndim == 2:
            cropped_mask = cropped_mask.unsqueeze(0)

        n = len(cropped_mask)

        mask = torch.zeros((n, h, w), dtype=torch.uint8)
        crop_region = seg.crop_region
        mask[:, crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).to(torch.uint8)
        mask = (mask / 255.0).to(torch.float32)

        for x in mask:
            masks.append(x)

    if len(masks) == 0:
        empty_mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
        masks = [empty_mask]

    return masks


def vae_decode(vae, samples, use_tile, hook, tile_size=512):
    if use_tile:
        pixels = nodes.VAEDecodeTiled().decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        pixels = hook.post_decode(pixels)

    return pixels


def vae_encode(vae, pixels, use_tile, hook, tile_size=512):
    if use_tile:
        samples = nodes.VAEEncodeTiled().encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        samples = hook.post_encode(samples)

    return samples


class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 denoise)

        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                     denoise=denoise)[0]


class KSamplerAdvancedWrapper:
    params = None

    def __init__(self, model, cfg, sampler_name, scheduler, positive, negative):
        self.params = model, cfg, sampler_name, scheduler, positive, negative

    def sample_advanced(self, add_noise, seed, steps, latent_image, start_at_step, end_at_step,
                        return_with_leftover_noise, hook=None, recover_special_sampler=False):
        model, cfg, sampler_name, scheduler, positive, negative = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent = \
                hook.pre_ksample_advanced(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step,
                                          return_with_leftover_noise)

        if recover_special_sampler and sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']:
            base_image = latent_image.copy()
        else:
            base_image = None

        try:
            latent_image = nodes.KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                           positive, negative, latent_image, start_at_step, end_at_step,
                                                           return_with_leftover_noise)[0]
        except ValueError as e:
            if str(e) == 'sigma_min and sigma_max must not be 0':
                print(f"\nWARN: sampling skipped - sigma_min and sigma_max are 0")
                return latent_image

        if recover_special_sampler and sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu']:
            compensate = 0 if sampler_name in ['uni_pc', 'uni_pc_bh2'] else 2
            sampler_name = 'dpmpp_fast' if sampler_name in ['uni_pc', 'uni_pc_bh2', 'dpmpp_sde', 'dpmpp_sde_gpu'] else 'dpmpp_2m'
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()

            noise_mask = latent_image['noise_mask']

            if len(noise_mask.shape) == 4:
                noise_mask = noise_mask.squeeze(0).squeeze(0)

            latent_image = \
                latent_compositor.composite(base_image, latent_image, 0, 0, False, noise_mask)[0]

            try:
                latent_image = nodes.KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                               positive, negative, latent_image, start_at_step-compensate, end_at_step,
                                                               return_with_leftover_noise)[0]
            except ValueError as e:
                if str(e) == 'sigma_min and sigma_max must not be 0':
                    print(f"\nWARN: sampling skipped - sigma_min and sigma_max are 0")

        return latent_image


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


class DetailerHook(PixelKSampleHook):
    def cycle_latent(self, latent):
        return latent


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


class SimpleCfgScheduleHook(PixelKSampleHook):
    target_cfg = 0

    def __init__(self, target_cfg):
        super().__init__()
        self.target_cfg = target_cfg

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        progress = self.cur_step / self.total_step
        gap = self.target_cfg - cfg
        current_cfg = cfg + gap * progress
        return model, seed, steps, current_cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class SimpleDenoiseScheduleHook(PixelKSampleHook):
    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        progress = self.cur_step / self.total_step
        gap = self.target_denoise - denoise
        current_denoise = denoise + gap * progress
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, current_denoise


class SimpleDetailerDenoiseSchedulerHook(DetailerHook):
    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        progress = self.cur_step / self.total_step
        gap = self.target_denoise - denoise
        current_denoise = denoise + gap * progress
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, current_denoise


def latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae, use_tile=False, tile_size=512,
                                        save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size)


def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512,
                                  save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return (vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size), pixels)


def latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512,
                                  save_temp_prefix=None, hook=None):
	return latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook)[0]


def latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, upscale_model, new_w, new_h, vae,
                                                   use_tile=False, tile_size=512, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            print(f"[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size)


def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                              tile_size=512, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]
        if current_w == w:
            print(f"[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return (vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size), pixels)

def latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                             tile_size=512, save_temp_prefix=None, hook=None):
    return latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook)[0]


class TwoSamplersForMaskUpscaler:
    def __init__(self, scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
                 full_sampler_opt=None, upscale_model_opt=None, hook_base_opt=None, hook_mask_opt=None,
                 hook_full_opt=None,
                 tile_size=512):

        mask = make_2d_mask(mask)

        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        self.params = scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae
        self.upscale_model = upscale_model_opt
        self.full_sampler = full_sampler_opt
        self.hook_base = hook_base_opt
        self.hook_mask = hook_mask_opt
        self.hook_full = hook_full_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.vae = vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix,
                                                            hook=self.hook_base, tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model,
                                                                       upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix,
                                                                       hook=self.hook_mask, tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def prepare_hook(self, step_info):
        if self.hook_base is not None:
            self.hook_base.set_steps(step_info)
        if self.hook_mask is not None:
            self.hook_mask.set_steps(step_info)
        if self.hook_full is not None:
            self.hook_full.set_steps(step_info)

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=self.use_tiled_vae,
                                                                  save_temp_prefix=save_temp_prefix,
                                                                  hook=self.hook_base,
                                                                  tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model,
                                                                             w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix,
                                                                             hook=self.hook_mask,
                                                                             tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def is_full_sample_time(self, step_info, sample_schedule):
        cur_step, total_step = step_info

        # make start from 1 instead of zero
        cur_step += 1
        total_step += 1

        if sample_schedule == "none":
            return False

        elif sample_schedule == "interleave1":
            return cur_step % 2 == 0

        elif sample_schedule == "interleave2":
            return cur_step % 3 == 0

        elif sample_schedule == "interleave3":
            return cur_step % 4 == 0

        elif sample_schedule == "last1":
            return cur_step == total_step

        elif sample_schedule == "last2":
            return cur_step >= total_step - 1

        elif sample_schedule == "interleave1+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

        elif sample_schedule == "interleave2+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

        elif sample_schedule == "interleave3+last1":
            return cur_step % 2 == 0 or cur_step >= total_step - 1

    def do_samples(self, step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent):
        mask = make_2d_mask(mask)

        if self.is_full_sample_time(step_info, sample_schedule):
            print(f"step_info={step_info} / full time")

            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)
            sampler = self.full_sampler if self.full_sampler is not None else base_sampler
            return sampler.sample(upscaled_latent, self.hook_full)

        else:
            print(f"step_info={step_info} / non-full time")
            # upscale mask
            if mask.ndim == 2:
                mask = mask[None, :, :, None]
            upscaled_mask = F.interpolate(mask, size=(upscaled_latent['samples'].shape[2], upscaled_latent['samples'].shape[3]), mode='bilinear', align_corners=True)
            upscaled_mask = upscaled_mask[:, :, :upscaled_latent['samples'].shape[2], :upscaled_latent['samples'].shape[3]]

            # base sampler
            upscaled_inv_mask = torch.where(upscaled_mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))
            upscaled_latent['noise_mask'] = upscaled_inv_mask
            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)

            # mask sampler
            upscaled_latent['noise_mask'] = upscaled_mask
            upscaled_latent = mask_sampler.sample(upscaled_latent, self.hook_mask)

            # remove mask
            del upscaled_latent['noise_mask']
            return upscaled_latent


class PixelKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 use_tiled_vae, upscale_model_opt=None, hook_opt=None, tile_size=512):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.is_tiled = False
        self.vae = vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix, hook=self.hook)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model,
                                                                       upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix,
                                                                       hook=self.hook,
                                                                       tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                 positive, negative, upscaled_latent, denoise)[0]
        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=self.use_tiled_vae,
                                                                  save_temp_prefix=save_temp_prefix, hook=self.hook,
                                                                  tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model,
                                                                             w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix,
                                                                             hook=self.hook,
                                                                             tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                 positive, negative, upscaled_latent, denoise)[0]
        return refined_latent


class ControlNetWrapper:
    def __init__(self, control_net, strength, preprocessor):
        self.control_net = control_net
        self.strength = strength
        self.preprocessor = preprocessor
        self.image = None

    def apply(self, conditioning, image, mask=None):
        if self.preprocessor is not None:
            image = self.preprocessor.apply(image, mask)

        return nodes.ControlNetApply().apply_controlnet(conditioning, self.control_net, image, self.strength)[0], image


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

        Unsampler = noise_nodes.Unsampler

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


# REQUIREMENTS: BlenderNeko/ComfyUI_TiledKSampler
class TiledKSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 tile_width, tile_height, tiling_strategy):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy

    def sample(self, latent_image, hook=None):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'TiledKSamplerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise Exception("'BNK_TiledKSampler' node isn't installed.")

        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 denoise)

        return \
        TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                               scheduler,
                               positive, negative, latent_image, denoise)[0]


class PixelTiledKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                 denoise,
                 tile_width, tile_height, tiling_strategy,
                 upscale_model_opt=None, hook_opt=None, tile_size=512):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.vae = vae
        self.tile_params = tile_width, tile_height, tiling_strategy
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_size = tile_size
        self.is_tiled = True

    def tiled_ksample(self, latent):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'PixelTiledKSampleUpscalerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise Exception("'BNK_TiledKSampler' node isn't installed.")

        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
        tile_width, tile_height, tiling_strategy = self.tile_params

        return \
        TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                               scheduler,
                               positive, negative, latent, denoise)[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=True, save_temp_prefix=save_temp_prefix,
                                                            hook=self.hook,
                                                            tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model,
                                                                       upscale_factor, vae,
                                                                       use_tile=True,
                                                                       save_temp_prefix=save_temp_prefix,
                                                                       hook=self.hook,
                                                                       tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent)

        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=True, save_temp_prefix=save_temp_prefix,
                                                                  hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method,
                                                                             self.upscale_model, w, h, vae,
                                                                             use_tile=True,
                                                                             save_temp_prefix=save_temp_prefix,
                                                                             hook=self.hook,
                                                                             tile_size=self.tile_size)

        refined_latent = self.tiled_ksample(upscaled_latent)

        return refined_latent


# REQUIREMENTS: biegert/ComfyUI-CLIPSeg
class BBoxDetectorBasedOnCLIPSeg:
    prompt = None
    blur = None
    threshold = None
    dilation_factor = None
    aux = None

    def __init__(self, prompt, blur, threshold, dilation_factor):
        self.prompt = prompt
        self.blur = blur
        self.threshold = threshold
        self.dilation_factor = dilation_factor

    def detect(self, image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size=1, detailer_hook=None):
        mask = self.detect_combined(image, bbox_threshold, bbox_dilation)

        mask = make_2d_mask(mask)

        segs = mask_to_segs(mask, False, bbox_crop_factor, True, drop_size, detailer_hook=detailer_hook)
        return segs

    def detect_combined(self, image, bbox_threshold, bbox_dilation):
        if "CLIPSeg" in nodes.NODE_CLASS_MAPPINGS:
            CLIPSeg = nodes.NODE_CLASS_MAPPINGS['CLIPSeg']
        else:
            utils.try_install_custom_node('https://github.com/biegert/ComfyUI-CLIPSeg/raw/main/custom_nodes/clipseg.py',
                                          "To use 'CLIPSegDetectorProvider', 'CLIPSeg' extension is required.")
            raise Exception("'CLIPSeg' node isn't installed.")
        
        if self.threshold is None:
            threshold = bbox_threshold
        else:
            threshold = self.threshold

        if self.dilation_factor is None:
            dilation_factor = bbox_dilation
        else:
            dilation_factor = self.dilation_factor

        prompt = self.aux if self.prompt == '' and self.aux is not None else self.prompt

        mask, _, _ = CLIPSeg().segment_image(image, prompt, self.blur, threshold, dilation_factor)
        mask = to_binary_mask(mask)
        return mask

    def setAux(self, x):
        self.aux = x

        
def update_node_status(node, text, progress=None):
    if PromptServer.instance.client_id is None:
        return

    PromptServer.instance.send_sync("impact/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, PromptServer.instance.client_id)


from comfy.cli_args import args, LatentPreviewMethod
import folder_paths
from latent_preview import TAESD, TAESDPreviewerImpl, Latent2RGBPreviewer

try:
    import comfy.latent_formats as latent_formats


    def get_previewer(device, latent_format=latent_formats.SD15(), force=False, method=None):
        previewer = None

        if method is None:
            method = args.preview_method

        if method != LatentPreviewMethod.NoPreviews or force:
            # TODO previewer methods
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", latent_format.taesd_decoder_name)

            if method == LatentPreviewMethod.Auto:
                method = LatentPreviewMethod.Latent2RGB
                if taesd_decoder_path:
                    method = LatentPreviewMethod.TAESD

            if method == LatentPreviewMethod.TAESD:
                if taesd_decoder_path:
                    taesd = TAESD(None, taesd_decoder_path).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
                else:
                    print("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(
                        latent_format.taesd_decoder_name))

            if previewer is None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
        return previewer

except:
    print(f"#########################################################################")
    print(f"[ERROR] ComfyUI-Impact-Pack: Please update ComfyUI to the latest version.")
    print(f"#########################################################################")
