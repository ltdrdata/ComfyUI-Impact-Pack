import os
import mmcv
from mmdet.apis import (inference_detector, init_detector)
from mmdet.evaluation import get_classes
from segment_anything import SamPredictor
import torch.nn.functional as F

from impact.utils import *
from collections import namedtuple
import numpy as np
from skimage.measure import label, regionprops

import nodes
import comfy_extras.nodes_upscale_model as model_upscale
from server import PromptServer

SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label'],
                 defaults=[None])


class NO_BBOX_DETECTOR:
    pass


class NO_SEGM_DETECTOR:
    pass


def load_mmdet(model_path):
    model_config = os.path.splitext(model_path)[0] + ".py"
    model = init_detector(model_config, model_path, device="cpu")
    return model


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def inference_segm_old(model, image, conf_threshold):
    image = image.numpy()[0] * 255
    mmdet_results = inference_detector(model, image)

    bbox_results, segm_results = mmdet_results
    label = "A"

    classes = get_classes("coco")
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n, m = bbox_results[0].shape
    if n == 0:
        return [[], [], []]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_idxs = np.where(bboxes[:, -1] > conf_threshold)[0]
    results = [[], [], []]
    for i in filter_idxs:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results


def inference_segm(image, modelname, conf_thres, lab="A"):
    image = image.numpy()[0] * 255
    mmdet_results = inference_detector(modelname, image).pred_instances
    bboxes = mmdet_results.bboxes.numpy()
    segms = mmdet_results.masks.numpy()
    scores = mmdet_results.scores.numpy()

    classes = get_classes("coco")

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]
    labels = mmdet_results.labels
    filter_inds = np.where(mmdet_results.scores > conf_thres)[0]
    results = [[], [], [], []]
    for i in filter_inds:
        results[0].append(lab + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(scores[i])

    return results


def inference_bbox(modelname, image, conf_threshold):
    image = image.numpy()[0] * 255
    label = "A"
    output = inference_detector(modelname, image).pred_instances
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in output.bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = output.bboxes.shape
    if n == 0:
        return [[], [], [], []]

    bboxes = output.bboxes.numpy()
    scores = output.scores.numpy()
    filter_idxs = np.where(scores > conf_threshold)[0]
    results = [[], [], [], []]
    for i in filter_idxs:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(scores[i])

    return results


def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    points = []
    plabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(mask.shape[0]/20))
    x_step = max(3, int(mask.shape[1]/20))
    
    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x+j, y+i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x+j, y+i))
                plabs.append(0)
    
    return points, plabs


def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(w/20))
    x_step = max(3, int(h/20))
    
    for i in range(10, h-10, y_step):
        for j in range(10, w-10, x_step):
            if not (x1-10 <= j and j <= x2+10 and y1-10 <= i and i <= y2+10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs


def enhance_detail(image, model, vae, guide_size, guide_size_for, bbox, seed, steps, cfg, sampler_name, scheduler,
                   positive, negative, denoise, noise_mask, force_inpaint):
    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if bbox_h >= guide_size and bbox_w >= guide_size:
        print(f"Detailer: segment skip")
        None

    if guide_size_for == "bbox":
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            print(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None

        if new_w == 0 or new_h == 0:
            print(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            print(f"Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale
    upscaled_image = scale_tensor(new_w, new_h, torch.from_numpy(image))

    # ksampler
    latent_image = to_latent_image(upscaled_image, vae)

    if noise_mask is not None:
        # upscale the mask tensor by a factor of 2 using bilinear interpolation
        noise_mask = torch.from_numpy(noise_mask)
        upscaled_mask = torch.nn.functional.interpolate(noise_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                                                        mode='bilinear', align_corners=False)

        # remove the extra dimensions added by unsqueeze
        upscaled_mask = upscaled_mask.squeeze().squeeze()
        latent_image['noise_mask'] = upscaled_mask

    refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]

    # non-latent downscale - latent downscale cause bad quality
    refined_image = vae.decode(refined_latent['samples'])

    # downscale
    refined_image = scale_tensor_and_to_pil(w, h, refined_image)

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image


def composite_to(dest_latent, crop_region, src_latent):
    x1 = crop_region[0]
    y1 = crop_region[1]

    # composite to original latent
    lc = nodes.LatentComposite()

    # 현재 mask 를 고려한 composite 가 없음... 이거 처리 필요.

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
    predictor = SamPredictor(sam_model)
    image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    print(f"image.shape: {image.shape}")

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

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
    else:
        mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")  # empty mask

    return mask


def segs_bitwise_and_mask(segs, mask):
    if mask is None:
        print("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return ([], )

    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label)
        items.append(item)

    return segs[0], items


class BBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
        drop_size = max(drop_size, 1)
        mmdet_results = inference_bbox(self.bbox_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x in segmasks:
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)
                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items

    def detect_combined(self, image, threshold, dilation):
        mmdet_results = inference_bbox(self.bbox_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass


class ONNXDetector(BBoxDetector):
    onnx_model = None

    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
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
                        crop_x1, crop_y1, crop_x2, crop_y2, = crop_region

                        # prepare cropped mask
                        cropped_mask = np.zeros((crop_y2-crop_y1,crop_x2-crop_x1))
                        inner_mask = np.ones((y2-y1, x2-x1))
                        cropped_mask[y1-crop_y1:y2-crop_y1, x1-crop_x1:x2-crop_x1] = inner_mask

                        # make items
                        item = SEG(None, cropped_mask, scores[i], crop_region, item_bbox)
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


class SegmDetector(BBoxDetector):
    segm_model = None

    def __init__(self, segm_model):
        self.segm_model = segm_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
        drop_size = max(drop_size, 1)
        mmdet_results = inference_segm(image, self.segm_model, threshold)
        segmasks = create_segmasks(mmdet_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]
        for x in segmasks:
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)
                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox)
                items.append(item)

        return image.shape, items

    def detect_combined(self, image, threshold, dilation):
        mmdet_results = inference_bbox(self.bbox_model, image, threshold)
        segmasks = create_segmasks(mmdet_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass


def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1):
    drop_size = max(drop_size, 1)
    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([], )

    mask = mask.cpu().numpy()

    result = []
    if combined == "True":
        # Find the indices of the non-zero elements
        indices = np.nonzero(mask)

        if len(indices[0]) > 0 and len(indices[1]) > 0:
            # Determine the bounding box of the non-zero elements
            bbox = np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0])
            crop_region = make_crop_region(mask.shape[1], mask.shape[0], bbox, crop_factor)
            x1, y1, x2, y2 = crop_region

            if x2 - x1 > 0 and y2 - y1 > 0:
                cropped_mask = mask[y1:y2, x1:x2]
                item = SEG(None, cropped_mask, 1.0, crop_region, bbox, 'A')
                result.append(item)

    else:
        # label the connected components
        labelled_mask = label(mask)

        # get the region properties for each connected component
        regions = regionprops(labelled_mask)

        # iterate over the regions and print their bounding boxes
        for region in regions:
            y1, x1, y2, x2 = region.bbox
            bbox = x1, y1, x2, y2
            crop_region = make_crop_region(mask.shape[1], mask.shape[0], bbox, crop_factor)

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                cropped_mask = np.array(mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]])

                if bbox_fill:
                    cropped_mask.fill(1.0)

                item = SEG(None, cropped_mask, 1.0, crop_region, bbox, 'A')

                result.append(item)

    if not result:
        print(f"[mask_to_segs] Empty mask.")

    print(f"# of Detected SEGS: {len(result)}")
    # for r in result:
    #     print(f"\tbbox={r.bbox}, crop={r.crop_region}, label={r.label}")

    return mask.shape, result


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


def vae_decode(vae, samples, use_tile, hook):
    if use_tile:
        pixels = nodes.VAEDecodeTiled().decode(vae, samples)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        hook.post_decode(pixels)

    return pixels


def vae_encode(vae, pixels, use_tile, hook):
    if use_tile:
        samples = nodes.VAEEncodeTiled().encode(vae, pixels)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        hook.post_encode(samples)

    return samples


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


class KSamplerAdvancedWrapper:
    params = None

    def __init__(self, model, cfg, sampler_name, scheduler, positive, negative):
        self.params = model, cfg, sampler_name, scheduler, positive, negative

    def sample_advanced(self, add_noise, seed, steps, latent_image, start_at_step, end_at_step, return_with_leftover_noise, hook=None):
        model, cfg, sampler_name, scheduler, positive, negative = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent = \
                hook.pre_ksample_advanced(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step,
                                          return_with_leftover_noise)

        return nodes.KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                               positive, negative, latent_image, start_at_step, end_at_step,
                                               return_with_leftover_noise)[0]


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

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
         return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


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

    def post_decode(self, pixels):
        return self.hook2.post_decode(self.hook1.post_decode(pixels))

    def post_upscale(self, pixels):
        return self.hook2.post_upscale(self.hook1.post_upscale(pixels))

    def post_encode(self, samples):
        return self.hook2.post_encode(self.hook1.post_encode(samples))

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
            self.hook1.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise)

        return self.hook2.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise)


class SimpleCfgScheduleHook(PixelKSampleHook):
    target_cfg = 0

    def __init__(self, target_cfg):
        super().__init__()
        self.target_cfg = target_cfg
    
    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        progress = self.cur_step/self.total_step
        gap = self.target_cfg - cfg
        current_cfg = cfg + gap*progress
        return model, seed, steps, current_cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class SimpleDenoiseScheduleHook(PixelKSampleHook):
    target_denoise = 0

    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        progress = self.cur_step / self.total_step
        gap = self.target_denoise - denoise
        current_denoise = denoise + gap * progress
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, current_denoise


def latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae, use_tile=False, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook)


def latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=False, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook)


def latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        pixels = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, pixels)[0]
        current_w = pixels.shape[2]

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook)


def latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False, save_temp_prefix=None, hook=None):
    pixels = vae_decode(vae, samples, use_tile, hook)

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

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook)


class TwoSamplersForMaskUpscaler:
    params = None
    upscale_model = None
    hook_base = None
    hook_mask = None
    hook_full = None
    use_tiled_vae = False
    is_tiled = False

    def __init__(self, scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
                 full_sampler_opt=None, upscale_model_opt=None, hook_base_opt=None, hook_mask_opt=None, hook_full_opt=None):
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        self.params = scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae
        self.upscale_model = upscale_model_opt
        self.full_sampler = full_sampler_opt
        self.hook_base = hook_base_opt
        self.hook_mask = hook_mask_opt
        self.hook_full = hook_full_opt
        self.use_tiled_vae = use_tiled_vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix, hook=self.hook_base)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model, upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix, hook=self.hook_mask)

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

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=self.use_tiled_vae,
                                                                  save_temp_prefix=save_temp_prefix, hook=self.hook_base)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model, w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix, hook=self.hook_mask)

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
            return cur_step >= total_step-1

        elif sample_schedule == "interleave1+last1":
            return cur_step % 2 == 0 or cur_step >= total_step-1

        elif sample_schedule == "interleave2+last1":
            return cur_step % 2 == 0 or cur_step >= total_step-1

        elif sample_schedule == "interleave3+last1":
            return cur_step % 2 == 0 or cur_step >= total_step-1

    def do_samples(self, step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent):
        if self.is_full_sample_time(step_info, sample_schedule):
            print(f"step_info={step_info} / full time")

            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)
            sampler = self.full_sampler if self.full_sampler is not None else base_sampler
            return sampler.sample(upscaled_latent, self.hook_full)

        else:
            print(f"step_info={step_info} / non-full time")
            # upscale mask
            upscaled_mask = F.interpolate(mask, size=(upscaled_latent['samples'].shape[2], upscaled_latent['samples'].shape[3]),
                                          mode='bilinear', align_corners=True)
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
    params = None
    upscale_model = None
    hook = None
    use_tiled_vae = False
    is_tiled = False

    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 use_tiled_vae, upscale_model_opt=None, hook_opt=None):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.use_tiled_vae = use_tiled_vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix, hook=self.hook)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model, upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix, hook=self.hook)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise)

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
                                                                  save_temp_prefix=save_temp_prefix, hook=self.hook)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model, w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix, hook=self.hook)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise)

        refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                 positive, negative, upscaled_latent, denoise)[0]
        return refined_latent


# REQUIREMENTS: BlenderNeko/ComfyUI_TiledKSampler
try:
    class TiledKSamplerWrapper:
        params = None

        def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                     tile_width, tile_height, tiling_strategy):
            self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy

        def sample(self, latent_image, hook=None):
            from custom_nodes.ComfyUI_TiledKSampler.nodes import TiledKSampler

            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy = self.params

            if hook is not None:
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                    hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

            return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, denoise)[0]

    class PixelTiledKSampleUpscaler:
        params = None
        upscale_model = None
        tile_params = None
        hook = None
        is_tiled = True

        def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, 
                     tile_width, tile_height, tiling_strategy,
                     upscale_model_opt=None, hook_opt=None):
            self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
            self.tile_params = tile_width, tile_height, tiling_strategy
            self.upscale_model = upscale_model_opt
            self.hook = hook_opt

        def tiled_ksample(self, latent):
            from custom_nodes.ComfyUI_TiledKSampler.nodes import TiledKSampler

            scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
            tile_width, tile_height, tiling_strategy = self.tile_params

            return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent, denoise)[0]

        def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
            scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

            if self.hook is not None:
                self.hook.set_steps(step_info)

            if self.upscale_model is None:
                upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                                use_tile=True, save_temp_prefix=save_temp_prefix, hook=self.hook)
            else:
                upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model, upscale_factor, vae,
                                                                           use_tile=True, save_temp_prefix=save_temp_prefix, hook=self.hook)

            refined_latent = self.tiled_ksample(upscaled_latent)

            return refined_latent

        def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
            scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

            if self.hook is not None:
                self.hook.set_steps(step_info)

            if self.upscale_model is None:
                upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                      use_tile=True, save_temp_prefix=save_temp_prefix, hook=self.hook)
            else:
                upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model, w, h, vae,
                                                                                 use_tile=True, save_temp_prefix=save_temp_prefix, hook=self.hook)

            refined_latent = self.tiled_ksample(upscaled_latent)

            return refined_latent
except:
    pass


# REQUIREMENTS: biegert/ComfyUI-CLIPSeg
try:
    class BBoxDetectorBasedOnCLIPSeg(BBoxDetector):
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

        def detect(self, image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size=1):
            mask = self.detect_combined(image, bbox_threshold, bbox_dilation)
            segs = mask_to_segs(mask, False, bbox_crop_factor, True, drop_size)
            return segs

        def detect_combined(self, image, bbox_threshold, bbox_dilation):
            from custom_nodes.clipseg import CLIPSeg

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
except:
    pass


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

    def get_previewer(device, latent_format=latent_formats.SD15(), force=False):
        previewer = None
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
                    print("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

            if previewer is None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
        return previewer

except:
    print(f"#########################################################################")
    print(f"[ERROR] ComfyUI-Impact-Pack: Please update ComfyUI to the latest version.")
    print(f"#########################################################################")

