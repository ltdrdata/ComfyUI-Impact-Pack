import torch
import torchvision
import cv2
import numpy as np
import folder_paths
import nodes
from . import config
from PIL import Image
import comfy
import time
import logging
import inspect


class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)


def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1), device=image.device, dtype=image.dtype)
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Similar error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Same error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")


def resize_with_padding(image, target_w: int, target_h: int):
    _tensor_check_image(image)
    b, h, w, c = image.shape
    image = image.permute(0, 3, 1, 2)  # B, C, H, W

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    image = image.permute(0, 2, 3, 1)  # B, H, W, C
    return image, (pad_top, pad_bottom, pad_left, pad_right)


def remove_padding(image, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    return image[:, pad_top:image.shape[1] - pad_bottom, pad_left:image.shape[2] - pad_right, :]


def shift_within_canvas(data, shift_x, shift_y, canvas_w=None, canvas_h=None, fill_value=0.0):
    is_torch = isinstance(data, torch.Tensor)
    data_device = data.device if is_torch else None
    data_dtype = data.dtype if is_torch else None
    np_data = data.detach().cpu().numpy() if is_torch else np.asarray(data)

    if np_data.ndim not in (2, 3, 4):
        raise ValueError(f"Unsupported ndim for shift_within_canvas: {np_data.ndim}")

    if np_data.ndim == 4:
        _, src_h, src_w, _ = np_data.shape
    elif np_data.ndim == 3:
        _, src_h, src_w = np_data.shape
    else:
        src_h, src_w = np_data.shape

    canvas_w = src_w if canvas_w is None else int(canvas_w)
    canvas_h = src_h if canvas_h is None else int(canvas_h)

    if np_data.ndim == 4:
        out_shape = (np_data.shape[0], canvas_h, canvas_w, np_data.shape[3])
    elif np_data.ndim == 3:
        out_shape = (np_data.shape[0], canvas_h, canvas_w)
    else:
        out_shape = (canvas_h, canvas_w)
    result = np.full(out_shape, fill_value, dtype=np_data.dtype)

    dst_x1 = max(0, int(shift_x))
    dst_y1 = max(0, int(shift_y))
    src_x1 = max(0, -int(shift_x))
    src_y1 = max(0, -int(shift_y))

    copy_w = min(src_w - src_x1, canvas_w - dst_x1)
    copy_h = min(src_h - src_y1, canvas_h - dst_y1)

    if copy_w > 0 and copy_h > 0:
        if np_data.ndim == 4:
            result[:, dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w, :] = np_data[:, src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w, :]
        elif np_data.ndim == 3:
            result[:, dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = np_data[:, src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]
        else:
            result[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = np_data[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

    if is_torch:
        result = torch.from_numpy(result).to(device=data_device, dtype=data_dtype)

    return result


def adjust_bbox_after_resize(bbox, original_size, target_size, padding):
    """
    bbox: (x1, y1, x2, y2) in original image
    original_size: (original_h, original_w)
    target_size: (target_h, target_w)
    padding: (pad_top, pad_bottom, pad_left, pad_right)
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    pad_top, pad_bottom, pad_left, pad_right = padding

    scale = min(target_w / orig_w, target_h / orig_h)

    # Apply scale
    x1 = int(bbox[0] * scale + pad_left)
    y1 = int(bbox[1] * scale + pad_top)
    x2 = int(bbox[2] * scale + pad_left)
    y2 = int(bbox[3] * scale + pad_top)

    return x1, y1, x2, y2


def general_tensor_resize(image, w: int, h: int, mode="bilinear"):
    _tensor_check_image(image)
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid resize target: {(w, h)}")

    cur_w, cur_h = tensor_get_size(image)
    if cur_w == w and cur_h == h:
        return image

    original_device = image.device
    original_dtype = image.dtype

    # Resize directly with torch instead of round-tripping through PIL.
    #
    # PIL.Image.resize can terminate the interpreter with SIGFPE/SIGSEGV in native
    # code for some resize inputs. That cannot be recovered with try/except, so the
    # detailer path must avoid PIL for tensor resizing entirely.
    nchw = image.movedim(-1, 1).contiguous().to(dtype=torch.float32)

    if mode in ("bilinear", "bicubic"):
        resized = torch.nn.functional.interpolate(nchw, size=(h, w), mode=mode, align_corners=False)
    elif mode in ("nearest", "area"):
        resized = torch.nn.functional.interpolate(nchw, size=(h, w), mode=mode)
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    resized = resized.movedim(1, -1).contiguous()

    if image.shape[-1] >= 3:
        resized = resized.clamp(0.0, 1.0)

    return resized.to(device=original_device, dtype=original_dtype)


# Kept for compatibility with callers/imports, but tensor_resize no longer uses
# PIL because a native PIL resize crash cannot be handled safely from Python.
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    mode = "bicubic" if image.shape[3] >= 3 else "bilinear"
    return general_tensor_resize(image, w, h, mode=mode)


def tensor_center_crop_or_pad(image, w: int, h: int):
    _tensor_check_image(image)
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid crop/pad target: {(w, h)}")

    _, cur_h, cur_w, channels = image.shape
    if cur_w == w and cur_h == h:
        return image

    src_x0 = max((cur_w - w) // 2, 0)
    src_y0 = max((cur_h - h) // 2, 0)
    dst_x0 = max((w - cur_w) // 2, 0)
    dst_y0 = max((h - cur_h) // 2, 0)
    copy_w = min(cur_w, w)
    copy_h = min(cur_h, h)

    out = torch.zeros((image.shape[0], h, w, channels), device=image.device, dtype=image.dtype)
    out[:, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w, :] = image[:, src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w, :]
    return out


def tensor_resize_for_detailer_output(image, w: int, h: int):
    _tensor_check_image(image)
    w = int(w)
    h = int(h)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid detailer resize target: {(w, h)}")

    cur_w, cur_h = tensor_get_size(image)
    if cur_w == w and cur_h == h:
        return image

    # VAEs often produce dimensions rounded to latent/tile multiples. When the
    # difference is only a small padding margin, do not run interpolation at all:
    # crop/pad the tensor directly. This avoids a native/PyTorch resize path at
    # the most fragile post-decode boundary.
    if abs(cur_w - w) <= 16 and abs(cur_h - h) <= 16:
        logging.info(f"Detailer: correcting post-decode padding by crop/pad {(cur_w, cur_h)} -> {(w, h)}")
        return tensor_center_crop_or_pad(image, w, h)

    # For final paste-back geometry, prefer area for downscale and bilinear for
    # upscale. Avoid bicubic here: this path has already been observed to crash
    # or wedge in native resize code after VAE decode.
    mode = "area" if w <= cur_w and h <= cur_h else "bilinear"
    logging.info(f"Detailer: post-decode resize {(cur_w, cur_h)} -> {(w, h)} using torch {mode} on {image.device}")
    return general_tensor_resize(image, w, h, mode=mode)


def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(0), 0, 255).astype(np.uint8))


def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")


def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")


def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, torch.Tensor):
        return image.numpy()
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Cannot convert {type(image)} to numpy.ndarray")

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return


def tensor_crop(image, crop_region):
    _tensor_check_image(image)
    return crop_ndarray4(image, crop_region)


def tensor2numpy(image):
    _tensor_check_image(image)
    return image.numpy()


def tensor_paste(image1, image2, left_top, mask):
    """
    Pastes image2 onto image1 at position left_top using mask.
    Supports both RGB and RGBA images.

    Large paste regions are processed in horizontal chunks to avoid allocating
    full-frame blend temporaries for FaceDetailer crops that cover most of the
    image.
    """
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)

    x, y = left_top
    _, h1, w1, c1 = image1.shape
    _, h2, w2, c2 = image2.shape

    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]

    pixels = int(w) * int(h)
    if pixels >= 4_000_000:
        rows_per_chunk = max(64, min(512, 8_000_000 // max(int(w), 1)))
        logging.info(f"Detailer: tensor_paste chunked region {(w, h)} rows_per_chunk={rows_per_chunk}")
    else:
        rows_per_chunk = h

    for y0 in range(0, h, rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, h)
        dst_y0 = y + y0
        dst_y1 = y + y1

        mask_chunk = mask[:, y0:y1, :, :]
        region1 = image1[:, dst_y0:dst_y1, x:x+w, :]
        region2 = image2[:, y0:y1, :w, :]

        if c1 == 3 and c2 == 3:
            image1[:, dst_y0:dst_y1, x:x+w, :] = (1 - mask_chunk) * region1 + mask_chunk * region2

        elif c1 == 4 and c2 == 4:
            image1[:, dst_y0:dst_y1, x:x+w, :3] = (
                (1 - mask_chunk) * region1[:, :, :, :3] +
                mask_chunk * region2[:, :, :, :3]
            )
            a1 = region1[:, :, :, 3:4]
            a2 = region2[:, :, :, 3:4] * mask_chunk
            image1[:, dst_y0:dst_y1, x:x+w, 3:4] = a1 + a2 * (1 - a1)

        elif c1 == 4 and c2 == 3:
            image1[:, dst_y0:dst_y1, x:x+w, :3] = (
                (1 - mask_chunk) * region1[:, :, :, :3] +
                mask_chunk * region2
            )
            image1[:, dst_y0:dst_y1, x:x+w, 3:4] = region1[:, :, :, 3:4] * (1 - mask_chunk) + mask_chunk

        elif c1 == 3 and c2 == 4:
            effective_mask = mask_chunk * region2[:, :, :, 3:4]
            image1[:, dst_y0:dst_y1, x:x+w, :] = (
                (1 - effective_mask) * region1 +
                effective_mask * region2[:, :, :, :3]
            )

    return

def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w/2, bbox[1] + h/2


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def combine_masks2(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i]).astype(np.uint8)

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def bitwise_and_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
        return torch.from_numpy(cv2_mask)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def to_binary_mask(mask, threshold=0):
    mask = make_3d_mask(mask)

    mask = mask.clone().cpu()
    mask[mask > threshold] = 1.
    mask[mask <= threshold] = 0.
    return mask


def use_gpu_opencv():
    return not config.get_config()['disable_gpu_opencv']


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return make_2d_mask(mask)

    mask = make_2d_mask(mask)

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        mask = cv2.UMat(mask)
        kernel = cv2.UMat(kernel)

    if dilation_factor > 0:
        result = cv2.dilate(mask, kernel, iter)
    else:
        result = cv2.erode(mask, kernel, iter)

    if use_gpu_opencv():
        return result.get()
    else:
        return result


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        kernel = cv2.UMat(kernel)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if use_gpu_opencv():
            cv2_mask = cv2.UMat(cv2_mask)

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        if use_gpu_opencv():
            dilated_mask = dilated_mask.get()

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks

import torch.nn.functional as F
def feather_mask(mask, thickness):
    mask = mask.permute(0, 3, 1, 2)

    # Gaussian kernel for blurring
    kernel_size = 2 * int(thickness) + 1
    sigma = thickness / 3  # Adjust the sigma value as needed
    blur_kernel = _gaussian_kernel(kernel_size, sigma).to(mask.device, mask.dtype)

    # Apply blur to the mask
    blurred_mask = F.conv2d(mask, blur_kernel.unsqueeze(0).unsqueeze(0), padding=thickness)

    blurred_mask = blurred_mask.permute(0, 2, 3, 1)

    return blurred_mask

def _gaussian_kernel(kernel_size, sigma):
    # Generate a 1D Gaussian kernel
    kernel = torch.exp(-(torch.arange(kernel_size) - kernel_size // 2)**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size*2+1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest/2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask  # skip feathering

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask


def subtract_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
        return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def add_masks(mask1, mask2):
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()
    cv2_mask1 = np.array(mask1) * 255
    cv2_mask2 = np.array(mask2) * 255

    if cv2_mask1.shape == cv2_mask2.shape:
        cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
        return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
    else:
        # do nothing - incompatible mask shape: mostly empty mask
        return mask1


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


crop_tensor4 = crop_ndarray4


def crop_ndarray3(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2]

    return cropped


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)


def to_latent_image(pixels, vae, vae_tiled_encode=False, auto_vae_tiled_encode=False, tile_size=0, overlap=0):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]

    start = time.time()
    tile_size, overlap = get_vae_tiled_encode_settings(pixels, tile_size=tile_size, overlap=overlap)
    force_low_memory_tiling = tile_size < 512
    should_tile = vae_tiled_encode or auto_vae_tiled_encode or force_low_memory_tiling or tile_size > 0 or overlap > 0

    if should_tile:
        encoder = nodes.VAEEncodeTiled()
        try:
            supports_overlap = 'overlap' in inspect.signature(encoder.encode).parameters
        except (TypeError, ValueError):
            supports_overlap = False

        if supports_overlap:
            encoded = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            encoded = encoder.encode(vae, pixels, tile_size)[0]
        logging.info(f"[Impact Pack] vae encoded (tiled {tile_size}/{overlap}) in {time.time() - start:.1f}s")
    else:
        encoded = nodes.VAEEncode().encode(vae, pixels)[0]
        logging.info(f"[Impact Pack] vae encoded in {time.time() - start:.1f}s")

    return encoded


def get_vae_tiled_encode_settings(pixels, tile_size=0, overlap=0):
    h = int(pixels.shape[1])
    w = int(pixels.shape[2])
    megapixels = (h * w) / 1_000_000.0

    # FaceDetailer encode should stay on the tiled path once selected; the tile
    # geometry adapts by crop size, but 512/64 is still the tiled path.
    resolved_tile_size = int(tile_size) if tile_size is not None else 0
    if resolved_tile_size <= 0:
        if megapixels >= 3.0:
            resolved_tile_size = 128
        elif megapixels >= 1.5:
            resolved_tile_size = 256
        else:
            resolved_tile_size = 512

    resolved_overlap = int(overlap) if overlap is not None else 0
    if resolved_overlap <= 0:
        resolved_overlap = max(16, resolved_tile_size // 8)

    resolved_overlap = min(max(0, resolved_overlap), max(0, resolved_tile_size - 1))
    return resolved_tile_size, resolved_overlap


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask


def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask


def make_4d_mask(mask):
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)

    return mask


def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index


def collect_non_reroute_nodes(node_map, links, res, node_id):
    if node_map[node_id]['type'] != 'Reroute' and node_map[node_id]['type'] != 'Reroute (rgthree)':
        res.append(node_id)
    else:
        for link in node_map[node_id]['outputs'][0]['links']:
            next_node_id = str(links[link][2])
            collect_non_reroute_nodes(node_map, links, res, next_node_id)


from torchvision.transforms.functional import to_pil_image


def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized_mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    return resized_mask.squeeze(0)


def apply_mask_alpha_to_pil(decoded_pil, mask):
    decoded_rgba = decoded_pil.convert('RGBA')
    mask_pil = to_pil_image(mask)
    decoded_rgba.putalpha(mask_pil)

    return decoded_rgba


def flatten_mask(all_masks):
    merged_mask = (all_masks[0] * 255).to(torch.uint8)
    for mask in all_masks[1:]:
        merged_mask |= (mask * 255).to(torch.uint8)

    return merged_mask


def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Impact Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception:
        logging.info(msg)
        logging.info("[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")


def apply_differential_diffusion(model):
    # ComfyUI ≥0.3.63 exposes V3 schema (classmethod `execute`); older versions use instance method `apply`.
    # Import is deferred so callers with guarded imports (e.g. segs_upscaler.py) still work when the
    # comfy_extras module is absent on very old ComfyUI — the ImportError propagates as before.
    from comfy_extras import nodes_differential_diffusion
    dd = nodes_differential_diffusion.DifferentialDiffusion()
    if hasattr(dd, 'execute'):
        return dd.execute(model)[0]
    if hasattr(dd, 'apply'):
        return dd.apply(model)[0]
    raise AttributeError(
        "DifferentialDiffusion has neither 'execute' nor 'apply'. "
        "Update ComfyUI (≥0.3.63 for V3) or reinstall Impact Pack."
    )


# author: Trung0246 --->
class TautologyStr(str):
    def __ne__(self, other):
        return False


class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


class NonListIterable:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]


def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in folder_paths.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)
# <---

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")