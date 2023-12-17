import torch
import torchvision
import cv2
import numpy as np
import folder_paths
import nodes
from . import config
from PIL import Image


def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
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


def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))


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
        return torch.from_numpy(np.array(image))
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
    """Mask and image2 has to be the same size"""
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)
    if image2.shape[1:3] != mask.shape[1:3]:
        raise ValueError(f"Inconsistent size: Image ({image2.shape[1:3]}) != Mask ({mask.shape[1:3]})")

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    # If the patch is out of bound, nothing to do!
    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]
    image1[:, y:y+h, x:x+w, :] = (
        (1 - mask) * image1[:, y:y+h, x:x+w, :] +
        mask * image2[:, :h, :w, :]
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
    mask = make_2d_mask(mask)

    mask = mask.clone().cpu()
    mask[mask > threshold] = 1.
    mask[mask <= threshold] = 0.
    return mask


def use_gpu_opencv():
    return not config.get_config()['disable_gpu_opencv']


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return mask

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


def tensor_feather_mask(mask, thickness, base_alpha=1.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if thickness <= 0:
        return mask

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    _tensor_check_mask(mask)
    feathered_mask = mask * base_alpha

    # Create a feathered mask by applying a Gaussian blur to the mask
    mask = mask[:, None, ..., 0]

    thickness = thickness * 2 - 1  # NOTE: GaussianBlur requires odd number for thickness

    blurred_mask = torchvision.transforms.GaussianBlur(thickness)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    tensor_paste(feathered_mask, blurred_mask, (0, 0), blurred_mask)
    return feathered_mask


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


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)


def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]
    pixels = nodes.VAEEncode.vae_encode_crop_pixels(pixels)
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask


class NonListIterable:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]


# author: Trung0246
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


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")
