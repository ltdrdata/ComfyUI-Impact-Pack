import numpy as np
import torch


def ensure_nhwc_mask_torch(masks):
    """
    Ensure that the input masks are in the NHWC format, if not, switch to the NHWC format.
    """

    if masks is None or not isinstance(masks, torch.Tensor) or masks.ndim < 2:
        print(
            "[ERROR] The input masks are not in the expected format. The required types are torch.Tensor with at least two dimensions."
        )
        print(
            "   - If it's a list or one-dimensional array, please ensure it's been transformed into an array or tensor with at least two dimensions."
        )
        print("   - If the masks is null, ensure to provide non-empty input.")
        return None

    # [N, C, H, W] -> [N, H, W, C]
    if masks.ndim == 4:
        N, C, H, W = masks.shape
        if C in [1, 3] and H > 3 and W > 3:
            return masks.permute(0, 2, 3, 1)
        else:
            # Convert to NHWC format
            return masks.permute(0, 2, 3, 1)
    # [1, H, W] -> [1, H, W, 1]
    elif masks.ndim == 3 and masks.shape[0] == 1:
        return masks.unsqueeze(-1)
    # [H, W] -> [1, H, W, 1]
    elif masks.ndim == 2:
        return masks.unsqueeze(0).unsqueeze(-1)
    # [H, W, C] -> [1, H, W, C]
    elif masks.ndim == 3:
        H, W, C = masks.shape
        if C in [1, 3] and H > 3 and W > 3:
            # Masks are in the HWC format, need to add a batch dimension.
            return masks.unsqueeze(0)
        else:
            print(
                "[ERROR] The three-dimensional input tensor [H, W, C] is not in the correct shape. Please ensure that the C is between [1, 3], and H and W are representing the width and height of the pixel."
            )
            return None

    return None


def ensure_nhwc_mask_numpy(masks):
    """
    Transform the shape of the input masks into NHWC format (NumPy version).
    """

    if masks is None or not isinstance(masks, (np.ndarray)) or masks.ndim < 2:
        print(
            "[ERROR] The input masks are not in the expected format. The required types are np.ndarray with at least two dimensions."
        )
        print(
            "   - If it's a list or one-dimensional array, please ensure it's been transformed into an array with at least two dimensions."
        )
        print("   - If the masks is null, ensure to provide non-empty input.")
        return None

    # [N, C, H, W] -> [N, H, W, C]
    if masks.ndim == 4:
        N, H, W, C = masks.shape
        if C in [1, 3] and H > 3 and W > 3:
            return masks
        else:
            # convert to NHWC format
            return np.transpose(masks, (0, 2, 3, 1))
    # [1, H, W] -> [1, H, W, 1]
    elif masks.ndim == 3 and masks.shape[0] == 1:
        return np.expand_dims(masks, axis=-1)
    # [H, W] -> [1, H, W, 1]
    elif masks.ndim == 2:
        return np.expand_dims(np.expand_dims(masks, axis=0), axis=-1)
    # [H, W, C] -> [1, H, W, C]
    elif masks.ndim == 3:
        H, W, C = masks.shape
        if C in [1, 3] and H > 3 and W > 3:
            # masks is in the HWC format, need to add a batch dimension.
            return np.expand_dims(masks, axis=0)
        else:
            print(
                "[ERROR] The input array of three dimensions [H, W, C] is not in the correct shape. Please ensure that the C is between [1, 3], and H and W are representing the width and height of the pixel."
            )
            return None

    return None
