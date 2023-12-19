import os
from PIL import ImageOps
from impact.utils import *

from . import core
import random

class PreviewBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "image": ("STRING", {"default": ""}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("IMAGE", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prev_hash = None

    @staticmethod
    def load_image(pb_id):
        is_fail = False
        if pb_id not in core.preview_bridge_image_id_map:
            is_fail = True

        image_path, ui_item = core.preview_bridge_image_id_map[pb_id]

        if not os.path.isfile(image_path):
            is_fail = True

        if not is_fail:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        if is_fail:
            image = empty_pil_tensor()
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            ui_item = {
                "filename": 'empty.png',
                "subfolder": '',
                "type": 'temp'
            }

        return (image, mask.unsqueeze(0), ui_item)

    def doit(self, images, image, unique_id):
        need_refresh = False

        if unique_id not in core.preview_bridge_cache:
            need_refresh = True

        elif core.preview_bridge_cache[unique_id][0] is not images:
            need_refresh = True

        if not need_refresh:
            pixels, mask, path_item = PreviewBridge.load_image(image)
            image = [path_item]
        else:
            res = nodes.PreviewImage().save_images(images, filename_prefix="PreviewBridge/PB-")
            image2 = res['ui']['images']
            pixels = images
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', image2[0]['filename'])
            core.set_previewbridge_image(unique_id, path, image2[0])
            core.preview_bridge_image_id_map[image] = (path, image2[0])
            core.preview_bridge_image_name_map[unique_id, path] = (image, image2[0])
            core.preview_bridge_cache[unique_id] = (images, image2)

            image = image2

        return {
            "ui": {"images": image},
            "result": (pixels, mask, ),
        }


def decode_latent(latent_tensor, preview_method):
    from comfy.cli_args import LatentPreviewMethod
    import comfy.latent_formats as latent_formats

    if preview_method == "Latent2RGB-SD15":
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "TAESD15":
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.TAESD
    elif preview_method == "TAESDXL":
        latent_format = latent_formats.SDXL()
        method = LatentPreviewMethod.TAESD
    else:  # preview_method == "Latent2RGB-SDXL"
        latent_format = latent_formats.SDXL()
        method = LatentPreviewMethod.Latent2RGB

    previewer = core.get_previewer("cpu", latent_format=latent_format, force=True, method=method)

    decoded_pil = previewer.decode_latent_to_preview(latent_tensor)
    return decoded_pil


class PreviewBridgeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "image": ("STRING", {"default": ""}),
                    "preview_method": (["Latent2RGB-SDXL", "Latent2RGB-SD15", "TAESDXL", "TAESD15"],),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prev_hash = None
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @staticmethod
    def load_image(pb_id):
        is_fail = False
        if pb_id not in core.preview_bridge_image_id_map:
            is_fail = True

        image_path, ui_item = core.preview_bridge_image_id_map[pb_id]

        if not os.path.isfile(image_path):
            is_fail = True

        if not is_fail:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = None

        if is_fail:
            image = empty_pil_tensor()
            mask = None
            ui_item = {
                "filename": 'empty.png',
                "subfolder": '',
                "type": 'temp'
            }

        return (image, mask, ui_item)

    def doit(self, latent, image, preview_method, unique_id):
        need_refresh = False

        if unique_id not in core.preview_bridge_cache:
            need_refresh = True

        elif core.preview_bridge_cache[unique_id][0] is not latent or core.preview_bridge_cache[unique_id][1] != preview_method:
            need_refresh = True

        if not need_refresh:
            pixels, mask, path_item = PreviewBridge.load_image(image)

            if mask is None:
                mask = torch.ones(latent['samples'].shape[2:], dtype=torch.float32, device="cpu").unsqueeze(0)
                if 'noise_mask' in latent:
                    res_latent = latent.copy()
                    del res_latent['noise_mask']
                else:
                    res_latent = latent
            else:
                res_latent = latent.copy()
                res_latent['noise_mask'] = mask

            res_image = [path_item]
        else:
            decoded_pil = decode_latent(latent['samples'], preview_method)

            if 'noise_mask' in latent:
                mask = latent['noise_mask']

                inverted_mask = 1 - mask  # invert
                resized_mask = resize_mask(inverted_mask, decoded_pil.size)
                result_pil = apply_mask_alpha_to_pil(decoded_pil, resized_mask)

                full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path("PreviewBridge/PBL-"+self.prefix_append, folder_paths.get_temp_directory(), result_pil.size[0], result_pil.size[1])
                file = f"{filename}_{counter}.png"
                result_pil.save(os.path.join(full_output_folder, file), compress_level=4)
                res_image = [{
                                'filename': file,
                                'subfolder': 'PreviewBridge',
                                'type': 'temp',
                            }]
            else:
                mask = torch.ones(latent['samples'].shape[2:], dtype=torch.float32, device="cpu").unsqueeze(0)
                decoded_image = to_tensor(decoded_pil).unsqueeze(0)
                res = nodes.PreviewImage().save_images(decoded_image, filename_prefix="PreviewBridge/PBL-")
                res_image = res['ui']['images']

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', res_image[0]['filename'])
            core.set_previewbridge_image(unique_id, path, res_image[0])
            core.preview_bridge_image_id_map[image] = (path, res_image[0])
            core.preview_bridge_image_name_map[unique_id, path] = (image, res_image[0])
            core.preview_bridge_cache[unique_id] = (latent, preview_method, res_image)

            res_latent = latent

        return {
            "ui": {"images": res_image},
            "result": (res_latent, mask, ),
        }
