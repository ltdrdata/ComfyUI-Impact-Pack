import os
from PIL import ImageOps
import logging
import folder_paths
import torch
import nodes
from PIL import Image
import numpy as np
from impact import utils

# NOTE: this should not be `from . import core`.
# I don't know why but... 'from .' and 'from impact' refer to different core modules.
# This separates global variables of the core module and breaks the preview bridge.
from impact import core
# <--
import random


def over_checkerboard(images):
    """RGBA images as RGB, with their transparency drawn over a checkerboard."""
    height, width = images.shape[1:3]
    rows = torch.arange(height, device=images.device).unsqueeze(1) // 16
    columns = torch.arange(width, device=images.device).unsqueeze(0) // 16
    board = torch.where((rows + columns) % 2 == 0, 1.0, 0.8).unsqueeze(-1).to(images)
    alpha = images[..., 3:]
    return images[..., :3] * alpha + board * (1 - alpha)


def shown_frame(pixels, shown):
    """Which image of a batch a file of the mask editor was made from: the one it differs least from as shown."""
    if shown.shape[0] == 1:
        return 0
    return int(((pixels - shown.to(pixels)).abs().amax(dim=-1) > 1.5 / 255).flatten(1).sum(dim=1).argmin())


class PreviewBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "image": ("STRING", {"default": ""}),
                    },
                "optional": {
                    "block": ("BOOLEAN", {"default": False, "label_on": "if_empty_mask", "label_off": "never", "tooltip": "if_empty_mask: If the mask is empty, the execution is stopped.\nnever: The execution is never stopped."}),
                    "restore_mask": (["never", "always", "if_same_size"], {"tooltip": "if_same_size: If the changed input image is the same size as the previous image, restore using the last saved mask\nalways: Whenever the input image changes, always restore using the last saved mask\nnever: Do not restore the mask.\n`restore_mask` has higher priority than `block`"}),
                    "alpha_as_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "How the alpha channel of an RGBA input image is read. An RGB input is not affected.\ndisabled: The alpha is part of the image. It is passed on unchanged and is never used as the mask. The preview and the mask editor show the transparency over a checkerboard.\nenabled: The alpha is the mask the image brings. It becomes the mask to edit, `restore_mask` is not applied to it, and the image output carries the current mask as its alpha."}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("IMAGE", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    DESCRIPTION = "This is a feature that allows you to edit and send a Mask over a image.\nIf the block is set to 'if_empty_mask', the execution is stopped when the mask is empty."

    @staticmethod
    def load_image(pb_id):
        registered = core.preview_bridge_image_id_map.get(pb_id)
        if registered is None or not os.path.isfile(registered[0]):
            return utils.empty_pil_tensor(), None, {
                "filename": 'empty.png', "subfolder": '', "type": 'temp',
            }

        image_path, ui_item = registered
        with Image.open(image_path) as source:
            i = ImageOps.exif_transpose(source)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = (1. - torch.from_numpy(mask)).unsqueeze(0)
            else:
                mask = None
        return image, mask, ui_item

    @staticmethod
    def paint_layer(image_path):
        """What the mask editor's paint tools drew. The editor saves it beside the file it hands back."""
        folder, name = os.path.split(image_path)
        path = os.path.join(folder, name.replace("clipspace-painted-masked-", "clipspace-paint-", 1))
        if not name.startswith("clipspace-painted-masked-") or not os.path.isfile(path):
            return None
        with Image.open(path) as layer:
            paint = torch.from_numpy(np.array(layer.convert("RGBA")).astype(np.float32) / 255.0)[None,]
        return paint if paint[..., 3].any() else None

    @staticmethod
    def register_clipspace_image(clipspace_path, node_id):
        clean_path = clipspace_path.removesuffix(" [input]").removesuffix("[input]")
        input_dir = os.path.abspath(folder_paths.get_input_directory())
        candidates = [
            os.path.join(input_dir, clean_path),
            os.path.join(input_dir, "clipspace", os.path.basename(clean_path)),
        ]
        for candidate in candidates:
            try:
                # Match /view: allow configured symlinks but reject path traversal.
                actual_file = os.path.abspath(candidate)
                if os.path.commonpath((input_dir, actual_file)) != input_dir:
                    return False
            except ValueError:
                # Paths on different Windows drives cannot be inside input_dir.
                return False
            if os.path.isfile(actual_file):
                break
        else:
            return False

        relative_path = os.path.relpath(actual_file, input_dir)
        ui_item = {
            'filename': os.path.basename(relative_path),
            'subfolder': os.path.dirname(relative_path),
            'type': 'input'
        }
        core.set_previewbridge_image(node_id, actual_file, ui_item)
        # Also register under the original clipspace path for compatibility
        core.preview_bridge_image_id_map[clipspace_path] = (actual_file, ui_item)
        return True

    @staticmethod
    def alpha_preview(unique_id, alpha_as_mask, index):
        """One image of the RGBA input a node ran on last, as the node shows it once alpha_as_mask has this value.

        The frontend shows it as soon as the option is switched, before the bridge runs again. There is none where
        the next run leaves the file the node shows as it is."""
        cached = core.preview_bridge_cache.get(unique_id)
        if cached is None or not isinstance(cached[0], torch.Tensor) or cached[0].shape[-1] != 4 or not 0 <= index < cached[0].shape[0]:
            return None
        images = cached[0]
        if not (images[..., 3] < 1).any():
            return None
        if alpha_as_mask:
            return images[index:index + 1]
        held = core.preview_bridge_image_id_map.get(cached[2])
        if cached[3] and held is not None and os.path.abspath(held[0]) not in cached[4]:
            # A file saved in the mask editor while the option was on. One of another size, or with paint, stays as
            # it is; else its mask is kept, on the image it was saved for.
            pixels, mask, _ = PreviewBridge.load_image(cached[2])
            paint = PreviewBridge.paint_layer(held[0])
            if pixels.shape[1:3] != images.shape[1:3] or (paint is not None and paint.shape[1:3] == images.shape[1:3]):
                return None
            shown = over_checkerboard(images)
            index = shown_frame(pixels, shown)
            picture = shown[index:index + 1].to(pixels)
            return picture if mask is None else torch.cat((picture, (1 - mask + 0.5 / 255).unsqueeze(-1)), dim=-1)
        return over_checkerboard(images[index:index + 1])

    def doit(self, images, image, unique_id, block=False, restore_mask="never", prompt=None, extra_pnginfo=None, alpha_as_mask=False):
        need_refresh = False
        cached = core.preview_bridge_cache.get(unique_id)
        # A different workflow can reuse this ID for the other bridge type.
        if cached is not None and not isinstance(cached[0], torch.Tensor):
            core.preview_bridge_cache.pop(unique_id)
            core.preview_bridge_last_mask_cache.pop(unique_id, None)
            cached = None

        # The preview file has one alpha channel and the mask editor reads it as the mask, so an input's own
        # alpha is either that mask (alpha_as_mask) or drawn into the file over a checkerboard and passed on
        # from the input.
        has_alpha = images.shape[-1] == 4
        incoming = None
        if has_alpha and alpha_as_mask:
            incoming = 1 - images[..., 3].to(device="cpu", dtype=torch.float32)
            if not incoming.any():
                incoming = None

        # Check if images have changed (this determines if we start fresh)
        if cached is None:
            need_refresh = True
        elif cached[0] is not images and (
            images.shape[0] != 1
            or cached[0].device != images.device
            or not torch.equal(cached[0], images)
        ):
            need_refresh = True
        elif incoming is not None and not cached[3]:
            # Switched on: the alpha of this image is read now.
            need_refresh = True

        if need_refresh and restore_mask == "never":
            core.preview_bridge_last_mask_cache.pop(unique_id, None)

        # Keep submitted registrations immutable; reuse this node's refreshed pixels separately.
        preview_id = cached[2] if not need_refresh and image == cached[1] else image
        # API prompts can submit a raw saved file without frontend registration.
        if preview_id not in core.preview_bridge_image_id_map and (not need_refresh or restore_mask != "never"):
            is_clipspace = image and ("clipspace" in image.lower() or "[input]" in image)
            registered = is_clipspace and PreviewBridge.register_clipspace_image(image, unique_id)
            need_refresh = need_refresh or not registered

        # A mask that is still the alpha an input brought is dropped once the alpha is no longer read as the mask.
        # An edited mask is saved to another file and is kept.
        held = core.preview_bridge_image_id_map.get(cached[2] if cached is not None and image == cached[1] else image)
        from_alpha = cached is not None and held is not None and os.path.abspath(held[0]) in cached[4]
        stale_alpha = from_alpha and not alpha_as_mask
        if stale_alpha:
            core.preview_bridge_last_mask_cache.pop(unique_id, None)
            need_refresh = True

        alpha_files = ()
        kept_alpha = False
        # Disabled, the file shows the picture to the mask editor over a checkerboard, every image of a batch.
        shown = over_checkerboard(images) if has_alpha and not alpha_as_mask else images[..., :3]
        if not need_refresh:
            preview = None
            pixels, mask, path_item = PreviewBridge.load_image(preview_id)
            res_image = [path_item]
            if has_alpha and not alpha_as_mask and pixels.shape[1:3] == images.shape[1:3]:
                # The file is one image, the one of the batch that the mask editor was given. Only its mask is
                # read: the image is the input, with what the editor painted laid over it.
                index = shown_frame(pixels, shown)
                paint = PreviewBridge.paint_layer(core.preview_bridge_image_id_map[preview_id][0])
                pixels = images[index:index + 1].to(pixels)
                alpha = pixels[..., 3:]
                if paint is not None and paint.shape == pixels.shape:
                    cover = paint[..., 3:]
                    opacity = cover + alpha * (1 - cover)
                    colour = (paint[..., :3] * cover + pixels[..., :3] * alpha * (1 - cover)) / opacity.clamp(min=1e-6)
                    pixels = torch.where(cover > 0, torch.cat((colour, opacity), dim=-1), pixels)
                elif cached[3] and (alpha < 1).any():
                    # Switched off: the file is written again in the form the editor gets while the option is
                    # off. A mask saved in the editor is kept as saved, with the transparent area it was opened
                    # with, and the frontend warns about it. Half a level keeps the 8-bit levels the mask had
                    # when it is saved again.
                    preview = shown[index:index + 1].to(pixels)
                    if mask is not None:
                        preview = torch.cat((preview, (1 - mask + 0.5 / 255).unsqueeze(-1)), dim=-1)
                if cached[3] and mask is not None:
                    kept_alpha = bool((mask.unsqueeze(-1) * (1 - alpha)).any())
            elif has_alpha and pixels.shape[1:3] == images.shape[1:3]:
                alpha = torch.ones_like(pixels[..., :1]) if mask is None else (1 - mask).unsqueeze(-1)
                pixels = torch.cat((pixels, alpha), dim=-1)
            if mask is None:
                mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            if from_alpha:
                alpha_files = cached[4]
        else:
            if incoming is not None:
                # As with a latent that brings a noise mask: the incoming mask is used and nothing is restored over it.
                # The mask is one mask: of a batch, that of the first image, which the node shows.
                mask = incoming[:1]
            elif restore_mask != "never" and not stale_alpha:
                # Registrations for later queued edits may already have updated the node cache.
                mask = core.get_previewbridge_mask(image)
                if mask is None:
                    mask = core.preview_bridge_last_mask_cache.get(unique_id)
                if mask is not None and restore_mask == "if_same_size" and mask.shape[1:] != images.shape[1:3]:
                    mask = None
            else:
                mask = None

            pixels = images
            if incoming is not None:
                preview = images
            elif mask is None:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                preview = shown
            else:
                preview = utils.tensor_convert_rgba(shown)
                resized_mask = utils.resize_mask(mask, (images.shape[1], images.shape[2])).unsqueeze(3)
                resized_mask = 1 - resized_mask
                utils.tensor_putalpha(preview, resized_mask)
                if has_alpha and alpha_as_mask:
                    pixels = preview

        if preview is not None:
            res_image = nodes.PreviewImage().save_images(preview, filename_prefix="PreviewBridge/PB-", prompt=prompt, extra_pnginfo=extra_pnginfo)['ui']['images']
            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', res_image[0]['filename'])
            preview_id = core.set_previewbridge_image(unique_id, path, res_image[0])
            if incoming is not None:
                # The form in which the frontend registers the file it displays, which is any image of a batch.
                alpha_files = [os.path.abspath(os.path.join(os.path.dirname(path), item['filename'])) for item in res_image]

        core.preview_bridge_cache[unique_id] = (images, image, preview_id, alpha_as_mask, alpha_files)
        is_empty_mask = torch.all(mask == 0)

        if block and is_empty_mask and core.is_execution_model_version_supported():
            from comfy_execution.graph import ExecutionBlocker
            result = ExecutionBlocker(None), ExecutionBlocker(None)
        elif block and is_empty_mask:
            logging.warning("[Impact Pack] PreviewBridge: ComfyUI is outdated - blocking feature is disabled.")
            result = pixels, mask
        else:
            result = pixels, mask

        if not is_empty_mask:
            core.preview_bridge_last_mask_cache[unique_id] = mask

        return {
            "ui": {"images": res_image, "source_image": [image], "alpha_in_mask": [kept_alpha], "alpha_as_mask": [alpha_as_mask]},
            "result": result,
        }


def decode_latent(latent, preview_method, vae_opt=None):
    if vae_opt is not None:
        image = nodes.VAEDecode().decode(vae_opt, latent)[0]
        return image

    from comfy.cli_args import LatentPreviewMethod
    import comfy.latent_formats as latent_formats

    if preview_method.startswith("TAE"):
        decoder_name = None

        if preview_method == "TAESD15":
            decoder_name = "taesd"
        elif preview_method == 'TAESDXL':
            decoder_name = "taesdxl"
        elif preview_method == 'TAESD3':
            decoder_name = "taesd3"
        elif preview_method == 'TAEF1':
            decoder_name = "taef1"

        if decoder_name:
            vae = nodes.VAELoader().load_vae(decoder_name)[0]
            image = nodes.VAEDecode().decode(vae, latent)[0]
            return image

    if preview_method == "Latent2RGB-SD15":
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-SDXL":
        latent_format = latent_formats.SDXL()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-SD3":
        latent_format = latent_formats.SD3()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-SD-X4":
        latent_format = latent_formats.SD_X4()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-Playground-2.5":
        latent_format = latent_formats.SDXL_Playground_2_5()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-SC-Prior":
        latent_format = latent_formats.SC_Prior()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-SC-B":
        latent_format = latent_formats.SC_B()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-FLUX.1":
        latent_format = latent_formats.Flux()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "Latent2RGB-LTXV":
        latent_format = latent_formats.LTXV()
        method = LatentPreviewMethod.Latent2RGB
    else:
        logging.warning(f"[Impact Pack] PreviewBridgeLatent: '{preview_method}' is unsupported preview method.")
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.Latent2RGB

    previewer = core.get_previewer("cpu", latent_format=latent_format, force=True, method=method)
    samples = latent_format.process_in(latent['samples'])

    pil_image = previewer.decode_latent_to_preview(samples)
    pixels_size = pil_image.size[0]*8, pil_image.size[1]*8
    resized_image = pil_image.resize(pixels_size, resample=utils.LANCZOS)

    return utils.to_tensor(resized_image).unsqueeze(0)


class PreviewBridgeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "image": ("STRING", {"default": ""}),
                    "preview_method": (["Latent2RGB-FLUX.1",
                                        "Latent2RGB-SDXL", "Latent2RGB-SD15", "Latent2RGB-SD3",
                                        "Latent2RGB-SD-X4", "Latent2RGB-Playground-2.5",
                                        "Latent2RGB-SC-Prior", "Latent2RGB-SC-B",
                                        "Latent2RGB-LTXV",
                                        "TAEF1", "TAESDXL", "TAESD15", "TAESD3"],),
                    },
                "optional": {
                    "vae_opt": ("VAE", ),
                    "block": ("BOOLEAN", {"default": False, "label_on": "if_empty_mask", "label_off": "never", "tooltip": "if_empty_mask: If the mask is empty, the execution is stopped.\nnever: The execution is never stopped. Instead, it returns a white mask."}),
                    "restore_mask": (["never", "always", "if_same_size"], {"tooltip": "if_same_size: If the changed input latent is the same size as the previous latent, restore using the last saved mask\nalways: Whenever the input latent changes, always restore using the last saved mask\nnever: Do not restore the mask.\n`restore_mask` has higher priority than `block`\nIf the input latent already has a mask, do not restore mask."}),
                },
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("LATENT", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    DESCRIPTION = "This is a feature that allows you to edit and send a Mask over a latent image.\nIf the block is set to 'if_empty_mask', the execution is stopped when the mask is empty."

    def __init__(self):
        super().__init__()
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    def doit(self, latent, image, preview_method, vae_opt=None, block=False, unique_id=None, restore_mask='never', prompt=None, extra_pnginfo=None):
        latent_channels = latent['samples'].shape[1]

        if 'SD3' in preview_method or 'SC-Prior' in preview_method or 'FLUX.1' in preview_method or 'TAEF1' == preview_method:
            preview_method_channels = 16
        elif 'LTXV' in preview_method:
            preview_method_channels = 128
        else:
            preview_method_channels = 4

        if vae_opt is None and latent_channels != preview_method_channels:
            logging.warning("[PreviewBridgeLatent] The version of latent is not compatible with preview_method.\nSD3, SD1/SD2, SDXL, SC-Prior, SC-B and FLUX.1 are not compatible with each other.")
            raise Exception("The version of latent is not compatible with preview_method.<BR>SD3, SD1/SD2, SDXL, SC-Prior, SC-B and FLUX.1 are not compatible with each other.")

        need_refresh = False
        cached = core.preview_bridge_cache.get(unique_id)
        if cached is not None and not isinstance(cached[0], dict):
            core.preview_bridge_cache.pop(unique_id)
            core.preview_bridge_last_mask_cache.pop(unique_id, None)
            cached = None

        # Check if latent has changed
        if cached is None:
            need_refresh = True
        elif (cached[0] is not latent
              or (vae_opt is None and cached[2] is not None)
              or (vae_opt is None and cached[1] != preview_method)
              or (vae_opt is not None and cached[2] is not vae_opt)):
            need_refresh = True

        if need_refresh and restore_mask == "never":
            core.preview_bridge_last_mask_cache.pop(unique_id, None)

        preview_id = cached[4] if not need_refresh and image == cached[3] else image
        # API prompts can submit a raw saved file without frontend registration.
        if preview_id not in core.preview_bridge_image_id_map and (not need_refresh or restore_mask != "never"):
            is_clipspace = image and ("clipspace" in image.lower() or "[input]" in image)
            registered = is_clipspace and PreviewBridge.register_clipspace_image(image, unique_id)
            need_refresh = need_refresh or not registered

        if not need_refresh:
            _, mask, path_item = PreviewBridge.load_image(preview_id)
            res_latent = latent.copy()
            if mask is None:
                # RGB previews have no edit; an explicit saved clear has alpha.
                mask = torch.ones(latent['samples'].shape[2:], dtype=torch.float32, device="cpu").unsqueeze(0)
                res_latent.pop('noise_mask', None)
            else:
                res_latent['noise_mask'] = mask
            res_image = [path_item]
        else:
            res_latent = latent
            decoded_image = decode_latent(latent, preview_method, vae_opt)

            if 'noise_mask' in latent:
                mask = latent['noise_mask'].squeeze(0)  # 4D mask -> 3D mask

                decoded_pil = utils.to_pil(decoded_image)

                inverted_mask = 1 - mask  # invert
                resized_mask = utils.resize_mask(inverted_mask, (decoded_image.shape[1], decoded_image.shape[2]))
                result_pil = utils.apply_mask_alpha_to_pil(decoded_pil, resized_mask)

                full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path("PreviewBridge/PBL-"+self.prefix_append, folder_paths.get_temp_directory(), result_pil.size[0], result_pil.size[1])
                file = f"{filename}_{counter}.png"
                result_pil.save(os.path.join(full_output_folder, file), compress_level=4)
                res_image = [{
                                'filename': file,
                                'subfolder': 'PreviewBridge',
                                'type': 'temp',
                            }]

            else:
                if restore_mask != "never":
                    mask = core.get_previewbridge_mask(image)
                    if mask is None:
                        mask = core.preview_bridge_last_mask_cache.get(unique_id)
                    if mask is not None and restore_mask == "if_same_size" and mask.shape[1:] != decoded_image.shape[1:3]:
                        mask = None
                else:
                    mask = None

                if mask is None:
                    mask = torch.ones(latent['samples'].shape[2:], dtype=torch.float32, device="cpu").unsqueeze(0)
                    res = nodes.PreviewImage().save_images(decoded_image, filename_prefix="PreviewBridge/PBL-", prompt=prompt, extra_pnginfo=extra_pnginfo)
                else:
                    res_latent = latent.copy()
                    res_latent['noise_mask'] = mask
                    masked_images = utils.tensor_convert_rgba(decoded_image)
                    resized_mask = utils.resize_mask(mask, (decoded_image.shape[1], decoded_image.shape[2])).unsqueeze(3)
                    resized_mask = 1 - resized_mask
                    utils.tensor_putalpha(masked_images, resized_mask)
                    res = nodes.PreviewImage().save_images(masked_images, filename_prefix="PreviewBridge/PBL-", prompt=prompt, extra_pnginfo=extra_pnginfo)

                res_image = res['ui']['images']

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', res_image[0]['filename'])
            preview_id = core.set_previewbridge_image(unique_id, path, res_image[0])

        core.preview_bridge_cache[unique_id] = (latent, preview_method, vae_opt, image, preview_id)
        # The white no-mask fallback is distinct from an explicit full selection.
        is_empty_mask = 'noise_mask' not in res_latent or torch.all(mask == 0)
        if block and is_empty_mask and core.is_execution_model_version_supported():
            from comfy_execution.graph import ExecutionBlocker
            result = ExecutionBlocker(None), ExecutionBlocker(None)
        elif block and is_empty_mask:
            logging.warning("[Impact Pack] PreviewBridgeLatent: ComfyUI is outdated - blocking feature is disabled.")
            result = res_latent, mask
        else:
            result = res_latent, mask

        if not is_empty_mask:
            core.preview_bridge_last_mask_cache[unique_id] = mask

        return {
            "ui": {"images": res_image, "source_image": [image]},
            "result": result,
        }
