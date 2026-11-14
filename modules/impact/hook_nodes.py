import sys
from comfy_api.latest import io
from . import hooks
from . import defs


class DetailerAdvancedOptionsHook(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DetailerAdvancedOptionsHook",
            display_name="Detailer Advanced Options Hook",
            category="ImpactPack/Detailer",
            description="Additional Detailer settings. Combine with existing hooks using DetailerHookCombine.",
            inputs=[
                io.Boolean.Input("return_by_cycle_step", default=False,
                                 tooltip="Return a full image for each cycle, ordered by input image then cycle. Applies to image Detailers, not AnimateDiff."),
                io.Combo.Input("vae_encode_mode", options=["inherit", "disabled", "tiled"], default="inherit",
                               tooltip="Inherit the Detailer switch, disable explicit tiling, or use the specified tile. Also applies to inpaint conditioning."),
                io.Combo.Input("vae_decode_mode", options=["inherit", "disabled", "tiled"], default="inherit",
                               tooltip="Inherit the Detailer switch, disable explicit tiling, or use the specified tile. Normal decode retains its tiled fallback on failure."),
                io.Int.Input("vae_tile_size", default=512, min=64, max=4096, step=64,
                             tooltip="Tile size in pixels for tiled encode/decode. Memory follows the tile, not the crop, so use the largest your device holds."),
                io.Int.Input("vae_tile_overlap", default=64, min=0, max=4096, step=32,
                             tooltip="Overlap in pixels; capped at one quarter of the selected tile. Zero means no overlap."),
                io.Boolean.Input("return_upscaled_crops", default=False,
                                 tooltip="Return cropped and cropped_refined at the size the Detailer samples at instead of the crop size."),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
        )

    @classmethod
    def execute(cls, return_by_cycle_step=False, vae_encode_mode="inherit", vae_decode_mode="inherit",
                vae_tile_size=512, vae_tile_overlap=64, return_upscaled_crops=False):
        options = {"return_by_cycle_step": return_by_cycle_step, "return_upscaled_crops": return_upscaled_crops}
        if vae_encode_mode != "inherit":
            options["vae_encode_mode"] = vae_encode_mode
        if vae_decode_mode != "inherit":
            options["vae_decode_mode"] = vae_decode_mode
        if "tiled" in (vae_encode_mode, vae_decode_mode):
            options.update(vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap)
        return io.NodeOutput(hooks.AdvancedOptionsDetailerHook(options))


class SEGSOrderedFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2"],),
                        "order": ("BOOLEAN", {"default": True, "label_on": "descending", "label_off": "ascending"}),
                        "take_start": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "take_count": ("INT", {"default": 1, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, order, take_start, take_count):
        hook = hooks.SEGSOrderedFilterDetailerHook(target, order, take_start, take_count)
        return (hook, )


class SEGSRangeFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent"],),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "inside", "label_off": "outside"}),
                        "min_value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "max_value": ("INT", {"default": 67108864, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, mode, min_value, max_value):
        hook = hooks.SEGSRangeFilterDetailerHook(target, mode, min_value, max_value)
        return (hook, )


class SEGSLabelFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "preset": (['all'] + defs.detection_labels,),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the types of segments to be allowed, separated by commas"}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, preset, labels):
        hook = hooks.SEGSLabelFilterDetailerHook(labels)
        return (hook, )


class PreviewDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"quality": ("INT", {"default": 95, "min": 20, "max": 100})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("DETAILER_HOOK", "UPSCALER_HOOK")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    NOT_IDEMPOTENT = True

    def doit(self, quality, unique_id):
        hook = hooks.PreviewDetailerHook(unique_id, quality)
        return hook, hook


class LamaRemoverDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_threshold":("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "skip_sampling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, mask_threshold, gaussblur_radius, skip_sampling):
        hook = hooks.LamaRemoverDetailerHook(mask_threshold, gaussblur_radius, skip_sampling)
        return (hook, )


class BlackPatchRetryHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mean_thresh": ("INT", {"default": 10, "min": 0, "max": 255}),
                "var_thresh": ("INT", {"default": 5, "min": 0, "max": 255})
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    NOT_IDEMPOTENT = True

    def doit(self, mean_thresh, var_thresh):
        hook = hooks.BlackPatchRetryHook(mean_thresh, var_thresh)
        return hook, 
