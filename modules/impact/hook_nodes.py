import sys
from comfy_api.latest import io
from . import hooks
from .node_base import ImpactNode
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


class SEGSOrderedFilterDetailerHookProvider(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SEGSOrderedFilterDetailerHookProvider",
            display_name="SEGSOrderedFilterDetailerHookProvider",
            category="ImpactPack/Util",
            inputs=[
                io.Combo.Input("target", options=["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2"]),
                io.Boolean.Input("order", default=True, label_on="descending", label_off="ascending"),
                io.Int.Input("take_start", default=0, min=0, max=sys.maxsize, step=1),
                io.Int.Input("take_count", default=1, min=0, max=sys.maxsize, step=1),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
        )

    @classmethod
    def execute(cls, target, order, take_start, take_count):
        hook = hooks.SEGSOrderedFilterDetailerHook(target, order, take_start, take_count)
        return io.NodeOutput(hook)


class SEGSRangeFilterDetailerHookProvider(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SEGSRangeFilterDetailerHookProvider",
            display_name="SEGSRangeFilterDetailerHookProvider",
            category="ImpactPack/Util",
            inputs=[
                io.Combo.Input("target", options=["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent"]),
                io.Boolean.Input("mode", default=True, label_on="inside", label_off="outside"),
                io.Int.Input("min_value", default=0, min=0, max=sys.maxsize, step=1),
                io.Int.Input("max_value", default=67108864, min=0, max=sys.maxsize, step=1),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
        )

    @classmethod
    def execute(cls, target, mode, min_value, max_value):
        hook = hooks.SEGSRangeFilterDetailerHook(target, mode, min_value, max_value)
        return io.NodeOutput(hook)


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


class SEGSRoundCornersDetailerHookProvider(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SEGSRoundCornersDetailerHookProvider",
            display_name="SEGSRoundCornersDetailerHookProvider",
            category="ImpactPack/Util",
            description="Provides a hook that rounds the corners of the mask of each detected area at the bounds of the mask, with `radius` in pixels limited to half the shorter side of those bounds. "
                        "The detector rounds the mask after its dilation, and the Detailer feathers the rounded mask when it pastes.",
            inputs=[
                io.Int.Input("radius", default=16, min=0, max=512, step=1),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
        )

    @classmethod
    def execute(cls, radius):
        hook = hooks.SEGSRoundCornersDetailerHook(radius)
        return io.NodeOutput(hook)


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


class LamaRemoverDetailerHookProvider(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LamaRemoverDetailerHookProvider",
            display_name="LamaRemoverDetailerHookProvider",
            category="ImpactPack/Util",
            inputs=[
                io.Int.Input("mask_threshold", default=250, min=0, max=255, step=1, display_mode=io.NumberDisplay.slider),
                io.Int.Input("gaussblur_radius", default=8, min=0, max=20, step=1, display_mode=io.NumberDisplay.slider),
                io.Boolean.Input("skip_sampling", default=True),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
        )

    @classmethod
    def execute(cls, mask_threshold, gaussblur_radius, skip_sampling):
        hook = hooks.LamaRemoverDetailerHook(mask_threshold, gaussblur_radius, skip_sampling)
        return io.NodeOutput(hook)


class BlackPatchRetryHookProvider(ImpactNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BlackPatchRetryHookProvider",
            display_name="BlackPatchRetryHookProvider",
            category="ImpactPack/Util",
            inputs=[
                io.Int.Input("mean_thresh", default=10, min=0, max=255),
                io.Int.Input("var_thresh", default=5, min=0, max=255),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output()],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, mean_thresh, var_thresh):
        hook = hooks.BlackPatchRetryHook(mean_thresh, var_thresh)
        return io.NodeOutput(hook)
