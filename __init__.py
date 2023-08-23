"""
@author: Dr.Lt.Data
@title: Impact Pack
@nickname: Impact Pack
@description: This extension offers various detector nodes and detailer nodes that allow you to configure a workflow that automatically enhances facial details. And provide iterative upscaler.
"""

import shutil
import folder_paths
import os
import sys

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
subpack_path = os.path.join(os.path.dirname(__file__), "subpack")
modules_path = os.path.join(os.path.dirname(__file__), "modules")
wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
custom_wildcards_path = os.path.join(os.path.dirname(__file__), "custom_wildcards")

sys.path.append(modules_path)
sys.path.append(subpack_path)


import impact.config
import impact.hacky
print(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")


def do_install():
    import importlib
    spec = importlib.util.spec_from_file_location('impact_install', os.path.join(os.path.dirname(__file__), 'install.py'))
    impact_install = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impact_install)


# ensure dependency
if impact.config.get_config()['dependency_version'] < impact.config.dependency_version:
    print(f"## ComfyUI-Impact-Pack: Updating dependencies")
    do_install()


# Core
# recheck dependencies for colab
try:
    import folder_paths
    import torch
    import cv2
    import numpy as np
    import comfy.samplers
    import comfy.sd
    import warnings
    from PIL import Image, ImageFilter
    from skimage.measure import label, regionprops
    from collections import namedtuple
    import piexif

    if not impact.config.get_config()['mmdet_skip']:
        import mmcv
        from mmdet.apis import (inference_detector, init_detector)
        from mmdet.evaluation import get_classes
except:
    import importlib
    print("### ComfyUI-Impact-Pack: Reinstall dependencies (several dependencies are missing.)")
    do_install()

import impact.impact_server  # to load server api


def setup_js():
    import nodes
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "impact-pack")

    if hasattr(nodes, "EXTENSION_WEB_DIRS"):
        if os.path.exists(js_dest_path):
            shutil.rmtree(js_dest_path)
    else:
        print(f"[WARN] ComfyUI-Impact-Pack: Your ComfyUI version is outdated. Please update to the latest version.")
        # setup js
        if not os.path.exists(js_dest_path):
            os.makedirs(js_dest_path)

        js_src_path = os.path.join(impact_path, "js", "impact-pack.js")
        shutil.copy(js_src_path, js_dest_path)

        js_src_path = os.path.join(impact_path, "js", "impact-sam-editor.js")
        shutil.copy(js_src_path, js_dest_path)

        js_src_path = os.path.join(impact_path, "js", "comboBoolMigration.js")
        shutil.copy(js_src_path, js_dest_path)


    
setup_js()

from impact.impact_pack import *
from impact.detectors import *
from impact.pipe import *
from impact.logics import *
from impact.util_nodes import *

impact.wildcards.read_wildcard_dict(wildcards_path)
impact.wildcards.read_wildcard_dict(custom_wildcards_path)

NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader,
    "CLIPSegDetectorProvider": CLIPSegDetectorProvider,
    "ONNXDetectorProvider": ONNXDetectorProvider,

    "BitwiseAndMaskForEach": BitwiseAndMaskForEach,
    "SubtractMaskForEach": SubtractMaskForEach,

    "DetailerForEach": DetailerForEach,
    "DetailerForEachDebug": DetailerForEachTest,
    "DetailerForEachPipe": DetailerForEachPipe,
    "DetailerForEachDebugPipe": DetailerForEachTestPipe,

    "SAMDetectorCombined": SAMDetectorCombined,
    "SAMDetectorSegmented": SAMDetectorSegmented,

    "FaceDetailer": FaceDetailer,
    "FaceDetailerPipe": FaceDetailerPipe,

    "ToDetailerPipe": ToDetailerPipe,
    "FromDetailerPipe": FromDetailerPipe,
    "FromDetailerPipe_v2": FromDetailerPipe_v2,
    "ToBasicPipe": ToBasicPipe,
    "FromBasicPipe": FromBasicPipe,
    "FromBasicPipe_v2": FromBasicPipe_v2,
    "BasicPipeToDetailerPipe": BasicPipeToDetailerPipe,
    "DetailerPipeToBasicPipe": DetailerPipeToBasicPipe,
    "EditBasicPipe": EditBasicPipe,
    "EditDetailerPipe": EditDetailerPipe,

    "LatentPixelScale": LatentPixelScale,
    "PixelKSampleUpscalerProvider": PixelKSampleUpscalerProvider,
    "PixelKSampleUpscalerProviderPipe": PixelKSampleUpscalerProviderPipe,
    "IterativeLatentUpscale": IterativeLatentUpscale,
    "IterativeImageUpscale": IterativeImageUpscale,
    "PixelTiledKSampleUpscalerProvider": PixelTiledKSampleUpscalerProvider,
    "PixelTiledKSampleUpscalerProviderPipe": PixelTiledKSampleUpscalerProviderPipe,
    "TwoSamplersForMaskUpscalerProvider": TwoSamplersForMaskUpscalerProvider,
    "TwoSamplersForMaskUpscalerProviderPipe": TwoSamplersForMaskUpscalerProviderPipe,

    "PixelKSampleHookCombine": PixelKSampleHookCombine,
    "DenoiseScheduleHookProvider": DenoiseScheduleHookProvider,
    "CfgScheduleHookProvider": CfgScheduleHookProvider,
    "NoiseInjectionHookProvider": NoiseInjectionHookProvider,
    "NoiseInjectionDetailerHookProvider": NoiseInjectionDetailerHookProvider,

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "AddMask": AddMask,
    "Segs & Mask": SegsBitwiseAndMask,
    "Segs & Mask ForEach": SegsBitwiseAndMaskForEach,
    "EmptySegs": EmptySEGS,

    "MaskToSEGS": MaskToSEGS,
    "ToBinaryMask": ToBinaryMask,
    "MasksToMaskList": MasksToMaskList,

    "BboxDetectorSEGS": BboxDetectorForEach,
    "SegmDetectorSEGS": SegmDetectorForEach,
    "ONNXDetectorSEGS": ONNXDetectorForEach,
    "ImpactSimpleDetectorSEGS": SimpleDetectorForEach,
    "ImpactSimpleDetectorSEGSPipe": SimpleDetectorForEachPipe,

    "BboxDetectorCombined_v2": BboxDetectorCombined,
    "SegmDetectorCombined_v2": SegmDetectorCombined,
    "SegsToCombinedMask": SegsToCombinedMask,

    "KSamplerProvider": KSamplerProvider,
    "TwoSamplersForMask": TwoSamplersForMask,
    "TiledKSamplerProvider": TiledKSamplerProvider,

    "KSamplerAdvancedProvider": KSamplerAdvancedProvider,
    "TwoAdvancedSamplersForMask": TwoAdvancedSamplersForMask,

    "PreviewBridge": PreviewBridge,
    "ImageSender": ImageSender,
    "ImageReceiver": ImageReceiver,
    "LatentSender": LatentSender,
    "LatentReceiver": LatentReceiver,
    "ImageMaskSwitch": ImageMaskSwitch,
    "LatentSwitch": GeneralSwitch,
    "SEGSSwitch": GeneralSwitch,
    "ImpactSwitch": GeneralSwitch,

    # "SaveConditioning": SaveConditioning,
    # "LoadConditioning": LoadConditioning,

    "ImpactWildcardProcessor": ImpactWildcardProcessor,
    "ImpactWildcardEncode": ImpactWildcardEncode,

    "SEGSDetailer": SEGSDetailer,
    "SEGSPaste": SEGSPaste,
    "SEGSPreview": SEGSPreview,
    "SEGSToImageList": SEGSToImageList,
    "ImpactSEGSToMaskList": SEGSToMaskList,
    "ImpactSEGSConcat": SEGSConcat,

    "ImpactKSamplerBasicPipe": KSamplerBasicPipe,
    "ImpactKSamplerAdvancedBasicPipe": KSamplerAdvancedBasicPipe,

    "ReencodeLatent": ReencodeLatent,
    "ReencodeLatentPipe": ReencodeLatentPipe,

    "ImpactImageBatchToImageList": ImageBatchToImageList,
    "ImpactMakeImageList": MakeImageList,

    "RegionalSampler": RegionalSampler,
    "CombineRegionalPrompts": CombineRegionalPrompts,
    "RegionalPrompt": RegionalPrompt,

    "ImpactSEGSLabelFilter": SEGSLabelFilter,
    "ImpactSEGSRangeFilter": SEGSRangeFilter,
    "ImpactSEGSOrderedFilter": SEGSOrderedFilter,

    "ImpactCompare": ImpactCompare,
    "ImpactConditionalBranch": ImpactConditionalBranch,
    "ImpactInt": ImpactInt,
    # "ImpactFloat": ImpactFloat,
    "ImpactValueSender": ImpactValueSender,
    "ImpactValueReceiver": ImpactValueReceiver,
    "ImpactImageInfo": ImpactImageInfo,
    "ImpactMinMax": ImpactMinMax,
    "ImpactNeg": ImpactNeg,
    "ImpactConditionalStopIteration": ImpactConditionalStopIteration,
    "ImpactStringSelector": StringSelector,

    "ImpactLogger": ImpactLogger,
    "ImpactDummyInput": ImpactDummyInput,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS)",
    "ImpactSimpleDetectorSEGS": "Simple Detector (SEGS)",
    "ImpactSimpleDetectorSEGSPipe": "Simple Detector (SEGS/pipe)",

    "BboxDetectorCombined_v2": "BBOX Detector (combined)",
    "SegmDetectorCombined_v2": "SEGM Detector (combined)",
    "SegsToCombinedMask": "SEGS to MASK (combined)",
    "MaskToSEGS": "MASK to SEGS",
    "BitwiseAndMaskForEach": "Bitwise(SEGS & SEGS)",
    "SubtractMaskForEach": "Bitwise(SEGS - SEGS)",
    "Segs & Mask": "Bitwise(SEGS & MASK)",
    "Segs & Mask ForEach": "Bitwise(SEGS & MASKS ForEach)",
    "BitwiseAndMask": "Bitwise(MASK & MASK)",
    "SubtractMask": "Bitwise(MASK - MASK)",
    "AddMask": "Bitwise(MASK + MASK)",
    "DetailerForEach": "Detailer (SEGS)",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "DetailerForEachDebug": "DetailerDebug (SEGS)",
    "DetailerForEachDebugPipe": "DetailerDebug (SEGS/pipe)",
    "SAMDetectorCombined": "SAMDetector (combined)",
    "SAMDetectorSegmented": "SAMDetector (segmented)",
    "FaceDetailerPipe": "FaceDetailer (pipe)",

    "BasicPipeToDetailerPipe": "BasicPipe -> DetailerPipe",
    "DetailerPipeToBasicPipe": "DetailerPipe -> BasicPipe",
    "EditBasicPipe": "Edit BasicPipe",
    "EditDetailerPipe": "Edit DetailerPipe",

    "LatentPixelScale": "Latent Scale (on Pixel Space)",
    "IterativeLatentUpscale": "Iterative Upscale (Latent)",
    "IterativeImageUpscale": "Iterative Upscale (Image)",

    "TwoSamplersForMaskUpscalerProvider": "TwoSamplersForMask Upscaler Provider",
    "TwoSamplersForMaskUpscalerProviderPipe": "TwoSamplersForMask Upscaler Provider (pipe)",

    "ReencodeLatent": "Reencode Latent",
    "ReencodeLatentPipe": "Reencode Latent (pipe)",

    "ImpactKSamplerBasicPipe": "KSampler (pipe)",
    "ImpactKSamplerAdvancedBasicPipe": "KSampler (Advanced/pipe)",
    "ImpactSEGSLabelFilter": "SEGS Filter (label)",
    "ImpactSEGSRangeFilter": "SEGS Filter (range)",
    "ImpactSEGSOrderedFilter": "SEGS Filter (ordered)",
    "ImpactSEGSConcat": "SEGS Concat",

    "PreviewBridge": "Preview Bridge",
    "ImageSender": "Image Sender",
    "ImageReceiver": "Image Receiver",
    "ImageMaskSwitch": "Switch (images, mask)",
    "ImpactSwitch": "Switch (Any)",

    "MasksToMaskList": "Masks to Mask List",
    "ImpactImageBatchToImageList": "Image batch to Image List",
    "ImpactMakeImageList": "Make Image List",
    "ImpactStringSelector": "String Selector",

    "LatentSwitch": "Switch (latent/legacy)",
    "SEGSSwitch": "Switch (SEGS/legacy)"
}

if not impact.config.get_config()['mmdet_skip']:
    from impact.mmdet_nodes import *
    import impact.legacy_nodes
    NODE_CLASS_MAPPINGS.update({
        "MMDetDetectorProvider": MMDetDetectorProvider,
        "MMDetLoader": impact.legacy_nodes.MMDetLoader,
        "MaskPainter": impact.legacy_nodes.MaskPainter,
        "SegsMaskCombine": impact.legacy_nodes.SegsMaskCombine,
        "BboxDetectorForEach": impact.legacy_nodes.BboxDetectorForEach,
        "SegmDetectorForEach": impact.legacy_nodes.SegmDetectorForEach,
        "BboxDetectorCombined": impact.legacy_nodes.BboxDetectorCombined,
        "SegmDetectorCombined": impact.legacy_nodes.SegmDetectorCombined,
    })

    NODE_DISPLAY_NAME_MAPPINGS.update({
        "MaskPainter": "MaskPainter (Deprecated)",
        "MMDetLoader": "MMDetLoader (Legacy)",
        "SegsMaskCombine": "SegsMaskCombine (Legacy)",
        "BboxDetectorForEach": "BboxDetectorForEach (Legacy)",
        "SegmDetectorForEach": "SegmDetectorForEach (Legacy)",
        "BboxDetectorCombined": "BboxDetectorCombined (Legacy)",
        "SegmDetectorCombined": "SegmDetectorCombined (Legacy)",
    })

try:
    import impact.subpack_nodes

    NODE_CLASS_MAPPINGS.update(impact.subpack_nodes.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(impact.subpack_nodes.NODE_DISPLAY_NAME_MAPPINGS)

except:
    pass

WEB_DIRECTORY = "js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
