import shutil
import folder_paths
import os
import sys
import importlib

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
modules_path = os.path.join(os.path.dirname(__file__), "modules")
wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
custom_wildcards_path = os.path.join(os.path.dirname(__file__), "custom_wildcards")

sys.path.append(modules_path)

import impact.config
print(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")

def do_install():
    spec = importlib.util.spec_from_file_location('impact_install', os.path.join(os.path.dirname(__file__), 'install.py'))
    impact_install = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impact_install)

# ensure dependency
if impact.config.read_config()[1] < impact.config.dependency_version:
    do_install()

# Core
# recheck dependencies for colab
try:
    import folder_paths
    import torch
    import cv2
    import mmcv
    import numpy as np
    from mmdet.apis import (inference_detector, init_detector)
    import comfy.samplers
    import comfy.sd
    import warnings
    from PIL import Image, ImageFilter
    from mmdet.evaluation import get_classes
    from skimage.measure import label, regionprops
    from collections import namedtuple
    import piexif
except:
    import importlib
    print("### ComfyUI-Impact-Pack: Reinstall dependencies (several dependencies are missing.)")
    do_install()

import impact.impact_server  # to load server api

def setup_js():
    # remove garbage
    old_js_path = os.path.join(comfy_path, "web", "extensions", "core", "impact-pack.js")
    if os.path.exists(old_js_path):
        os.remove(old_js_path)

    # setup js
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "impact-pack")
    if not os.path.exists(js_dest_path):
        os.makedirs(js_dest_path)

    js_src_path = os.path.join(impact_path, "js", "impact-pack.js")
    shutil.copy(js_src_path, js_dest_path)

    js_src_path = os.path.join(impact_path, "js", "impact-sam-editor.js")
    shutil.copy(js_src_path, js_dest_path)
    
setup_js()

import impact.legacy_nodes
from impact.impact_pack import *
from impact.detectors import *
from impact.pipe import *

impact.wildcards.read_wildcard_dict(wildcards_path)
impact.wildcards.read_wildcard_dict(custom_wildcards_path)

NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader,
    "MMDetDetectorProvider": MMDetDetectorProvider,
    "CLIPSegDetectorProvider": CLIPSegDetectorProvider,
    "ONNXDetectorProvider": ONNXDetectorProvider,

    "BitwiseAndMaskForEach": BitwiseAndMaskForEach,
    "SubtractMaskForEach": SubtractMaskForEach,

    "DetailerForEach": DetailerForEach,
    "DetailerForEachDebug": DetailerForEachTest,
    "DetailerForEachPipe": DetailerForEachPipe,
    "DetailerForEachDebugPipe": DetailerForEachTestPipe,

    "SAMDetectorCombined": SAMDetectorCombined,

    "FaceDetailer": FaceDetailer,
    "FaceDetailerPipe": FaceDetailerPipe,

    "ToDetailerPipe": ToDetailerPipe,
    "FromDetailerPipe": FromDetailerPipe,
    "ToBasicPipe": ToBasicPipe,
    "FromBasicPipe": FromBasicPipe,
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

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "Segs & Mask": SegsBitwiseAndMask,
    "EmptySegs": EmptySEGS,

    "MaskToSEGS": MaskToSEGS,
    "ToBinaryMask": ToBinaryMask,

    "BboxDetectorSEGS": BboxDetectorForEach,
    "SegmDetectorSEGS": SegmDetectorForEach,
    "ONNXDetectorSEGS": ONNXDetectorForEach,

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
    "LatentSwitch": LatentSwitch,
    "SEGSSwitch": SEGSSwitch,

    # "SaveConditioning": SaveConditioning,
    # "LoadConditioning": LoadConditioning,

    "ImpactWildcardProcessor": ImpactWildcardProcessor,
    "ImpactLogger": ImpactLogger,

    "SEGSDetailer": SEGSDetailer,
    "SEGSPaste": SEGSPaste,
    "SEGSPreview": SEGSPreview,
    "SEGSToImageList": SEGSToImageList,

    # "SEGPick": SEGPick,
    # "SEGEdit": SEGEdit,

    "RegionalSampler": RegionalSampler,
    "CombineRegionalPrompts": CombineRegionalPrompts,
    "RegionalPrompt": RegionalPrompt,

    "MaskPainter": impact.legacy_nodes.MaskPainter,
    "MMDetLoader": impact.legacy_nodes.MMDetLoader,
    "SegsMaskCombine": impact.legacy_nodes.SegsMaskCombine,
    "BboxDetectorForEach": impact.legacy_nodes.BboxDetectorForEach,
    "SegmDetectorForEach": impact.legacy_nodes.SegmDetectorForEach,
    "BboxDetectorCombined": impact.legacy_nodes.BboxDetectorCombined,
    "SegmDetectorCombined": impact.legacy_nodes.SegmDetectorCombined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS)",
    "BboxDetectorCombined_v2": "BBOX Detector (combined)",
    "SegmDetectorCombined_v2": "SEGM Detector (combined)",
    "SegsToCombinedMask": "SEGS to MASK (combined)",
    "MaskToSEGS": "MASK to SEGS",
    "BitwiseAndMaskForEach": "Bitwise(SEGS & SEGS)",
    "SubtractMaskForEach": "Bitwise(SEGS - SEGS)",
    "Segs & Mask": "Bitwise(SEGS & MASK)",
    "BitwiseAndMask": "Bitwise(MASK & MASK)",
    "SubtractMask": "Bitwise(MASK - MASK)",
    "DetailerForEach": "Detailer (SEGS)",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "DetailerForEachDebug": "DetailerDebug (SEGS)",
    "DetailerForEachDebugPipe": "DetailerDebug (SEGS/pipe)",
    "SAMDetectorCombined": "SAMDetector (combined)",
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

    "PreviewBridge": "Preview Bridge",
    "ImageSender": "Image Sender",
    "ImageReceiver": "Image Receiver",
    "ImageMaskSwitch": "Switch (images, mask)",
    "LatentSwitch": "Switch (latent)",
    "SEGSSwitch": "Switch (SEGS)",
    
    "MaskPainter": "MaskPainter (Deprecated)",
    "MMDetLoader": "MMDetLoader (Legacy)",
    "SegsMaskCombine": "SegsMaskCombine (Legacy)",
    "BboxDetectorForEach": "BboxDetectorForEach (Legacy)",
    "SegmDetectorForEach": "SegmDetectorForEach (Legacy)",
    "BboxDetectorCombined": "BboxDetectorCombined (Legacy)",
    "SegmDetectorCombined": "SegmDetectorCombined (Legacy)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
