import shutil
import folder_paths
import os
import sys

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.dirname(__file__)

sys.path.append(impact_path)

import impact_config
print(f"### Loading: ComfyUI-Impact-Pack ({impact_config.version})")

# ensure dependency
if impact_config.read_config()[1] < impact_config.dependency_version:
    import install  # to install dependencies
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
except:
    print("### ComfyUI-Impact-Pack: Reinstall dependencies (several dependencies are missing.)")
    import install

import impact_server  # to load server api

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
    
setup_js()

import legacy_nodes
from impact_pack import *
from detectors import *
from impact_pipe import *

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

    "ToDetailerPipe": ToDetailerPipe ,
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

    "PixelKSampleHookCombine": PixelKSampleHookCombine,
    "DenoiseScheduleHookProvider": DenoiseScheduleHookProvider,
    "CfgScheduleHookProvider": CfgScheduleHookProvider,

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "Segs & Mask": SegsBitwiseAndMask,
    "EmptySegs": EmptySEGS,

    "MaskToSEGS": MaskToSEGS,
    "ToBinaryMask": ToBinaryMask,

    "MaskPainter": MaskPainter,

    "BboxDetectorSEGS": BboxDetectorForEach,
    "SegmDetectorSEGS": SegmDetectorForEach,
    "ONNXDetectorSEGS": ONNXDetectorForEach,

    "BboxDetectorCombined": BboxDetectorCombined,
    "SegmDetectorCombined": SegmDetectorCombined,
    "SegsToCombinedMask": SegsToCombinedMask,

    "KSamplerProvider": KSamplerProvider,
    "TwoSamplersForMask": TwoSamplersForMask,

    "MMDetLoader": legacy_nodes.MMDetLoader,
    "SegsMaskCombine": legacy_nodes.SegsMaskCombine,
    "BboxDetectorForEach": legacy_nodes.BboxDetectorForEach,
    "SegmDetectorForEach": legacy_nodes.SegmDetectorForEach,
    "BboxDetectorCombined": legacy_nodes.BboxDetectorCombined,
    "SegmDetectorCombined": legacy_nodes.SegmDetectorCombined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS)",
    "BboxDetectorCombined": "BBOX Detector (combined)",
    "SegmDetectorCombined": "SEGM Detector (combined)",
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

    "MMDetLoader": "MMDetLoader (Legacy)",
    "SegsMaskCombine": "SegsMaskCombine (Legacy)",
    "BboxDetectorForEach": "BboxDetectorForEach (Legacy)",
    "SegmDetectorForEach": "SegmDetectorForEach (Legacy)",
    "BboxDetectorCombined": "BboxDetectorCombined (Legacy)",
    "SegmDetectorCombined": "SegmDetectorCombined (Legacy)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
