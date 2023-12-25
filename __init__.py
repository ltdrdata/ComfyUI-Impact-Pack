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
import traceback

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
subpack_path = os.path.join(os.path.dirname(__file__), "impact_subpack")
modules_path = os.path.join(os.path.dirname(__file__), "modules")
wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
custom_wildcards_path = os.path.join(os.path.dirname(__file__), "custom_wildcards")

sys.path.append(modules_path)


import impact.config
import impact.sample_error_enhancer
print(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")


def do_install():
    import importlib
    spec = importlib.util.spec_from_file_location('impact_install', os.path.join(os.path.dirname(__file__), 'install.py'))
    impact_install = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impact_install)


# ensure dependency
if not os.path.exists(os.path.join(subpack_path, ".git")) and os.path.exists(subpack_path):
    print(f"### CompfyUI-Impact-Pack: corrupted subpack detected.")
    shutil.rmtree(subpack_path)

if impact.config.get_config()['dependency_version'] < impact.config.dependency_version or not os.path.exists(subpack_path):
    print(f"### ComfyUI-Impact-Pack: Updating dependencies [{impact.config.get_config()['dependency_version']} -> {impact.config.dependency_version}]")
    do_install()

sys.path.append(subpack_path)

# Core
# recheck dependencies for colab
try:
    import impact.subpack_nodes  # This import must be done before cv2.

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
from impact.segs_nodes import *
from impact.special_samplers import *
from impact.hf_nodes import *
from impact.bridge_nodes import *

import threading

wildcard_path = impact.config.get_config()['custom_wildcards']


def wildcard_load():
    with wildcards.wildcard_lock:
        impact.wildcards.read_wildcard_dict(wildcards_path)

        try:
            impact.wildcards.read_wildcard_dict(impact.config.get_config()['custom_wildcards'])
        except Exception as e:
            print(f"[Impact Pack] Failed to load custom wildcards directory.")

        print(f"[Impact Pack] Wildcards loading done.")


threading.Thread(target=wildcard_load).start()


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
    "MaskDetailerPipe": MaskDetailerPipe,

    "ToDetailerPipe": ToDetailerPipe,
    "ToDetailerPipeSDXL": ToDetailerPipeSDXL,
    "FromDetailerPipe": FromDetailerPipe,
    "FromDetailerPipe_v2": FromDetailerPipe_v2,
    "FromDetailerPipeSDXL": FromDetailerPipe_SDXL,
    "ToBasicPipe": ToBasicPipe,
    "FromBasicPipe": FromBasicPipe,
    "FromBasicPipe_v2": FromBasicPipe_v2,
    "BasicPipeToDetailerPipe": BasicPipeToDetailerPipe,
    "BasicPipeToDetailerPipeSDXL": BasicPipeToDetailerPipeSDXL,
    "DetailerPipeToBasicPipe": DetailerPipeToBasicPipe,
    "EditBasicPipe": EditBasicPipe,
    "EditDetailerPipe": EditDetailerPipe,
    "EditDetailerPipeSDXL": EditDetailerPipeSDXL,

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
    "DetailerHookCombine": DetailerHookCombine,
    "DenoiseScheduleHookProvider": DenoiseScheduleHookProvider,
    "CfgScheduleHookProvider": CfgScheduleHookProvider,
    "NoiseInjectionHookProvider": NoiseInjectionHookProvider,
    "UnsamplerHookProvider": UnsamplerHookProvider,
    "NoiseInjectionDetailerHookProvider": NoiseInjectionDetailerHookProvider,
    "UnsamplerDetailerHookProvider": UnsamplerDetailerHookProvider,
    "CoreMLDetailerHookProvider": CoreMLDetailerHookProvider,
    "DenoiseSchedulerDetailerHookProvider": DenoiseSchedulerDetailerHookProvider,

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "AddMask": AddMask,
    "ImpactSegsAndMask": SegsBitwiseAndMask,
    "ImpactSegsAndMaskForEach": SegsBitwiseAndMaskForEach,
    "EmptySegs": EmptySEGS,

    "MediaPipeFaceMeshToSEGS": MediaPipeFaceMeshToSEGS,
    "MaskToSEGS": MaskToSEGS,
    "MaskToSEGS_for_AnimateDiff": MaskToSEGS_for_AnimateDiff,
    "ToBinaryMask": ToBinaryMask,
    "MasksToMaskList": MasksToMaskList,
    "MaskListToMaskBatch": MaskListToMaskBatch,
    "ImageListToImageBatch": ImageListToImageBatch,
    "SetDefaultImageForSEGS": DefaultImageForSEGS,

    "BboxDetectorSEGS": BboxDetectorForEach,
    "SegmDetectorSEGS": SegmDetectorForEach,
    "ONNXDetectorSEGS": BboxDetectorForEach,
    "ImpactSimpleDetectorSEGS_for_AD": SimpleDetectorForAnimateDiff,
    "ImpactSimpleDetectorSEGS": SimpleDetectorForEach,
    "ImpactSimpleDetectorSEGSPipe": SimpleDetectorForEachPipe,
    "ImpactControlNetApplySEGS": ControlNetApplySEGS,

    "ImpactDecomposeSEGS": DecomposeSEGS,
    "ImpactAssembleSEGS": AssembleSEGS,
    "ImpactFrom_SEG_ELT": From_SEG_ELT,
    "ImpactEdit_SEG_ELT": Edit_SEG_ELT,
    "ImpactDilate_Mask_SEG_ELT": Dilate_SEG_ELT,
    "ImpactDilateMask": DilateMask,
    "ImpactGaussianBlurMask": GaussianBlurMask,
    "ImpactDilateMaskInSEGS": DilateMaskInSEGS,
    "ImpactGaussianBlurMaskInSEGS": GaussianBlurMaskInSEGS,
    "ImpactScaleBy_BBOX_SEG_ELT": SEG_ELT_BBOX_ScaleBy,

    "BboxDetectorCombined_v2": BboxDetectorCombined,
    "SegmDetectorCombined_v2": SegmDetectorCombined,
    "SegsToCombinedMask": SegsToCombinedMask,

    "KSamplerProvider": KSamplerProvider,
    "TwoSamplersForMask": TwoSamplersForMask,
    "TiledKSamplerProvider": TiledKSamplerProvider,

    "KSamplerAdvancedProvider": KSamplerAdvancedProvider,
    "TwoAdvancedSamplersForMask": TwoAdvancedSamplersForMask,

    "PreviewBridge": PreviewBridge,
    "PreviewBridgeLatent": PreviewBridgeLatent,
    "ImageSender": ImageSender,
    "ImageReceiver": ImageReceiver,
    "LatentSender": LatentSender,
    "LatentReceiver": LatentReceiver,
    "ImageMaskSwitch": ImageMaskSwitch,
    "LatentSwitch": GeneralSwitch,
    "SEGSSwitch": GeneralSwitch,
    "ImpactSwitch": GeneralSwitch,
    "ImpactInversedSwitch": GeneralInversedSwitch,

    "ImpactWildcardProcessor": ImpactWildcardProcessor,
    "ImpactWildcardEncode": ImpactWildcardEncode,

    "SEGSDetailer": SEGSDetailer,
    "SEGSPaste": SEGSPaste,
    "SEGSPreview": SEGSPreview,
    "SEGSToImageList": SEGSToImageList,
    "ImpactSEGSToMaskList": SEGSToMaskList,
    "ImpactSEGSToMaskBatch": SEGSToMaskBatch,
    "ImpactSEGSConcat": SEGSConcat,
    "ImpactSEGSPicker": SEGSPicker,

    "SEGSDetailerForAnimateDiff": SEGSDetailerForAnimateDiff,

    "ImpactKSamplerBasicPipe": KSamplerBasicPipe,
    "ImpactKSamplerAdvancedBasicPipe": KSamplerAdvancedBasicPipe,

    "ReencodeLatent": ReencodeLatent,
    "ReencodeLatentPipe": ReencodeLatentPipe,

    "ImpactImageBatchToImageList": ImageBatchToImageList,
    "ImpactMakeImageList": MakeImageList,
    "ImpactMakeImageBatch": MakeImageBatch,

    "RegionalSampler": RegionalSampler,
    "RegionalSamplerAdvanced": RegionalSamplerAdvanced,
    "CombineRegionalPrompts": CombineRegionalPrompts,
    "RegionalPrompt": RegionalPrompt,

    "ImpactCombineConditionings": CombineConditionings,
    "ImpactConcatConditionings": ConcatConditionings,

    "ImpactSEGSLabelFilter": SEGSLabelFilter,
    "ImpactSEGSRangeFilter": SEGSRangeFilter,
    "ImpactSEGSOrderedFilter": SEGSOrderedFilter,

    "ImpactCompare": ImpactCompare,
    "ImpactConditionalBranch": ImpactConditionalBranch,
    "ImpactInt": ImpactInt,
    "ImpactFloat": ImpactFloat,
    "ImpactValueSender": ImpactValueSender,
    "ImpactValueReceiver": ImpactValueReceiver,
    "ImpactImageInfo": ImpactImageInfo,
    "ImpactMinMax": ImpactMinMax,
    "ImpactNeg": ImpactNeg,
    "ImpactConditionalStopIteration": ImpactConditionalStopIteration,
    "ImpactStringSelector": StringSelector,

    "RemoveNoiseMask": RemoveNoiseMask,

    "ImpactLogger": ImpactLogger,
    "ImpactDummyInput": ImpactDummyInput,

    "ImpactQueueTrigger": ImpactQueueTrigger,
    "ImpactQueueTriggerCountdown": ImpactQueueTriggerCountdown,
    "ImpactSetWidgetValue": ImpactSetWidgetValue,
    "ImpactNodeSetMuteState": ImpactNodeSetMuteState,
    "ImpactControlBridge": ImpactControlBridge,
    "ImpactIsNotEmptySEGS": ImpactNotEmptySEGS,
    "ImpactSleep": ImpactSleep,
    "ImpactRemoteBoolean": ImpactRemoteBoolean,
    "ImpactRemoteInt": ImpactRemoteInt,

    "ImpactHFTransformersClassifierProvider": HF_TransformersClassifierProvider,
    "ImpactSEGSClassify": SEGS_Classify
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS/legacy) - use BBOXDetector",
    "ImpactSimpleDetectorSEGS_for_AD": "Simple Detector for AnimateDiff (SEGS)",
    "ImpactSimpleDetectorSEGS": "Simple Detector (SEGS)",
    "ImpactSimpleDetectorSEGSPipe": "Simple Detector (SEGS/pipe)",
    "ImpactControlNetApplySEGS": "ControlNetApply (SEGS)",

    "BboxDetectorCombined_v2": "BBOX Detector (combined)",
    "SegmDetectorCombined_v2": "SEGM Detector (combined)",
    "SegsToCombinedMask": "SEGS to MASK (combined)",
    "MediaPipeFaceMeshToSEGS": "MediaPipe FaceMesh to SEGS",
    "MaskToSEGS": "MASK to SEGS",
    "MaskToSEGS_for_AnimateDiff": "MASK to SEGS for AnimateDiff",
    "BitwiseAndMaskForEach": "Bitwise(SEGS & SEGS)",
    "SubtractMaskForEach": "Bitwise(SEGS - SEGS)",
    "ImpactSegsAndMask": "Bitwise(SEGS & MASK)",
    "ImpactSegsAndMaskForEach": "Bitwise(SEGS & MASKS ForEach)",
    "BitwiseAndMask": "Bitwise(MASK & MASK)",
    "SubtractMask": "Bitwise(MASK - MASK)",
    "AddMask": "Bitwise(MASK + MASK)",
    "DetailerForEach": "Detailer (SEGS)",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "DetailerForEachDebug": "DetailerDebug (SEGS)",
    "DetailerForEachDebugPipe": "DetailerDebug (SEGS/pipe)",
    "SEGSDetailerForAnimateDiff": "Detailer For AnimateDiff (SEGS/pipe)",

    "SAMDetectorCombined": "SAMDetector (combined)",
    "SAMDetectorSegmented": "SAMDetector (segmented)",
    "FaceDetailerPipe": "FaceDetailer (pipe)",
    "MaskDetailerPipe": "MaskDetailer (Pipe)",

    "FromDetailerPipeSDXL": "FromDetailer (SDXL/pipe)",
    "BasicPipeToDetailerPipeSDXL": "BasicPipe -> DetailerPipe (SDXL)",
    "EditDetailerPipeSDXL": "Edit DetailerPipe (SDXL)",

    "BasicPipeToDetailerPipe": "BasicPipe -> DetailerPipe",
    "DetailerPipeToBasicPipe": "DetailerPipe -> BasicPipe",
    "EditBasicPipe": "Edit BasicPipe",
    "EditDetailerPipe": "Edit DetailerPipe",

    "LatentPixelScale": "Latent Scale (on Pixel Space)",
    "IterativeLatentUpscale": "Iterative Upscale (Latent/on Pixel Space)",
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
    "ImpactSEGSToMaskList": "SEGS to Mask List",
    "ImpactSEGSToMaskBatch": "SEGS to Mask Batch",
    "ImpactSEGSPicker": "Picker (SEGS)",

    "ImpactDecomposeSEGS": "Decompose (SEGS)",
    "ImpactAssembleSEGS": "Assemble (SEGS)",
    "ImpactFrom_SEG_ELT": "From SEG_ELT",
    "ImpactEdit_SEG_ELT": "Edit SEG_ELT",
    "ImpactDilate_Mask_SEG_ELT": "Dilate Mask (SEG_ELT)",
    "ImpactScaleBy_BBOX_SEG_ELT": "ScaleBy BBOX (SEG_ELT)",
    "ImpactDilateMask": "Dilate Mask",
    "ImpactGaussianBlurMask": "Gaussian Blur Mask",
    "ImpactDilateMaskInSEGS": "Dilate Mask (SEGS)",
    "ImpactGaussianBlurMaskInSEGS": "Gaussian Blur Mask (SEGS)",

    "PreviewBridge": "Preview Bridge (Image)",
    "PreviewBridgeLatent": "Preview Bridge (Latent)",
    "ImageSender": "Image Sender",
    "ImageReceiver": "Image Receiver",
    "ImageMaskSwitch": "Switch (images, mask)",
    "ImpactSwitch": "Switch (Any)",
    "ImpactInversedSwitch": "Inversed Switch (Any)",

    "MasksToMaskList": "Masks to Mask List",
    "MaskListToMaskBatch": "Mask List to Masks",
    "ImpactImageBatchToImageList": "Image batch to Image List",
    "ImageListToImageBatch": "Image List to Image Batch",
    "ImpactMakeImageList": "Make Image List",
    "ImpactMakeImageBatch": "Make Image Batch",
    "ImpactStringSelector": "String Selector",
    "ImpactIsNotEmptySEGS": "SEGS isn't Empty",
    "SetDefaultImageForSEGS": "Set Default Image for SEGS",

    "RemoveNoiseMask": "Remove Noise Mask",

    "ImpactCombineConditionings": "Combine Conditionings",
    "ImpactConcatConditionings": "Concat Conditionings",

    "ImpactQueueTrigger": "Queue Trigger",
    "ImpactQueueTriggerCountdown": "Queue Trigger (Countdown)",
    "ImpactSetWidgetValue": "Set Widget Value",
    "ImpactNodeSetMuteState": "Set Mute State",
    "ImpactControlBridge": "Control Bridge",
    "ImpactSleep": "Sleep",
    "ImpactRemoteBoolean": "Remote Boolean (on prompt)",
    "ImpactRemoteInt": "Remote Int (on prompt)",

    "ImpactHFTransformersClassifierProvider": "HF Transformers Classifier Provider",
    "ImpactSEGSClassify": "SEGS Classify",

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
except Exception as e:
    print("### ComfyUI-Impact-Pack: (IMPORT FAILED) Subpack\n")
    print("  The module at the `custom_nodes/ComfyUI-Impact-Pack/impact_subpack` path appears to be incomplete.")
    print("  Recommended to delete the path and restart ComfyUI.")
    print("  If the issue persists, please report it to https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues.")
    print("\n---------------------------------")
    traceback.print_exc()
    print("---------------------------------\n")

WEB_DIRECTORY = "js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
