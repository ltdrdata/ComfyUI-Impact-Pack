"""
@author: Dr.Lt.Data
@title: Impact Pack
@nickname: Impact Pack
@description: This extension offers various detector nodes and detailer nodes that allow you to configure a workflow that automatically enhances facial details. And provide iterative upscaler.
"""

import folder_paths
import os
import sys
import logging

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)

import impact.config
logging.info(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")

try:
    import folder_paths
    import torch                  # noqa: F401
    import cv2                    # noqa: F401
    from cv2 import setNumThreads # noqa: F401
    import numpy as np            # noqa: F401
    import comfy.samplers
    import comfy.sd               # noqa: F401
    from PIL import Image, ImageFilter             # noqa: F401
    from skimage.measure import label, regionprops # noqa: F401
    from collections import namedtuple             # noqa: F401
    import piexif                                  # noqa: F401
    import nodes
except Exception as e:
    import logging
    logging.error("[Impact Pack] Failed to import due to several dependencies are missing!!!!")
    raise e


import impact.impact_server

from .modules.impact.impact_pack import (
    SAMLoader, CLIPSegDetectorProvider, ONNXDetectorProvider, BitwiseAndMaskForEach, SubtractMaskForEach, DetailerForEach, DetailerForEachAutoRetry,
    DetailerForEachTest, DetailerForEachPipe, DetailerForEachTestPipe, FaceDetailer, FaceDetailerPipe, MaskDetailerPipe, LatentPixelScale,
    PixelKSampleUpscalerProvider, PixelKSampleUpscalerProviderPipe, IterativeLatentUpscale, IterativeImageUpscale, PixelTiledKSampleUpscalerProvider,
    PixelTiledKSampleUpscalerProviderPipe, TwoSamplersForMaskUpscalerProvider, TwoSamplersForMaskUpscalerProviderPipe, PixelKSampleHookCombine,
    DenoiseScheduleHookProvider, StepsScheduleHookProvider, CfgScheduleHookProvider, NoiseInjectionHookProvider, UnsamplerHookProvider,
    CoreMLDetailerHookProvider, CustomSamplerDetailerHookProvider, DetailerHookCombine, NoiseInjectionDetailerHookProvider,
    UnsamplerDetailerHookProvider, DenoiseSchedulerDetailerHookProvider, VariationNoiseDetailerHookProvider, BitwiseAndMask, SubtractMask, AddMask,
    MaskRectArea, MaskRectAreaAdvanced, SegsBitwiseAndMask, SegsBitwiseAndMaskForEach, FlattenMask, ToBinaryMask, ImageSender, ImageReceiver,
    LatentSender, LatentReceiver, ImpactWildcardProcessor, ImpactWildcardEncode, ImpactSchedulerAdapter,
)
from .modules.impact.detectors import (
    SAMDetectorCombined, SAMDetectorSegmented, BboxDetectorForEach, SegmDetectorForEach, SimpleDetectorForAnimateDiff, SAM2VideoDetectorSEGS,
    SAM3VideoDetectorSEGS, SimpleDetectorForEach, SimpleDetectorForEachPipe, BboxDetectorCombined, SegmDetectorCombined,
)
from .modules.impact.pipe import (
    ToDetailerPipe, ToDetailerPipeSDXL, FromDetailerPipe, FromDetailerPipe_v2, FromDetailerPipe_SDXL, AnyPipeToBasic, ToBasicPipe, FromBasicPipe,
    FromBasicPipe_v2, BasicPipeToDetailerPipe, BasicPipeToDetailerPipeSDXL, DetailerPipeToBasicPipe, EditBasicPipe, EditDetailerPipe,
    EditDetailerPipeSDXL,
)
from .modules.impact.logics import (
    ImpactCompare, ImpactConditionalBranch, ImpactConditionalBranchSelMode, ImpactIfNone, ImpactConvertDataType, ImpactLogicalOperators, ImpactInt,
    ImpactFloat, ImpactBoolean, ImpactValueSender, ImpactValueReceiver, ImpactImageInfo, ImpactLatentInfo, ImpactMinMax, ImpactNeg,
    ImpactConditionalStopIteration, ImpactExecutionOrderController, ImpactListBridge, ImpactQueueTrigger, ImpactQueueTriggerCountdown,
    ImpactSetWidgetValue, ImpactNodeSetMuteState, ImpactControlBridge, ImpactNotEmptySEGS, ImpactSleep, ImpactRemoteBoolean, ImpactRemoteInt,
)
from .modules.impact.util_nodes import (
    MasksToMaskList, MaskListToMaskBatch, ImageListToImageBatch, ImageMaskSwitch, GeneralSwitch, GeneralInversedSwitch, ReencodeLatent,
    ReencodeLatentPipe, ImageBatchToImageList, MakeImageList, MakeImageBatch, MakeAnyList, MakeMaskList, MakeMaskBatch, NthItemOfAnyList,
    StringSelector, StringListToString, WildcardPromptFromString, RemoveNoiseMask, ImpactLogger, ImpactDummyInput,
)
from .modules.impact.segs_nodes import (
    EmptySEGS, MediaPipeFaceMeshToSEGS, MaskToSEGS, MaskToSEGS_for_AnimateDiff, DefaultImageForSEGS, RemoveImageFromSEGS, ControlNetApplySEGS,
    ControlNetApplyAdvancedSEGS, ControlNetClearSEGS, IPAdapterApplySEGS, DecomposeSEGS, AssembleSEGS, DecomposeSEGSList, AssembleSEGSList,
    From_SEG_ELT, Edit_SEG_ELT, Dilate_SEG_ELT, DilateMask, GaussianBlurMask, DilateMaskInSEGS, GaussianBlurMaskInSEGS, SEG_ELT_BBOX_ScaleBy,
    From_SEG_ELT_bbox, From_SEG_ELT_crop_region, Count_Elts_in_SEGS, SegsToCombinedMask, SEGSUpscaler, SEGSUpscalerPipe, SEGSDetailer, SEGSPreview,
    SEGSPreviewCNet, SEGSToImageList, SEGSToMaskList, SEGSToMaskBatch, SEGSToBoundingBoxes, BoundingBoxesToSEGS, SEGSConcat, SEGSPicker, MakeTileSEGS,
    SEGSMerge, SEGSLabelAssign, SEGSLabelFilter, SEGSRangeFilter, SEGSOrderedFilter, SEGSIntersectionFilter, SEGSNMSFilter,
)
from .modules.impact.special_samplers import (
    KSamplerProvider, TwoSamplersForMask, TiledKSamplerProvider, KSamplerAdvancedProvider, TwoAdvancedSamplersForMask,
    NegativeConditioningPlaceholder, KSamplerBasicPipe, KSamplerAdvancedBasicPipe, RegionalSampler, RegionalSamplerAdvanced, CombineRegionalPrompts,
    RegionalPrompt, CombineConditionings, ConcatConditionings, GITSSchedulerFuncProvider,
)
from .modules.impact.hf_nodes import (
    HF_TransformersClassifierProvider, SEGS_Classify,
)
from .modules.impact.bridge_nodes import (
    PreviewBridge, PreviewBridgeLatent,
)
from .modules.impact.hook_nodes import (
    PreviewDetailerHookProvider, BlackPatchRetryHookProvider, LamaRemoverDetailerHookProvider, DetailerAdvancedOptionsHook,
    SEGSOrderedFilterDetailerHookProvider, SEGSRangeFilterDetailerHookProvider, SEGSLabelFilterDetailerHookProvider,
)
from .modules.impact.animatediff_nodes import (
    DetailerForEachPipeForAnimateDiff, SEGSPaste, SEGSDetailerForAnimateDiff,
)
from .modules.impact.wildcard_source import LoraWildcardSource, TextWildcardSource

import threading


threading.Thread(target=impact.wildcards.wildcard_load).start()


NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader,
    "CLIPSegDetectorProvider": CLIPSegDetectorProvider,
    "ONNXDetectorProvider": ONNXDetectorProvider,

    "BitwiseAndMaskForEach": BitwiseAndMaskForEach,
    "SubtractMaskForEach": SubtractMaskForEach,

    "DetailerForEach": DetailerForEach,
    "DetailerForEachAutoRetry": DetailerForEachAutoRetry,
    "DetailerForEachDebug": DetailerForEachTest,
    "DetailerForEachPipe": DetailerForEachPipe,
    "DetailerForEachDebugPipe": DetailerForEachTestPipe,
    "DetailerForEachPipeForAnimateDiff": DetailerForEachPipeForAnimateDiff,

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
    "AnyPipeToBasic": AnyPipeToBasic,
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
    "DenoiseScheduleHookProvider": DenoiseScheduleHookProvider,
    "StepsScheduleHookProvider": StepsScheduleHookProvider,
    "CfgScheduleHookProvider": CfgScheduleHookProvider,
    "NoiseInjectionHookProvider": NoiseInjectionHookProvider,
    "UnsamplerHookProvider": UnsamplerHookProvider,
    "CoreMLDetailerHookProvider": CoreMLDetailerHookProvider,
    "PreviewDetailerHookProvider": PreviewDetailerHookProvider,
    "BlackPatchRetryHookProvider": BlackPatchRetryHookProvider,
    "CustomSamplerDetailerHookProvider": CustomSamplerDetailerHookProvider,
    "LamaRemoverDetailerHookProvider": LamaRemoverDetailerHookProvider,

    "DetailerHookCombine": DetailerHookCombine,
    "DetailerAdvancedOptionsHook": DetailerAdvancedOptionsHook,
    "NoiseInjectionDetailerHookProvider": NoiseInjectionDetailerHookProvider,
    "UnsamplerDetailerHookProvider": UnsamplerDetailerHookProvider,
    "DenoiseSchedulerDetailerHookProvider": DenoiseSchedulerDetailerHookProvider,
    "SEGSOrderedFilterDetailerHookProvider": SEGSOrderedFilterDetailerHookProvider,
    "SEGSRangeFilterDetailerHookProvider": SEGSRangeFilterDetailerHookProvider,
    "SEGSLabelFilterDetailerHookProvider": SEGSLabelFilterDetailerHookProvider,
    "VariationNoiseDetailerHookProvider": VariationNoiseDetailerHookProvider,

    "BitwiseAndMask": BitwiseAndMask,
    "SubtractMask": SubtractMask,
    "AddMask": AddMask,
    "MaskRectArea": MaskRectArea,
    "MaskRectAreaAdvanced": MaskRectAreaAdvanced,
    "ImpactSegsAndMask": SegsBitwiseAndMask,
    "ImpactSegsAndMaskForEach": SegsBitwiseAndMaskForEach,
    "EmptySegs": EmptySEGS,
    "ImpactFlattenMask": FlattenMask,

    "MediaPipeFaceMeshToSEGS": MediaPipeFaceMeshToSEGS,
    "MaskToSEGS": MaskToSEGS,
    "MaskToSEGS_for_AnimateDiff": MaskToSEGS_for_AnimateDiff,
    "ToBinaryMask": ToBinaryMask,
    "MasksToMaskList": MasksToMaskList,
    "MaskListToMaskBatch": MaskListToMaskBatch,
    "ImageListToImageBatch": ImageListToImageBatch,
    "SetDefaultImageForSEGS": DefaultImageForSEGS,
    "RemoveImageFromSEGS": RemoveImageFromSEGS,

    "BboxDetectorSEGS": BboxDetectorForEach,
    "SegmDetectorSEGS": SegmDetectorForEach,
    "ONNXDetectorSEGS": BboxDetectorForEach,
    "ImpactSimpleDetectorSEGS_for_AD": SimpleDetectorForAnimateDiff,
    "ImpactSAM2VideoDetectorSEGS": SAM2VideoDetectorSEGS,
    "ImpactSAM3VideoDetectorSEGS": SAM3VideoDetectorSEGS,
    "ImpactSimpleDetectorSEGS": SimpleDetectorForEach,
    "ImpactSimpleDetectorSEGSPipe": SimpleDetectorForEachPipe,
    "ImpactControlNetApplySEGS": ControlNetApplySEGS,
    "ImpactControlNetApplyAdvancedSEGS": ControlNetApplyAdvancedSEGS,
    "ImpactControlNetClearSEGS": ControlNetClearSEGS,
    "ImpactIPAdapterApplySEGS": IPAdapterApplySEGS,

    "ImpactDecomposeSEGS": DecomposeSEGS,
    "ImpactAssembleSEGS": AssembleSEGS,
    "ImpactDecomposeSEGSList": DecomposeSEGSList,
    "ImpactAssembleSEGSList": AssembleSEGSList,
    "ImpactFrom_SEG_ELT": From_SEG_ELT,
    "ImpactEdit_SEG_ELT": Edit_SEG_ELT,
    "ImpactDilate_Mask_SEG_ELT": Dilate_SEG_ELT,
    "ImpactDilateMask": DilateMask,
    "ImpactGaussianBlurMask": GaussianBlurMask,
    "ImpactDilateMaskInSEGS": DilateMaskInSEGS,
    "ImpactGaussianBlurMaskInSEGS": GaussianBlurMaskInSEGS,
    "ImpactScaleBy_BBOX_SEG_ELT": SEG_ELT_BBOX_ScaleBy,
    "ImpactFrom_SEG_ELT_bbox": From_SEG_ELT_bbox,
    "ImpactFrom_SEG_ELT_crop_region": From_SEG_ELT_crop_region,
    "ImpactCount_Elts_in_SEGS": Count_Elts_in_SEGS,

    "BboxDetectorCombined_v2": BboxDetectorCombined,
    "SegmDetectorCombined_v2": SegmDetectorCombined,
    "SegsToCombinedMask": SegsToCombinedMask,

    "KSamplerProvider": KSamplerProvider,
    "TwoSamplersForMask": TwoSamplersForMask,
    "TiledKSamplerProvider": TiledKSamplerProvider,

    "KSamplerAdvancedProvider": KSamplerAdvancedProvider,
    "TwoAdvancedSamplersForMask": TwoAdvancedSamplersForMask,

    "ImpactNegativeConditioningPlaceholder": NegativeConditioningPlaceholder,

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
    "LoraWildcardSource": LoraWildcardSource,
    "TextWildcardSource": TextWildcardSource,

    "SEGSUpscaler": SEGSUpscaler,
    "SEGSUpscalerPipe": SEGSUpscalerPipe,
    "SEGSDetailer": SEGSDetailer,
    "SEGSPaste": SEGSPaste,
    "SEGSPreview": SEGSPreview,
    "SEGSPreviewCNet": SEGSPreviewCNet,
    "SEGSToImageList": SEGSToImageList,
    "ImpactSEGSToMaskList": SEGSToMaskList,
    "ImpactSEGSToMaskBatch": SEGSToMaskBatch,
    "ImpactSEGSToBoundingBoxes": SEGSToBoundingBoxes,
    "ImpactBoundingBoxesToSEGS": BoundingBoxesToSEGS,
    "ImpactSEGSConcat": SEGSConcat,
    "ImpactSEGSPicker": SEGSPicker,
    "ImpactMakeTileSEGS": MakeTileSEGS,
    "ImpactSEGSMerge": SEGSMerge,

    "SEGSDetailerForAnimateDiff": SEGSDetailerForAnimateDiff,

    "ImpactKSamplerBasicPipe": KSamplerBasicPipe,
    "ImpactKSamplerAdvancedBasicPipe": KSamplerAdvancedBasicPipe,

    "ReencodeLatent": ReencodeLatent,
    "ReencodeLatentPipe": ReencodeLatentPipe,

    "ImpactImageBatchToImageList": ImageBatchToImageList,
    "ImpactMakeImageList": MakeImageList,
    "ImpactMakeImageBatch": MakeImageBatch,
    "ImpactMakeAnyList": MakeAnyList,
    "ImpactMakeMaskList": MakeMaskList,
    "ImpactMakeMaskBatch": MakeMaskBatch,
    "ImpactSelectNthItemOfAnyList": NthItemOfAnyList,

    "RegionalSampler": RegionalSampler,
    "RegionalSamplerAdvanced": RegionalSamplerAdvanced,
    "CombineRegionalPrompts": CombineRegionalPrompts,
    "RegionalPrompt": RegionalPrompt,

    "ImpactCombineConditionings": CombineConditionings,
    "ImpactConcatConditionings": ConcatConditionings,

    "ImpactSEGSLabelAssign": SEGSLabelAssign,
    "ImpactSEGSLabelFilter": SEGSLabelFilter,
    "ImpactSEGSRangeFilter": SEGSRangeFilter,
    "ImpactSEGSOrderedFilter": SEGSOrderedFilter,
    "ImpactSEGSIntersectionFilter": SEGSIntersectionFilter,
    "ImpactSEGSNMSFilter": SEGSNMSFilter,

    "ImpactCompare": ImpactCompare,
    "ImpactConditionalBranch": ImpactConditionalBranch,
    "ImpactConditionalBranchSelMode": ImpactConditionalBranchSelMode,
    "ImpactIfNone": ImpactIfNone,
    "ImpactConvertDataType": ImpactConvertDataType,
    "ImpactLogicalOperators": ImpactLogicalOperators,
    "ImpactInt": ImpactInt,
    "ImpactFloat": ImpactFloat,
    "ImpactBoolean": ImpactBoolean,
    "ImpactValueSender": ImpactValueSender,
    "ImpactValueReceiver": ImpactValueReceiver,
    "ImpactImageInfo": ImpactImageInfo,
    "ImpactLatentInfo": ImpactLatentInfo,
    "ImpactMinMax": ImpactMinMax,
    "ImpactNeg": ImpactNeg,
    "ImpactConditionalStopIteration": ImpactConditionalStopIteration,
    "ImpactStringSelector": StringSelector,
    "StringListToString": StringListToString,
    "WildcardPromptFromString": WildcardPromptFromString,
    "ImpactExecutionOrderController": ImpactExecutionOrderController,
    "ImpactListBridge": ImpactListBridge,

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
    "ImpactSEGSClassify": SEGS_Classify,

    "ImpactSchedulerAdapter": ImpactSchedulerAdapter,
    "GITSSchedulerFuncProvider": GITSSchedulerFuncProvider
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMLoader": "SAMLoader (Impact)",

    "BboxDetectorSEGS": "BBOX Detector (SEGS)",
    "SegmDetectorSEGS": "SEGM Detector (SEGS)",
    "ONNXDetectorSEGS": "ONNX Detector (SEGS/legacy) - use BBOXDetector",
    "ImpactSimpleDetectorSEGS_for_AD": "Simple Detector for Video (SEGS)",
    "ImpactSAM2VideoDetectorSEGS": "SAM2 Video Detector (SEGS)",
    "ImpactSAM3VideoDetectorSEGS": "SAM3 Video Detector (SEGS)",
    "ImpactSimpleDetectorSEGS": "Simple Detector (SEGS)",
    "ImpactSimpleDetectorSEGSPipe": "Simple Detector (SEGS/pipe)",
    "ImpactControlNetApplySEGS": "ControlNetApply (SEGS) - DEPRECATED",
    "ImpactControlNetApplyAdvancedSEGS": "ControlNetApply (SEGS)",
    "ImpactIPAdapterApplySEGS": "IPAdapterApply (SEGS)",

    "BboxDetectorCombined_v2": "BBOX Detector (combined)",
    "SegmDetectorCombined_v2": "SEGM Detector (combined)",
    "SegsToCombinedMask": "SEGS to MASK (combined)",
    "MediaPipeFaceMeshToSEGS": "MediaPipe FaceMesh to SEGS",
    "MaskToSEGS": "MASK to SEGS",
    "MaskToSEGS_for_AnimateDiff": "MASK to SEGS for Video",
    "BitwiseAndMaskForEach": "Pixelwise(SEGS & SEGS)",
    "SubtractMaskForEach": "Pixelwise(SEGS - SEGS)",
    "ImpactSegsAndMask": "Pixelwise(SEGS & MASK)",
    "ImpactSegsAndMaskForEach": "Pixelwise(SEGS & MASKS ForEach)",
    "BitwiseAndMask": "Pixelwise(MASK & MASK)",
    "SubtractMask": "Pixelwise(MASK - MASK)",
    "AddMask": "Pixelwise(MASK + MASK)",
    "MaskRectArea": "Mask Rect Area",
    "MaskRectAreaAdvanced": "Mask Rect Area (Advanced)",
    "ImpactFlattenMask": "Flatten Mask Batch",
    "DetailerForEach": "Detailer (SEGS)",
    "DetailerForEachAutoRetry": "Detailer (SEGS) with auto retry",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "DetailerForEachDebug": "DetailerDebug (SEGS)",
    "DetailerForEachDebugPipe": "DetailerDebug (SEGS/pipe)",
    "SEGSDetailerForAnimateDiff": "SEGSDetailer For Video (SEGS/pipe)",
    "DetailerForEachPipeForAnimateDiff": "Detailer For Video (SEGS/pipe)",
    "SEGSUpscaler": "Upscaler (SEGS)",
    "SEGSUpscalerPipe": "Upscaler (SEGS/pipe)",

    "SAMDetectorCombined": "SAMDetector (combined)",
    "SAMDetectorSegmented": "SAMDetector (segmented)",
    "FaceDetailerPipe": "FaceDetailer (pipe)",
    "MaskDetailerPipe": "MaskDetailer (pipe)",

    "FromDetailerPipeSDXL": "FromDetailer (SDXL/pipe)",
    "BasicPipeToDetailerPipeSDXL": "BasicPipe -> DetailerPipe (SDXL)",
    "EditDetailerPipeSDXL": "Edit DetailerPipe (SDXL)",

    "BasicPipeToDetailerPipe": "BasicPipe -> DetailerPipe",
    "DetailerPipeToBasicPipe": "DetailerPipe -> BasicPipe",
    "EditBasicPipe": "Edit BasicPipe",
    "EditDetailerPipe": "Edit DetailerPipe",
    "AnyPipeToBasic": "Any PIPE -> BasicPipe",

    "LatentPixelScale": "Latent Scale (on Pixel Space)",
    "IterativeLatentUpscale": "Iterative Upscale (Latent/on Pixel Space)",
    "IterativeImageUpscale": "Iterative Upscale (Image)",

    "TwoSamplersForMaskUpscalerProvider": "TwoSamplersForMask Upscaler Provider",
    "TwoSamplersForMaskUpscalerProviderPipe": "TwoSamplersForMask Upscaler Provider (pipe)",

    "ReencodeLatent": "Reencode Latent",
    "ReencodeLatentPipe": "Reencode Latent (pipe)",

    "ImpactKSamplerBasicPipe": "KSampler (pipe)",
    "ImpactKSamplerAdvancedBasicPipe": "KSampler (Advanced/pipe)",
    "ImpactSEGSLabelAssign": "SEGS Assign (label)",
    "ImpactSEGSLabelFilter": "SEGS Filter (label)",
    "ImpactSEGSRangeFilter": "SEGS Filter (range)",
    "ImpactSEGSOrderedFilter": "SEGS Filter (ordered)",
    "ImpactSEGSIntersectionFilter": "SEGS Filter (intersection)",
    "ImpactSEGSNMSFilter": "SEGS Filter (non max suppression)",
    "ImpactSEGSConcat": "SEGS Concat",
    "ImpactSEGSToMaskList": "SEGS to Mask List",
    "ImpactSEGSToMaskBatch": "SEGS to Mask Batch",
    "ImpactSEGSToBoundingBoxes": "SEGS to Bounding Boxes",
    "ImpactBoundingBoxesToSEGS": "Bounding Boxes to SEGS",
    "ImpactSEGSPicker": "Picker (SEGS)",
    "ImpactMakeTileSEGS": "Make Tile SEGS",
    "ImpactSEGSMerge": "SEGS Merge",

    "ImpactDecomposeSEGS": "Decompose (SEGS)",
    "ImpactAssembleSEGS": "Assemble (SEGS)",
    "ImpactDecomposeSEGSList": "Decompose (SEGS list)",
    "ImpactAssembleSEGSList": "Assemble (SEGS list)",
    "ImpactFrom_SEG_ELT": "From SEG_ELT",
    "ImpactEdit_SEG_ELT": "Edit SEG_ELT",
    "ImpactFrom_SEG_ELT_bbox": "From SEG_ELT bbox",
    "ImpactFrom_SEG_ELT_crop_region": "From SEG_ELT crop_region",
    "ImpactDilate_Mask_SEG_ELT": "Dilate Mask (SEG_ELT)",
    "ImpactScaleBy_BBOX_SEG_ELT": "ScaleBy BBOX (SEG_ELT)",
    "ImpactCount_Elts_in_SEGS": "Count Elts in SEGS",
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
    "ImpactExecutionOrderController": "Execution Order Controller",
    "ImpactListBridge": "List Bridge",

    "MasksToMaskList": "Mask Batch to Mask List",
    "MaskListToMaskBatch": "Mask List to Mask Batch",
    "ImpactImageBatchToImageList": "Image Batch to Image List",
    "ImageListToImageBatch": "Image List to Image Batch",

    "ImpactMakeImageList": "Make Image List",
    "ImpactMakeImageBatch": "Make Image Batch",
    "ImpactMakeMaskList": "Make Mask List",
    "ImpactMakeMaskBatch": "Make Mask Batch",
    "ImpactMakeAnyList": "Make List (Any)",
    "ImpactSelectNthItemOfAnyList": "Select Nth Item (Any list)",

    "ImpactStringSelector": "String Selector",
    "StringListToString": "String List to String",
    "WildcardPromptFromString": "Wildcard Prompt from String",
    "ImpactIsNotEmptySEGS": "SEGS isn't Empty",
    "SetDefaultImageForSEGS": "Set Default Image for SEGS",
    "RemoveImageFromSEGS": "Remove Image from SEGS",

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
    "SEGSSwitch": "Switch (SEGS/legacy)",

    "SEGSPreviewCNet": "SEGSPreview (CNET Image)",

    "ImpactSchedulerAdapter": "Impact Scheduler Adapter",
    "GITSSchedulerFuncProvider": "GITSScheduler Func Provider",
    "ImpactNegativeConditioningPlaceholder": "Negative Cond Placeholder",
    "LoraWildcardSource": "Lora Wildcard Source / Modes",
    "TextWildcardSource": "Text Wildcard Source / Modes"
}


# NOTE:  Inject directly into EXTENSION_WEB_DIRS instead of WEB_DIRECTORY
#        Provide the js path fixed as ComfyUI-Impact-Pack instead of the path name, making it available for external use

nodes.EXTENSION_WEB_DIRS["ComfyUI-Impact-Pack"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'js')


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
