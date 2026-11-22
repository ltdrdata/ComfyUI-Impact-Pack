"""Fixed inputs for the nodes rewritten on the V3 schema, and a summary of what a node returns for them.

`v1_node_runs.json` holds the summaries each node gave as a V1 node. Before a node is rewritten its
cases are added here and recorded from a checkout that still runs the V1 node:

    IMPACT_COMFY_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI:modules python tests/v1_node_run_cases.py

That adds the cases that have no record and leaves every recorded case as it is. Functions that need a model are
replaced by ones that return their arguments, so a summary shows that every input reaches them unchanged and in order.

The inputs come from `torch.rand` with a seeded generator. PyTorch does not promise the same numbers in every release:
if every case fails at once after an update of PyTorch, look there before looking at the nodes.
"""
import ast
import asyncio
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

RECORD = Path(__file__).resolve().parent / "v1_node_runs.json"
CLIPSEG_INSTALLED = "CLIPSegDetectorProvider#1"


def rand(seed, *shape):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def seg(SEG, seed, box, label, image=True):
    x1, y1, x2, y2 = box
    crop = (max(0, x1 - 4), max(0, y1 - 4), min(64, x2 + 4), min(64, y2 + 4))
    h, w = crop[3] - crop[1], crop[2] - crop[0]
    mask = (rand(seed, h, w) > 0.4).float().numpy()
    return SEG(rand(seed + 1, 1, h, w, 3).numpy() if image else None, mask, 0.5 + seed / 100, crop, box, label, "wrapper" if image else None)



def cases(core, hooks):
    SEGS = ((64, 64), [seg(core.SEG, 1, (8, 8, 30, 34), "face"), seg(core.SEG, 2, (36, 20, 60, 50), "hand", image=False)])
    OTHER = ((64, 64), [seg(core.SEG, 3, (4, 4, 40, 40), "face")])
    EMPTY = ((64, 64), [])
    MASK, MASK2 = rand(4, 1, 64, 64), (rand(5, 1, 64, 64) > 0.5).float()
    PIPE, PIPE2 = ("model", "clip", "vae", "positive", "negative"), ("model2", "clip2", "vae2", "positive2", "negative2")
    DETAILER_PIPE = ("model", "clip", "vae", "positive", "negative", "wildcard", "bbox", "segm", "sam", "hook", "refiner_model", "refiner_clip", "refiner_positive", "refiner_negative")
    LATENT = {"samples": rand(6, 2, 4, 8, 12), "noise_mask": rand(7, 1, 64, 96), "batch_index": [0, 1]}
    THREE = ((64, 64), SEGS[1] + OTHER[1])
    DISTINCT_ORDERS = ((64, 64), [seg(core.SEG, 21 + index, box, f"seg{index}", image=False)
                                  for index, box in enumerate(((16, 32, 24, 44), (28, 10, 46, 24), (14, 28, 30, 46), (38, 30, 50, 40)))])
    BLOBS = torch.zeros(1, 64, 64)
    BLOBS[0, 8:24, 6:30], BLOBS[0, 36:60, 34:58], BLOBS[0, 44:50, 42:48] = 1.0, 0.6, 0.0
    FRAMES = torch.stack([BLOBS[0], BLOBS[0].flip(0), BLOBS[0].flip(1)])
    IMAGE = rand(13, 1, 64, 64, 3)
    IMAGES = rand(16, 2, 64, 64, 3)
    FRAME_SEGS = ((64, 64), [core.SEG(rand(31, 2, 26, 30, 3), (rand(32, 2, 26, 30) > 0.4).float(), 0.8, (4, 4, 34, 30), (8, 8, 30, 26), "face", None),
                             core.SEG(rand(33, 2, 20, 24, 3).numpy(), (rand(34, 3, 20, 24) > 0.6).float(), 0.6, (36, 40, 60, 60), (40, 44, 56, 56), "hand", None)])
    # A face mesh in the colours comfyui_controlnet_aux draws it in, each part of another size.
    PARTS = {"face": ((6, 58, 10, 54), (0x0A, 0xC8, 0x0A)), "mouth": ((40, 50, 22, 42), (0x0A, 0xB4, 0x0A)),
             "left_eyebrow": ((9, 13, 14, 26), (0xB4, 0xDC, 0x0A)), "left_eye": ((16, 26, 14, 28), (0xB4, 0xC8, 0x0A)), "left_pupil": ((19, 23, 18, 22), (0xFA, 0xC8, 0x0A)),
             "right_eyebrow": ((9, 14, 36, 52), (0x0A, 0xDC, 0xB4)), "right_eye": ((16, 24, 36, 48), (0x0A, 0xC8, 0xB4)), "right_pupil": ((18, 21, 40, 45), (0x0A, 0xC8, 0xFA))}
    MESH = torch.zeros(1, 64, 64, 3)
    for (y1, y2, x1, x2), colour in PARTS.values():
        MESH[0, y1:y2, x1:x2] = (torch.tensor(colour) + 0.5) / 255
    WANTED_PARTS = (("face", "left_eyebrow", "left_pupil", "right_eye"), ("mouth", "left_eyebrow", "right_eyebrow", "right_eye"),
                    ("left_eye", "left_pupil", "right_eyebrow", "right_eye"), ("right_pupil",))
    TILES = dict(images=rand(14, 1, 96, 128, 3), bbox_size=64, crop_factor=1.5, min_overlap=8, filter_segs_dilation=4)
    SAM = dict(sam_model="sam", segs=SEGS, image=rand(8, 1, 64, 64, 3), detection_hint="center-1", dilation=3, threshold=0.9, bbox_expansion=2, mask_hint_threshold=0.6, mask_hint_use_negative="Small")
    CASES = {
        "SAMDetectorCombined": [SAM], "SAMDetectorSegmented": [SAM],
        "SEGSOrderedFilterDetailerHookProvider": [dict(target="width", order=False, take_start=1, take_count=2)],
        "SEGSRangeFilterDetailerHookProvider": [dict(target="x1", mode=False, min_value=3, max_value=90)],
        "LamaRemoverDetailerHookProvider": [dict(mask_threshold=200, gaussblur_radius=4, skip_sampling=False)],
        "BlackPatchRetryHookProvider": [dict(mean_thresh=12, var_thresh=7)],
        "CLIPSegDetectorProvider": [dict(text="face, hand", blur=5.0, threshold=0.3, dilation_factor=2)] * 2,
        "LatentPixelScale": [dict(samples=LATENT, scale_method="lanczos", scale_factor=1.25, vae="vae", use_tiled_vae=True),
                             dict(samples=LATENT, scale_method="area", scale_factor=2.0, vae="vae", use_tiled_vae=False, upscale_model_opt="upscaler")],
        "DenoiseSchedulerDetailerHookProvider": [dict(schedule_for_cycle="simple", target_denoise=0.45)],
        "CoreMLDetailerHookProvider": [dict(mode="512x768")],
        "CustomSamplerDetailerHookProvider": [dict(sampler="sampler")],
        "CfgScheduleHookProvider": [dict(schedule_for_iteration="simple", target_cfg=5.5), dict(schedule_for_iteration="other", target_cfg=5.5)],
        "DenoiseScheduleHookProvider": [dict(schedule_for_iteration="simple", target_denoise=0.35), dict(schedule_for_iteration="other", target_denoise=0.35)],
        "StepsScheduleHookProvider": [dict(schedule_for_iteration="simple", target_steps=33), dict(schedule_for_iteration="other", target_steps=33)],
        "DetailerHookCombine": [dict(hook1=hooks.SimpleDetailerDenoiseSchedulerHook(0.2), hook2=hooks.DetailerHook())],
        "PixelKSampleHookCombine": [dict(hook1=hooks.SimpleCfgScheduleHook(4.0), hook2=hooks.SimpleStepsScheduleHook(9))],
        "ImpactSegsAndMaskForEach": [dict(segs=SEGS, masks=rand(9, 2, 64, 64))],
        "BitwiseAndMaskForEach": [dict(base_segs=SEGS, mask_segs=OTHER)],
        "SubtractMaskForEach": [dict(base_segs=SEGS, mask_segs=OTHER)],
        "ToBinaryMask": [dict(mask=MASK, threshold=120)],
        "ImpactFlattenMask": [dict(masks=rand(10, 3, 64, 64))],
        "BitwiseAndMask": [dict(mask1=MASK, mask2=MASK2)], "SubtractMask": [dict(mask1=MASK, mask2=MASK2)], "AddMask": [dict(mask1=MASK, mask2=MASK2)],
        "ImpactIsNotEmptySEGS": [dict(segs=SEGS), dict(segs=EMPTY)],
        "ImpactFloat": [dict(value=2.5)], "ImpactBoolean": [dict(value=True)],
        "ImpactImageInfo": [dict(value=rand(11, 2, 48, 64, 3))], "ImpactLatentInfo": [dict(value=LATENT)],
        "ToBasicPipe": [dict(model="model", clip="clip", vae="vae", positive="positive", negative="negative")],
        "FromBasicPipe": [dict(basic_pipe=PIPE)], "FromBasicPipe_v2": [dict(basic_pipe=PIPE)],
        "DetailerPipeToBasicPipe": [dict(detailer_pipe=DETAILER_PIPE)],
        "ImpactSEGSToMaskBatch": [dict(segs=SEGS)], "ImpactCount_Elts_in_SEGS": [dict(segs=SEGS), dict(segs=EMPTY)],
        "ImpactFrom_SEG_ELT": [dict(seg_elt=SEGS[1][0]), dict(seg_elt=SEGS[1][1])],
        "ImpactFrom_SEG_ELT_bbox": [dict(bbox=(8.0, 8.5, 30, 34)), dict(bbox=np.array([1, 2, 3, 4]))],
        "ImpactFrom_SEG_ELT_crop_region": [dict(crop_region=(4, 4, 34, 38)), dict(crop_region=[4, 4, 34, 38])],
        "ImpactDilateMask": [dict(mask=MASK2, dilation=5), dict(mask=MASK2, dilation=-3)],
        "ImpactGaussianBlurMask": [dict(mask=MASK2, kernel_size=7, sigma=3.0), dict(mask=rand(12, 1, 1, 64, 64), kernel_size=5, sigma=2.0)],
        "ImpactDilateMaskInSEGS": [dict(segs=SEGS, dilation=4), dict(segs=EMPTY, dilation=4)],
        "ImpactGaussianBlurMaskInSEGS": [dict(segs=SEGS, kernel_size=5, sigma=2.0)],
        "ImpactDilate_Mask_SEG_ELT": [dict(seg_elt=SEGS[1][0], dilation=6)],
        "EmptySegs": [dict()],
        "SegsToCombinedMask": [dict(segs=SEGS), dict(segs=EMPTY)],
        "ImpactControlNetClearSEGS": [dict(segs=SEGS)],
        "RemoveImageFromSEGS": [dict(segs=SEGS), dict(segs=EMPTY)],
        "GITSSchedulerFuncProvider": [dict(coeff=1.25, denoise=0.8)],
        "ImpactNegativeConditioningPlaceholder": [dict()],
        "RemoveNoiseMask": [dict(samples=LATENT)],
        "ReencodeLatentPipe": [dict(samples=LATENT, tile_mode="Both", input_basic_pipe=PIPE, output_basic_pipe=PIPE2)],
        "MaskToSEGS": [dict(mask=BLOBS, combined=False, crop_factor=1.5, bbox_fill=False, drop_size=3),
                       dict(mask=BLOBS, combined=True, crop_factor=2.0, bbox_fill=True, drop_size=3, contour_fill=False),
                       dict(mask=BLOBS.unsqueeze(0), combined=False, crop_factor=1.5, bbox_fill=True, drop_size=20, contour_fill=True)],
        "MaskToSEGS_for_AnimateDiff": [dict(mask=FRAMES, combined=False, crop_factor=1.5, bbox_fill=False, drop_size=3, contour_fill=True),
                                       dict(mask=BLOBS, combined=False, crop_factor=1.5, bbox_fill=False, drop_size=3),
                                       dict(mask=BLOBS, combined=False, crop_factor=1.5, bbox_fill=False, drop_size=3, contour_fill=True),
                                       dict(mask=BLOBS, combined=True, crop_factor=1.5, bbox_fill=False, drop_size=3, contour_fill=True),
                                       dict(mask=FRAMES, combined=True, crop_factor=1.5, bbox_fill=False, drop_size=3)],
        "MediaPipeFaceMeshToSEGS": [dict(image=MESH, crop_factor=1.5 + case, bbox_fill=case == 1, crop_min_size=10, drop_size=1, dilation=case,
                                         **{part: part in wanted for part in PARTS}) for case, wanted in enumerate(WANTED_PARTS)],
        "ImpactMakeTileSEGS": [TILES, {**TILES, "mask_irregularity": 0, "irregular_mask_mode": "Reuse quality", "filter_in_segs_opt": SEGS, "filter_out_segs_opt": OTHER},
                               {**TILES, "bbox_size": 512, "min_overlap": 300, "filter_in_segs_opt": EMPTY}],
        "SEGSPaste": [dict(image=IMAGE, segs=SEGS, feather=3, alpha=200), dict(image=IMAGE, segs=SEGS, feather=0, ref_image_opt=rand(15, 1, 64, 64, 3)),
                      dict(image=IMAGES, segs=FRAME_SEGS, feather=2, alpha=255), dict(image=IMAGES, segs=FRAME_SEGS, feather=0, ref_image_opt=rand(35, 2, 64, 64, 3))],
        "SetDefaultImageForSEGS": [dict(segs=SEGS, image=IMAGE, override=True), dict(segs=SEGS, image=IMAGE, override=False), dict(segs=EMPTY, image=IMAGE, override=True),
                                   dict(segs=FRAME_SEGS, image=IMAGES, override=True), dict(segs=((64, 64), SEGS[1][1:]), image=IMAGES, override=False)],
        "ImpactSegsAndMask": [dict(segs=SEGS, mask=MASK2)],
        "ImpactSEGSLabelAssign": [dict(segs=SEGS, labels="eye, mouth"), dict(segs=SEGS, labels="only")],
        "ImpactScaleBy_BBOX_SEG_ELT": [dict(seg=SEGS[1][0], scale_by=1.5), dict(seg=SEGS[1][1], scale_by=0.5)],
        "ImpactSEGSOrderedFilter": [dict(segs=THREE, target="area(=w*h)", order=True, take_start=0, take_count=1),
                                    dict(segs=THREE, target="x1", order=False, take_start=1, take_count=1),
                                    dict(segs=THREE, target="none", order=True, take_start=2, take_count=5)]
                                   + [dict(segs=DISTINCT_ORDERS, target=target, order=True, take_start=1, take_count=2)
                                      for target in ("area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "confidence", "unknown")],
        "ImpactSEGSRangeFilter": [dict(segs=THREE, target="width", mode=True, min_value=0, max_value=31),
                                  dict(segs=THREE, target="confidence(0-100)", mode=False, min_value=51, max_value=52),
                                  dict(segs=THREE, target="length_percent", mode=True, min_value=100, max_value=115),
                                  dict(segs=THREE, target="area(=w*h)", mode=True, min_value=1000, max_value=1300),
                                  dict(segs=THREE, target="height", mode=True, min_value=34, max_value=38),
                                  dict(segs=THREE, target="x1", mode=True, min_value=20, max_value=40),
                                  dict(segs=THREE, target="y1", mode=True, min_value=10, max_value=20),
                                  dict(segs=THREE, target="x2", mode=True, min_value=60, max_value=64),
                                  dict(segs=THREE, target="y2", mode=True, min_value=50, max_value=60),
                                  dict(segs=THREE, target="unknown", mode=True, min_value=0, max_value=64)],
        "ImpactSEGSIntersectionFilter": [dict(segs1=SEGS, segs2=OTHER, ioa_threshold=0.5), dict(segs1=SEGS, segs2=OTHER, ioa_threshold=0.13)],
        "ImpactSEGSNMSFilter": [dict(segs=THREE, iou_threshold=0.2), dict(segs=THREE, iou_threshold=0.9),
                                dict(segs=((64, 64), SEGS[1][:1] * 2), iou_threshold=1.0)],
        "ImpactSEGSMerge": [dict(segs=THREE)],
        "ImpactNeg": [dict(value=True), dict(value=False)],
        "ImpactLogicalOperators": [dict(operator=operator, bool_a=bool_a, bool_b=bool_b)
                                   for bool_a in (True, False) for bool_b in (False, True) for operator in ("and", "or", "xor")],
        "EditBasicPipe": [dict(basic_pipe=PIPE), dict(basic_pipe=PIPE, model="model2", negative="negative2"),
                          dict(basic_pipe=PIPE, **dict(zip(("model", "clip", "vae", "positive", "negative"), PIPE2)))],
        "ImpactStringSelector": [dict(strings="first\nsecond\nthird", multiline=False, select=4), dict(strings="#one\nline\n#two\nmore", multiline=True, select=3),
                                 dict(strings="plain\nline\n#two\nmore", multiline=True, select=2)],
        "RegionalPrompt": [dict(mask=MASK2, advanced_sampler="sampler"),
                           dict(mask=MASK2, advanced_sampler="sampler", variation_seed=7, variation_strength=0.4, variation_method="slerp")],
    }
    return CASES


@contextmanager
def without_models(core, util_nodes):
    def named(name):
        return lambda *args, **kwargs: (name, args, kwargs)

    with patch.object(core, "make_sam_mask", named("make_sam_mask")), \
            patch.object(core, "make_sam_mask_segmented", lambda *a, **k: (("combined", a, k), ("batch", a, k))), \
            patch.object(core, "latent_upscale_on_pixel_space2", lambda *a, **k: (("latent", a, k), ("image", a, k))), \
            patch.object(core, "latent_upscale_on_pixel_space_with_model2", lambda *a, **k: (("latent with model", a, k), ("image with model", a, k))), \
            patch.object(util_nodes.ReencodeLatent, "doit", lambda self, *a, **k: (("reencoded", a, k),)):
        yield


def summary(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        data = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value, dtype=np.float64).reshape(-1)
        weights = 1.0 + (np.arange(len(data), dtype=np.int64) * 2654435761 % 4096) / 4096.0
        return {type(value).__name__: [list(value.shape), str(value.dtype), float(data.sum()), float((data * weights).sum())]}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {type(value).__name__: {k: summary(v) for k, v in value._asdict().items()}}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [summary(v) for v in value]}
    if isinstance(value, dict):
        return {"dict": {str(k): summary(v) for k, v in value.items()}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return [type(value).__name__, value]
    if isinstance(value, np.generic):
        return [type(value).__name__, value.item()]
    if callable(value) and not hasattr(value, "__dict__") or type(value).__name__ == "function":
        return "function " + value.__qualname__.split(".")[-1]
    return {"object " + type(value).__name__: {k: summary(v) for k, v in sorted(vars(value).items())}}


def same(new, old):
    if isinstance(new, float) and isinstance(old, float):
        return abs(new - old) <= 1e-6 * max(1.0, abs(new), abs(old))
    if isinstance(new, dict) and isinstance(old, dict):
        return list(new) == list(old) and all(same(new[k], old[k]) for k in new)
    if isinstance(new, list) and isinstance(old, list):
        return len(new) == len(old) and all(same(a, b) for a, b in zip(new, old))
    return type(new) is type(old) and new == old


def attempt(call):
    try:
        return summary(call())
    except Exception as error:
        return {"raised": [type(error).__name__, str(error)]}


def run(case, node, inputs, execution):
    listed = {name: [value] for name, value in inputs.items()}
    with patch.dict(execution.nodes.NODE_CLASS_MAPPINGS, {"CLIPSeg": object} if case == CLIPSEG_INSTALLED else {}):
        result = {"executor": attempt(lambda: asyncio.run(execution.get_output_data("prompt", "1", node(), listed))[0]),
                  "doit": attempt(lambda: node().doit(**inputs))}
    return json.loads(json.dumps(result))


def node_classes(root):
    init = ast.parse((root / "__init__.py").read_text())
    modules = [importlib.import_module("impact." + node.module.rsplit(".", 1)[-1]) for node in init.body
               if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("modules.impact.")]
    assert Path(modules[0].__file__).resolve().parents[2] == root
    mapping = next(node.value for node in init.body if isinstance(node, ast.Assign) and node.targets[0].id == "NODE_CLASS_MAPPINGS")
    return {key.value: next(getattr(module, value.id) for module in reversed(modules) if hasattr(module, value.id))
            for key, value in zip(mapping.keys, mapping.values)}


def record():
    sys.path.insert(0, os.environ["IMPACT_COMFY_ROOT"])
    from comfy.cli_args import args
    args.cpu = True
    import execution
    from aiohttp import web
    from server import PromptServer
    from types import SimpleNamespace
    PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None)
    from impact import impact_server  # noqa: F401
    from impact import core, hooks, util_nodes
    classes = node_classes(Path(core.__file__).resolve().parents[2])
    recorded = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    added = []
    with without_models(core, util_nodes):
        for node_id, runs in cases(core, hooks).items():
            for index, inputs in enumerate(runs):
                case = f"{node_id}#{index}"
                if case not in recorded:
                    assert not hasattr(classes[node_id], "define_schema"), f"{node_id} is rewritten already: its V1 node cannot be recorded any more"
                    recorded[case] = run(case, classes[node_id], inputs, execution)
                    added.append(case)
    RECORD.write_text("{\n" + ",\n".join(f" {json.dumps(case)}: {json.dumps(recorded[case])}" for case in sorted(recorded)) + "\n}\n")
    print(f"{len(added)} cases recorded, {len(recorded)} in {RECORD.name}")


if __name__ == "__main__":
    record()
