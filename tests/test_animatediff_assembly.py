"""AnimateDiff tensor assembly with real Impact/ComfyUI image operations.

Run separately from the wildcard tests, which stub Impact dependencies:
COMFYUI_ROOT=/path/to/ComfyUI python -m pytest tests/test_animatediff_assembly.py -v
VAE and sampling are controlled; no model weights, GPU or server are required.
"""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def core():
    comfy_root = Path(os.environ.get("COMFYUI_ROOT", ROOT.parents[1]))
    assert (comfy_root / "nodes.py").is_file(), "Set COMFYUI_ROOT to a ComfyUI checkout with dependencies installed"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(comfy_root))
        patch.syspath_prepend(str(ROOT / "modules"))
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        yield importlib.import_module("impact.core")


def case(core, monkeypatch, frames=4, mask=None, *, dtype=torch.float32, size=(48, 32)):
    width, height = size
    # Distinct, constant RGB colors have exact resize results without duplicating
    # the production conversion/resampling algorithm. Even bytes also halve exactly.
    colors = np.arange(1, frames * 3 + 1, dtype=np.float32).reshape(frames, 1, 1, 3) * 4 / 255
    images = np.broadcast_to(colors, (frames, 16, 24, 3)).copy()
    latent_height, latent_width = height // 8, width // 8
    encoded = torch.arange(frames * 4 * latent_height * latent_width, dtype=dtype).reshape(frames, 4, latent_height, latent_width) / 512
    decoded = np.broadcast_to(colors[..., ::-1], (frames, height, width, 3)).copy()
    vae = SimpleNamespace(
        encode=Mock(side_effect=[frame[None].clone() for frame in encoded]),
        decode=Mock(side_effect=[torch.from_numpy(frame[None]).clone() for frame in decoded]),
    )

    def sample(*args, **kwargs):
        latent = args[8]
        return dict(latent, samples=latent["samples"] + 0.25)

    sampler = Mock(side_effect=sample)
    monkeypatch.setattr(core.impact_sampling, "ksampler_wrapper", sampler)
    state = SimpleNamespace(images=images, encoded=encoded, decoded=decoded, vae=vae, sampler=sampler)
    state.arguments = dict(image_frames=images, model=SimpleNamespace(model_options={}),
                           clip=object(), vae=vae, guide_size=32, guide_size_for_bbox=False,
                           max_size=64, bbox=(0, 0, 24, 16), seed=632, steps=3, cfg=2.5,
                           sampler_name="euler", scheduler="normal", positive=[], negative=[],
                           denoise=0.4, noise_mask=mask if mask is not None else torch.ones(frames, 16, 24))
    return state


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("frames,mask_count,layout", [
    (1, 1, "2d"), (4, 1, "2d"), (4, 1, "batch"),
    (4, 4, "batch"), (4, 4, "nhwc"), (4, 2, "batch"), (4, 6, "batch"),
])
def test_pixels_shapes_order_and_inputs_are_preserved(core, monkeypatch, dtype, frames, mask_count, layout):
    masks = torch.zeros(mask_count, 16, 24, dtype=dtype)
    for index in range(mask_count):
        masks[index, 2:12, index * 4:index * 4 + 3] = 0.5
    if mask_count not in (1, frames):
        # The first mask is empty so this also exercises the existing union fallback
        # without depending on its unrelated first-mask quantization behavior.
        masks[0] = 0
    if layout == "2d":
        supplied = masks[0].numpy()
    elif layout == "nhwc":
        # Cropped/transposed masks need not be contiguous in memory.
        supplied = masks.transpose(1, 2).contiguous().transpose(1, 2)[..., None]
        assert not supplied.is_contiguous()
    else:
        supplied = masks
    before = torch.as_tensor(supplied).clone()
    state = case(core, monkeypatch, frames, supplied, dtype=dtype)
    original_images = state.images.copy()
    output, cnet_images = core.enhance_detail_for_animatediff(**state.arguments)

    expected_mask = torch.nn.functional.interpolate(masks[:, None], (32, 48), mode="bilinear", align_corners=False)[:, 0]
    if mask_count not in (1, frames):
        # Non-overlapping soft regions become one binary mask, broadcast to frames.
        expected_mask = ((expected_mask[1:] * 255).to(torch.uint8).amax(dim=0) / 255 > 0.1).float()
        expected_mask = expected_mask.expand(frames, -1, -1)
    state.sampler.assert_called_once()
    latent = state.sampler.call_args.args[8]
    torch.testing.assert_close(latent["noise_mask"], expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(latent["samples"], state.encoded, rtol=0, atol=0)
    expected_inputs = torch.from_numpy(original_images[:, :1, :1]).expand(-1, 32, 48, -1)
    assert state.vae.encode.call_count == state.vae.decode.call_count == frames
    for index in range(frames):
        torch.testing.assert_close(state.vae.encode.call_args_list[index].args[0], expected_inputs[index:index + 1], rtol=0, atol=0)
        torch.testing.assert_close(state.vae.decode.call_args_list[index].args[0], state.encoded[index:index + 1] + 0.25, rtol=0, atol=0)
    expected_output = torch.from_numpy(state.decoded[:, :1, :1]).expand(-1, 16, 24, -1)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert cnet_images is None
    torch.testing.assert_close(torch.as_tensor(supplied), before, rtol=0, atol=0)
    np.testing.assert_array_equal(state.images, original_images)


@pytest.mark.parametrize("count", [4, 8, 16])
@pytest.mark.parametrize("batches", ["masks", "latents", "both"])
def test_pre_sampling_concatenation_copies_at_most_one_batch(core, monkeypatch, count, batches):
    frames = 1 if batches == "masks" else count
    mask_count = 1 if batches == "latents" else count
    state = case(core, monkeypatch, frames, torch.ones(mask_count, 16, 24))

    class SamplingBoundary(Exception):
        pass

    def stop_before_sampling(*args, **kwargs):
        raise SamplingBoundary

    monkeypatch.setattr(core.impact_sampling, "ksampler_wrapper", stop_before_sampling)
    # Measure real allocations instead of elapsed time or a particular loop spelling.
    # The controlled VAE allocates no concatenations; per-image resize uses batch=1.
    # Baseline: PyTorch 2.10 CPU. Verify profiler accounting after upgrades.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        with pytest.raises(SamplingBoundary):
            core.enhance_detail_for_animatediff(**state.arguments)
    copied_bytes = sum(event.cpu_memory_usage for event in profile.events() if event.name == "aten::cat")
    # Shared masks/single frames require no concatenation. Test the batches
    # separately so savings in one cannot conceal repeated copies in the other.
    mask_elements = mask_count * 32 * 48 if mask_count > 1 else 0
    latent_elements = frames * 4 * 4 * 6 if frames > 1 else 0
    batch_bytes = (mask_elements + latent_elements) * state.encoded.element_size()
    assert copied_bytes <= batch_bytes, f"Pre-sampling concatenation copied {copied_bytes} bytes; one batch is {batch_bytes}"


def test_mismatched_soft_masks_keep_existing_bytewise_union(core, monkeypatch):
    masks = np.zeros((3, 16, 24), dtype=np.float32)
    masks[0, 2:6, 1:5] = 1.0
    masks[1, 2:6, 8:12] = 16.25 / 255
    masks[2, 2:6, 8:12] = 10.25 / 255
    masks[1, 8:12, 1:5] = 25.25 / 255
    masks[2, 8:12, 8:12] = 26.25 / 255
    original = masks.copy()
    state = case(core, monkeypatch, mask=masks, size=(24, 16))
    state.arguments["guide_size"] = 16
    core.enhance_detail_for_animatediff(**state.arguments)
    expected = torch.zeros(4, 16, 24)
    # 16 | 10 = 26, above the 0.1 binary threshold; max(16, 10) is below.
    expected[:, 2:6, 8:12] = 1
    expected[:, 8:12, 8:12] = 1
    # Preserve the existing unscaled first-mask quantization as well. This
    # assembly change does not repair the fallback's separate arithmetic.
    torch.testing.assert_close(state.sampler.call_args.args[8]["noise_mask"], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(masks, original)


@pytest.mark.parametrize("already_patched", [False, True])
def test_default_feather_preserves_mask_pixels_and_model_selection(core, monkeypatch, already_patched):
    from torchvision.transforms.functional import gaussian_blur

    masks = torch.zeros(4, 16, 24)
    for index in range(4):
        masks[index, 2:12, index * 4:index * 4 + 3] = 0.75
    before = masks.clone()
    state = case(core, monkeypatch, mask=masks.numpy())
    state.arguments["noise_mask_feather"] = 20  # Node default.
    model = state.arguments["model"]
    if already_patched:
        model.model_options["denoise_mask_function"] = object()
    patched_model = SimpleNamespace(model_options={"denoise_mask_function": object()})
    apply_diffusion = Mock(return_value=patched_model)
    monkeypatch.setattr(core.utils, "apply_differential_diffusion", apply_diffusion)
    core.enhance_detail_for_animatediff(**state.arguments)
    # A 16-pixel crop limits the default feather kernel to 9 pixels.
    blurred = gaussian_blur(before[:, None], [9, 9], [10.0, 10.0])
    expected = torch.nn.functional.interpolate(blurred, (32, 48), mode="bilinear", align_corners=False)[:, 0]
    torch.testing.assert_close(state.sampler.call_args.args[8]["noise_mask"], expected, rtol=0, atol=0)
    if already_patched:
        apply_diffusion.assert_not_called()
    else:
        apply_diffusion.assert_called_once_with(model)
    assert state.sampler.call_args.args[0] is (model if already_patched else patched_model)
    torch.testing.assert_close(masks, before, rtol=0, atol=0)


def test_callbacks_and_sampler_arguments_are_preserved(core, monkeypatch):
    state = case(core, monkeypatch, size=(40, 24))
    custom_sampler, scheduler_func = object(), object()
    refiner = [object() for _ in range(4)]
    conditioning = [object(), object()]
    control_images = [object()]

    class Hook:
        def touch_scaled_size(self, width, height):
            assert (width, height) == (48, 32)
            return 40, 24

        def get_custom_sampler(self):
            return custom_sampler

        def post_encode(self, latent):
            torch.testing.assert_close(latent["samples"], state.encoded, rtol=0, atol=0)
            assert latent["noise_mask"].shape == (4, 24, 40)
            return dict(latent, samples=latent["samples"] + 1)

        def pre_decode(self, latent):
            torch.testing.assert_close(latent["samples"], state.encoded + 1.25, rtol=0, atol=0)
            return dict(latent, samples=latent["samples"] + 2)

        def post_decode(self, images):
            torch.testing.assert_close(images, torch.from_numpy(state.decoded), rtol=0, atol=0)
            return images * 0.5

    class Control:
        def apply(self, positive, negative, images, mask, *, use_acn):
            assert positive is state.arguments["positive"] and negative is state.arguments["negative"]
            assert use_acn is True
            torch.testing.assert_close(images, torch.from_numpy(state.images))
            torch.testing.assert_close(mask, state.arguments["noise_mask"])
            return *conditioning, control_images

    state.arguments.update(detailer_hook=Hook(), control_net_wrapper=Control(),
                           refiner_ratio=0.3, refiner_model=refiner[0], refiner_clip=refiner[1],
                           refiner_positive=refiner[2], refiner_negative=refiner[3], scheduler_func=scheduler_func)
    output, images = core.enhance_detail_for_animatediff(**state.arguments)
    assert images is control_images
    state.sampler.assert_called_once()
    args, kwargs = state.sampler.call_args
    assert args[:8] == (state.arguments["model"], 632, 3, 2.5, "euler", "normal", *conditioning)
    assert args[9:] == (0.4, 0.3, *refiner)
    assert kwargs == dict(scheduler_func=scheduler_func, sampler_opt=custom_sampler)
    expected_inputs = torch.from_numpy(state.images[:, :1, :1]).expand(-1, 24, 40, -1)
    assert state.vae.encode.call_count == state.vae.decode.call_count == 4
    for index in range(4):
        torch.testing.assert_close(state.vae.encode.call_args_list[index].args[0], expected_inputs[index:index + 1], rtol=0, atol=0)
        torch.testing.assert_close(state.vae.decode.call_args_list[index].args[0], state.encoded[index:index + 1] + 3.25, rtol=0, atol=0)
    expected_output = torch.from_numpy(state.decoded[:, :1, :1]).expand(-1, 16, 24, -1) * 0.5
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


def test_encode_failure_never_samples_a_partial_batch(core, monkeypatch):
    state = case(core, monkeypatch)
    state.vae.encode.side_effect = [state.encoded[:1], RuntimeError("controlled VAE failure")]
    with pytest.raises(RuntimeError, match="controlled VAE failure"):
        core.enhance_detail_for_animatediff(**state.arguments)
    assert state.vae.encode.call_count == 2
    state.sampler.assert_not_called()
    state.vae.decode.assert_not_called()


@pytest.mark.parametrize("guide,max_size,size", [(16, 96, (48, 32)), (64, 72, (72, 48)), (4, 64, (24, 16))])
def test_bbox_guide_clamp_and_force_inpaint(core, monkeypatch, guide, max_size, size):
    state = case(core, monkeypatch, size=size)
    state.arguments.update(bbox=(8, 4, 16, 12), guide_size_for_bbox=True,
                           guide_size=guide, max_size=max_size)
    output, _ = core.enhance_detail_for_animatediff(**state.arguments)
    width, height = size
    assert state.sampler.call_args.args[8]["noise_mask"].shape == (4, height, width)
    expected = torch.from_numpy(state.images[:, :1, :1]).expand(-1, height, width, -1)
    for index, call in enumerate(state.vae.encode.call_args_list):
        torch.testing.assert_close(call.args[0], expected[index:index + 1], rtol=0, atol=0)
    assert output.shape == (4, 16, 24, 3)


@pytest.fixture(scope="module")
def detailer_nodes(core):
    from aiohttp import web
    from server import PromptServer

    # Node imports register HTTP handlers. No server or requests are needed here.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=Mock()), raising=False)
        from impact import animatediff_nodes, detectors, segs_nodes
        yield animatediff_nodes, detectors, segs_nodes


@pytest.mark.parametrize("source", ["mask", "Pivot SEGS", "Don't combine", "Combine neighboring frames", "tiles", "tiles-control"])
def test_tutorial_segs_to_detailer_to_paste(core, detailer_nodes, monkeypatch, source):
    """Exercise tutorial producers/callers; substitute only learned model boundaries."""
    from torchvision.transforms.functional import gaussian_blur

    animatediff, detectors, segs_nodes = detailer_nodes
    frames = torch.tensor([0.1, 0.2, 0.3]).reshape(3, 1, 1, 1).expand(3, 64, 96, 3).clone()
    # Coordinate ramps expose wrong crop pixels; blue identifies each frame.
    frames[..., 0] = torch.arange(96) / 128
    frames[..., 1] = torch.arange(64)[:, None] / 128
    original = frames.clone()
    masks = torch.zeros(3, 64, 96)
    for index in range(3):
        masks[index, 16:40, 8 + index * 8:32 + index * 8] = 1
    tiled = source.startswith("tiles")
    if tiled:
        segs = segs_nodes.MakeTileSEGS.doit(frames, 64, 1.5, 8, 0)[0]
        assert len(segs[1]) == 2  # Overlapping tiles also exercise sequential paste.
    elif source == "mask":
        segs = segs_nodes.MaskToSEGS.doit(masks[:1], False, 1.5, False, 10)[0]
    else:
        detected = [segs_nodes.MaskToSEGS.doit(mask, False, 1.5, False, 10)[0] for mask in masks]
        detector = SimpleNamespace(detect=Mock(side_effect=detected))
        segs = detectors.SimpleDetectorForAnimateDiff.detect(
            detector, frames, 0.5, 0, 1.5, 10, 0.5, 0, 0, 0.7,
            masking_mode=source, segs_pivot="Combined mask")[0]
        assert detector.detect.call_count == 3
    assert segs[1]
    before_masks = [torch.as_tensor(seg.cropped_mask).clone() for seg in segs[1]]
    if source == "tiles-control":
        # Use the real SEGS ControlNet wrapper from the tutorial. Only the
        # external Advanced-ControlNet conditioning operation is substituted.
        control = object()
        conditioning = [object(), object()]
        apply_control = Mock(return_value=(*conditioning, None))
        monkeypatch.setitem(core.nodes.NODE_CLASS_MAPPINGS, "ACN_AdvancedControlNetApply",
                            Mock(return_value=SimpleNamespace(apply_controlnet=apply_control)))
        control_image = torch.zeros_like(frames)
        control_image[:, :, ::2, 0] = 1
        control_image[:, ::2, :, 1] = 1
        control_before = control_image.clone()
        segs = segs_nodes.ControlNetApplySEGS.doit(segs, control, 1.0, control_image=control_image)[0]

    encoded_count = 0

    def encode(image):
        nonlocal encoded_count
        encoded_count += 1
        return torch.full((1, 4, image.shape[1] // 8, image.shape[2] // 8), encoded_count / 16)

    def decode(latent):
        # Decode the sampled marker, making reordered/dropped frames observable.
        color = latent[0, 0, 0, 0].item() * 16 * 16 / 255
        return torch.full((1, latent.shape[2] * 8, latent.shape[3] * 8, 3), color)

    vae = SimpleNamespace(encode=Mock(side_effect=encode), decode=Mock(side_effect=decode))
    sampler = Mock(side_effect=lambda *args, **kwargs: dict(args[8], samples=args[8]["samples"] + 0.25))
    monkeypatch.setattr(core.impact_sampling, "ksampler_wrapper", sampler)
    # Observe the real call: output markers alone cannot detect wrong VAE inputs.
    enhance = Mock(wraps=core.enhance_detail_for_animatediff)
    monkeypatch.setattr(core, "enhance_detail_for_animatediff", enhance)
    # The wrapper applies DifferentialDiffusion before calling core; its mask
    # arithmetic remains real, while the learned model is a boundary double.
    model = SimpleNamespace(model_options={"denoise_mask_function": object()})
    pipe = model, object(), vae, [], []
    arguments = dict(image_frames=frames, segs=segs, guide_size=64, guide_size_for=True,
                     max_size=96, seed=632, steps=20, cfg=8, sampler_name="euler",
                     scheduler="normal", denoise=0.8, basic_pipe=pipe, refiner_ratio=0.2,
                     noise_mask_feather=20)
    if tiled:
        output, enhanced, returned_pipe, cnet_images = animatediff.DetailerForEachPipeForAnimateDiff.doit(**arguments, feather=5)
        assert returned_pipe is pipe
    else:
        enhanced, _ = animatediff.SEGSDetailerForAnimateDiff().doit(**arguments)
        output = segs_nodes.SEGSPaste.doit(frames, enhanced, feather=5)[0]

    expected = original.clone()
    assert len(enhanced[1]) == enhance.call_count == sampler.call_count == len(segs[1])
    assert vae.encode.call_count == vae.decode.call_count == 3 * len(segs[1])
    if source == "tiles-control":
        assert apply_control.call_count == len(cnet_images) == len(segs[1])
    for index, (seg, result, mask) in enumerate(zip(segs[1], enhanced[1], before_masks)):
        assert (result.crop_region, result.bbox, result.label, result.confidence) == (
            seg.crop_region, seg.bbox, seg.label, seg.confidence)
        torch.testing.assert_close(torch.as_tensor(result.cropped_mask), mask, rtol=0, atol=0)
        mask_batch = mask[None] if mask.ndim == 2 else mask
        height, width = vae.encode.call_args_list[index * 3].args[0].shape[1:3]
        # Feather 20 uses a 41-pixel kernel for tiles; 36-pixel detected
        # crops cap it to 19. Paste feather 5 uses an 11-pixel kernel.
        kernel = 41 if tiled else 19
        blurred = gaussian_blur(mask_batch[:, None], [kernel, kernel], [10.0, 10.0])
        expected_noise = torch.nn.functional.interpolate(blurred, (height, width), mode="bilinear", align_corners=False)[:, 0]
        torch.testing.assert_close(sampler.call_args_list[index].args[8]["noise_mask"], expected_noise, rtol=0, atol=0)
        # The paste contract is ordinary alpha blending, independently of the
        # production crop/resize helpers. Shared masks broadcast across frames.
        alpha = gaussian_blur(mask_batch[:, None], [11, 11], [10.0, 10.0]).movedim(1, -1)
        colors = torch.arange(index * 3 + 5, index * 3 + 8).float().reshape(3, 1, 1, 1) * 16 / 255
        x1, y1, x2, y2 = seg.crop_region
        # Subsequent tiles must see the preceding paste, before any resizing.
        np.testing.assert_array_equal(enhance.call_args_list[index].args[0],
                                      expected[:, y1:y2, x1:x2].numpy())
        if source == "tiles-control":
            kwargs = apply_control.call_args_list[index].kwargs
            assert kwargs["positive"] == kwargs["negative"] == []
            assert kwargs["control_net"] is control
            assert (kwargs["strength"], kwargs["start_percent"], kwargs["end_percent"]) == (1.0, 0.0, 1.0)
            torch.testing.assert_close(kwargs["image"], control_before[:, y1:y2, x1:x2], rtol=0, atol=0)
            assert sampler.call_args_list[index].args[6:8] == tuple(conditioning)
            torch.testing.assert_close(cnet_images[index], kwargs["image"], rtol=0, atol=0)
        torch.testing.assert_close(torch.from_numpy(result.cropped_image), colors.expand(3, y2 - y1, x2 - x1, 3), rtol=0, atol=0)
        expected[:, y1:y2, x1:x2] = colors * alpha + expected[:, y1:y2, x1:x2] * (1 - alpha)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
    for seg, mask in zip(segs[1], before_masks):
        torch.testing.assert_close(torch.as_tensor(seg.cropped_mask), mask, rtol=0, atol=0)
    if source == "tiles-control":
        torch.testing.assert_close(control_image, control_before, rtol=0, atol=0)
