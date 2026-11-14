"""Existing hook contracts derived from the original tutorials.

Uses real Impact/ComfyUI hooks, image operations and Detailer loops on CPU.
Only sampling, VAE inference and external noise/Lama backends are controlled.
Run separately from wildcard dependency stubs with IMPACT_COMFY_ROOT and
IMPACT_TUTORIALS_DIR set.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from detailer_hook_cases import saved_inputs


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
            send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import impact_pack, core, hooks, hook_nodes
        assert Path(core.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
        yield SimpleNamespace(pack=impact_pack, core=core, hooks=hooks, providers=hook_nodes,
                              server=PromptServer, nodes=core.nodes)


@pytest.fixture
def state(runtime, monkeypatch):
    def encode(image, **kwargs):
        return torch.zeros(len(image), 4, image.shape[1] // 8, image.shape[2] // 8)

    def decode(samples, **kwargs):
        pixels = torch.nn.functional.interpolate(samples[:, :3], scale_factor=8, mode="nearest")
        return pixels.movedim(1, -1)

    vae = SimpleNamespace(encode=Mock(side_effect=encode), encode_tiled=Mock(side_effect=encode),
                          decode=Mock(side_effect=decode), decode_tiled=Mock(side_effect=decode),
                          temporal_compression_decode=lambda: None, spacial_compression_decode=lambda: 8,
                          spacial_compression_encode=lambda: 8)
    sampler = Mock(side_effect=lambda *a, **kw: dict(a[8], samples=a[8]["samples"] + 0.125))
    monkeypatch.setattr(runtime.core.impact_sampling, "ksampler_wrapper", sampler)
    arguments = dict(image=torch.full((1, 64, 64, 3), 0.25),
                     model=SimpleNamespace(model_options={}), clip=None, vae=vae,
                     guide_size=64, guide_size_for_bbox=True, max_size=64, bbox=(0, 0, 64, 64),
                     seed=123, steps=25, cfg=6.0, sampler_name="euler", scheduler="normal",
                     positive=[], negative=[], denoise=0.5, noise_mask=torch.ones(1, 64, 64),
                     force_inpaint=True, cycle=3)
    return SimpleNamespace(arguments=arguments, vae=vae, sampler=sampler)


def foreach_arguments(runtime, state, hook=None):
    args = dict(state.arguments)
    box = args.pop("bbox")
    args.update(segs=((64, 64), [runtime.core.SEG(None, np.ones((64, 64), np.float32), 1., box, box, "face")]),
                noise_mask=True, feather=0, detailer_hook=hook)
    return args


@pytest.mark.parametrize("cycle", [1, 3])
@pytest.mark.parametrize("combined", [False, True])
def test_tutorial_denoise_schedule_reaches_sampler(runtime, state, cycle, combined):
    hook = runtime.pack.DenoiseSchedulerDetailerHookProvider().doit(**saved_inputs("cycles", 21))[0]
    if combined:
        hook = runtime.hooks.DetailerHookCombine(hook, runtime.hooks.DetailerHook())
    output, control = runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, cycle=cycle))
    calls = state.sampler.call_args_list
    assert [c.args[1] for c in calls] == list(range(123, 123 + cycle))
    assert [c.args[9] for c in calls] == pytest.approx([0.5] if cycle == 1 else [0.5, 0.4, 0.3])
    for index, call in enumerate(calls):
        torch.testing.assert_close(call.args[8]["samples"], torch.full((1, 4, 8, 8), index * .125))
        torch.testing.assert_close(call.args[8]["noise_mask"], torch.ones(1, 64, 64))
    assert state.vae.encode.call_count == state.vae.decode.call_count == 1
    assert output.shape == (1, 64, 64, 3) and control is None
    torch.testing.assert_close(output, torch.full_like(output, int(cycle * .125 * 255) / 255))
    # Reusing a hook on another image must reset cycle progress.
    state.sampler.reset_mock()
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, cycle=1))
    assert state.sampler.call_args.args[9] == .5


@pytest.mark.parametrize("schedule,indices,strengths,ends", [
    ("from_start", [0, 1, 2], [9, 19 / 3, 11 / 3], [16, 17, 18]),
    ("skip_start", [1, 2], [9, 5], [16, 18]),
])
@pytest.mark.parametrize("backend", ["noise", "unsampler"])
def test_tutorial_cycle_hooks_preserve_masks_and_order(runtime, state, monkeypatch, schedule, indices, strengths, ends, backend):
    calls = []
    def inject(samples, strength, noise, mask):
        calls.append((strength, mask.clone(), samples["samples"].clone()))
        return ({"samples": samples["samples"] + .01},)
    noise = Mock(return_value=({"samples": torch.zeros(1, 4, 8, 8)},))
    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "BNK_NoisyLatentImage", lambda: SimpleNamespace(create_noisy_latents=noise))
    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "BNK_InjectNoise", lambda: SimpleNamespace(inject_noise=inject))
    def unsample(model, cfg, sampler, steps, end, scheduler, normalize, positive, negative, samples):
        calls.append((end, samples["noise_mask"].clone(), samples["samples"].clone()))
        return ({"samples": samples["samples"] + .01},)
    monkeypatch.setattr(runtime.hooks.noise_nodes.Unsampler, "unsampler", staticmethod(unsample))
    if backend == "noise":
        params = dict(saved_inputs("cycles", 16), schedule_for_cycle=schedule)
        hook = runtime.pack.NoiseInjectionDetailerHookProvider().doit(**params)[0]
    else:
        params = dict(saved_inputs("cycles", 17), schedule_for_cycle=schedule,
                      model=state.arguments["model"], positive=[], negative=[])
        hook = runtime.pack.UnsamplerDetailerHookProvider().doit(**params)[0]
    denoise = runtime.pack.DenoiseSchedulerDetailerHookProvider().doit(**saved_inputs("cycles", 21))[0]
    hook = runtime.hooks.DetailerHookCombine(denoise, hook)
    output, _ = runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook))
    assert [c[0] for c in calls] == pytest.approx(strengths if backend == "noise" else ends)
    if backend == "noise":
        seed = params["seed"]
        assert [c.args for c in noise.call_args_list] == [("GPU", seed + i * 2, 64, 64, 1) for i in range(len(indices))]
    for injection, (index, (_, mask, samples)) in enumerate(zip(indices, calls, strict=True)):
        torch.testing.assert_close(mask, torch.ones(1, 64, 64))
        torch.testing.assert_close(samples, torch.full_like(samples, index * .125 + injection * .01))
    assert [c.args[9] for c in state.sampler.call_args_list] == pytest.approx([.5, .4, .3])
    for index, call in enumerate(state.sampler.call_args_list):
        torch.testing.assert_close(call.args[8]["noise_mask"], torch.ones(1, 64, 64))
        expected = index * .125 + sum(i <= index for i in indices) * .01
        torch.testing.assert_close(call.args[8]["samples"], torch.full((1, 4, 8, 8), expected))
    assert len(calls) == len(indices)
    torch.testing.assert_close(output, torch.full_like(output, int((.375 + len(indices) * .01) * 255) / 255))


@pytest.mark.parametrize("order", ["sampler-first", "sampler-second", "two-samplers", "nested"])
def test_custom_sampler_identity_and_precedence(runtime, state, order):
    first, second = object(), object()
    one = runtime.pack.CustomSamplerDetailerHookProvider().doit(first)[0]
    two = runtime.pack.CustomSamplerDetailerHookProvider().doit(second)[0]
    plain = runtime.hooks.DetailerHook()
    pairs = {"sampler-first": (one, plain), "sampler-second": (plain, one), "two-samplers": (one, two),
             "nested": (runtime.hooks.DetailerHookCombine(plain, one), two)}
    hook = runtime.hooks.DetailerHookCombine(*pairs[order])
    scheduler_func = object()
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, scheduler_func=scheduler_func))
    assert state.sampler.call_count == 3
    assert all(c.kwargs["sampler_opt"] is first and c.kwargs["scheduler_func"] is scheduler_func
               for c in state.sampler.call_args_list)


@pytest.mark.parametrize("decode", ["normal", "tiled", "fallback"])
def test_coreml_encode_decode_hooks_run_once(runtime, state, decode):
    hook = runtime.hooks.CoreMLHook("64x64")
    hook.pre_decode = Mock(wraps=hook.pre_decode)
    hook.post_decode = Mock(wraps=hook.post_decode)
    if decode == "fallback":
        state.vae.decode.side_effect = RuntimeError("controlled decode failure")
    result = runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, vae_tiled_decode=decode == "tiled"))
    assert len(result) == 2 and result[0].shape == (1, 64, 64, 3)
    assert all(c.args[8]["samples"].shape[0] == 2 for c in state.sampler.call_args_list)
    hook.pre_decode.assert_called_once()
    hook.post_decode.assert_called_once()
    decoded = state.vae.decode if decode == "normal" else state.vae.decode_tiled
    decoded.assert_called_once()
    assert decoded.call_args.args[0].shape == (1, 4, 8, 8)


@pytest.mark.parametrize("kind", ["empty", "zero-mask", "large", "dummy"])
def test_skipped_segments_keep_image_and_skip_sampling(runtime, state, kind):
    hook = runtime.hooks.DetailerHook()
    hook.post_paste = Mock(wraps=hook.post_paste)
    args = foreach_arguments(runtime, state, hook)
    if kind == "empty":
        args["segs"] = ((64, 64), [])
    elif kind == "zero-mask":
        args["segs"][1][0].cropped_mask[:] = 0
    elif kind == "large":
        args["force_inpaint"] = False
    else:
        args["model"] = "DUMMY"
    result = runtime.pack.DetailerForEach.do_detail(**args)
    assert len(result) == 6
    torch.testing.assert_close(result[0], state.arguments["image"], rtol=0, atol=0)
    state.sampler.assert_not_called()
    state.vae.encode.assert_not_called()
    assert hook.post_paste.call_count == (1 if kind == "dummy" else 0)


@pytest.mark.parametrize("combined", [False, True])
def test_preview_reports_pasted_image_and_preserves_pixels(runtime, state, monkeypatch, tmp_path, combined):
    monkeypatch.setattr(runtime.hooks.folder_paths, "temp_directory", str(tmp_path))
    messages = Mock()
    monkeypatch.setattr(runtime.server.instance, "send_sync", messages)
    hook = runtime.providers.PreviewDetailerHookProvider().doit(95, "68")[0]
    if combined:
        hook = runtime.hooks.DetailerHookCombine(hook, runtime.hooks.DetailerHook())
    async def execute():
        result = runtime.pack.DetailerForEach.do_detail(**foreach_arguments(runtime, state, hook))
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*pending)
        return result
    result = asyncio.run(execute())
    assert messages.call_count == 1
    name, data = messages.call_args.args
    assert name == "impact-preview" and data["node_id"] == "68"
    with Image.open(tmp_path / data["item"]["subfolder"] / data["item"]["filename"]) as preview:
        assert preview.size == (64, 64)
        assert abs(np.asarray(preview).mean() - result[0].mean().item() * 255) < 3


def test_retry_hook_resets_cycles_and_returns_successful_patch(runtime, state):
    hook = runtime.providers.BlackPatchRetryHookProvider().doit(10, 5)[0]
    state.vae.decode.side_effect = [torch.zeros(1, 64, 64, 3), torch.full((1, 64, 64, 3), .5)]
    output = runtime.pack.DetailerForEachAutoRetry.do_detail(**dict(foreach_arguments(runtime, state, hook), max_retries=2))[0]
    assert [c.args[1] for c in state.sampler.call_args_list] == [123, 124, 125, 124, 125, 126]
    assert state.vae.encode.call_count == 2
    torch.testing.assert_close(output, torch.full_like(output, 127 / 255))


def test_tutorial_mask_detailer_batch_and_cycle_counts(runtime, state):
    params = saved_inputs("cycles", 3)
    params.update(guide_size=64, max_size=64, crop_factor=1., batch_size=2, feather=0)
    model, clip, vae, positive, negative = (state.arguments[key] for key in ("model", "clip", "vae", "positive", "negative"))
    hook = runtime.pack.DenoiseSchedulerDetailerHookProvider().doit(**saved_inputs("cycles", 21))[0]
    output = runtime.pack.MaskDetailerPipe().doit(image=state.arguments["image"], mask=torch.ones(1, 64, 64),
                                               basic_pipe=(model, clip, vae, positive, negative), detailer_hook=hook, **params)
    assert output[0].shape == (2, 64, 64, 3)
    assert state.sampler.call_count == 6 and state.vae.decode.call_count == 2
    assert [c.args[9] for c in state.sampler.call_args_list] == pytest.approx([.5, .4, .3] * 2)


@pytest.mark.parametrize("composition", ["before", "after", "nested"])
def test_variation_noise_survives_neutral_hook_composition(runtime, state, composition):
    variation = runtime.pack.VariationNoiseDetailerHookProvider().doit(987, .4)[0]
    neutral = runtime.hooks.DetailerHook()
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=variation))
    expected = [call.kwargs["noise"].clone() for call in state.sampler.call_args_list]
    assert len(expected) == 3 and all(noise.abs().sum() > 0 for noise in expected)
    state.sampler.reset_mock()
    hooks = {"before": runtime.hooks.DetailerHookCombine(neutral, variation),
             "after": runtime.hooks.DetailerHookCombine(variation, neutral),
             "nested": runtime.hooks.DetailerHookCombine(neutral, runtime.hooks.DetailerHookCombine(variation, neutral))}
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hooks[composition]))
    for call, noise in zip(state.sampler.call_args_list, expected, strict=True):
        torch.testing.assert_close(call.kwargs["noise"], noise, rtol=0, atol=0)


@pytest.mark.parametrize("composition,skip", [("single", True), ("both-skip", True), ("neutral", False)])
def test_lama_skip_sampling_composition(runtime, state, monkeypatch, composition, skip):
    remove = Mock(side_effect=lambda image, **kwargs: (torch.full_like(image, .75),))
    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "LamaRemover", lambda: SimpleNamespace(lama_remover=remove))
    hook = runtime.providers.LamaRemoverDetailerHookProvider().doit(250, 4, True)[0]
    if composition != "single":
        other = runtime.providers.LamaRemoverDetailerHookProvider().doit(250, 4, True)[0] if composition == "both-skip" else runtime.hooks.DetailerHook()
        hook = runtime.hooks.DetailerHookCombine(hook, other)
    output, _ = runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook))
    assert state.sampler.call_count == (0 if skip else 3)
    assert state.vae.encode.call_count == state.vae.decode.call_count == (0 if skip else 1)
    assert remove.call_count == (2 if composition == "both-skip" else 1)
    torch.testing.assert_close(output, torch.full_like(output, int((.75 if skip else .375) * 255) / 255))


@pytest.mark.parametrize("reverse,expected", [(False, [.5, .25, .1]), (True, [.5, .3, .3])])
def test_combined_denoise_order_is_observable(runtime, state, reverse, expected):
    first = runtime.hooks.SimpleDetailerDenoiseSchedulerHook(.3)
    second = runtime.hooks.SimpleDetailerDenoiseSchedulerHook(.1)
    hook = runtime.hooks.DetailerHookCombine(second, first) if reverse else runtime.hooks.DetailerHookCombine(first, second)
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook))
    assert [call.args[9] for call in state.sampler.call_args_list] == pytest.approx(expected)
