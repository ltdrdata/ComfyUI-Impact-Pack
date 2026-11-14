"""Advanced options through the existing Detailer entry points and hooks."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detailer_hook_cases import saved_inputs
from test_detailer_hooks import runtime as runtime, state as state, foreach_arguments


def options(runtime, enabled=True, hook=None):
    advanced = runtime.providers.DetailerAdvancedOptionsHook.execute(enabled)[0]
    return advanced if hook is None else runtime.hooks.DetailerHookCombine(hook, advanced)


def test_options_keep_types_and_explicit_overrides(runtime):
    source = {"return_by_cycle_step": True, "count": 3, "weight": .25, "label": "example"}
    one = runtime.hooks.AdvancedOptionsDetailerHook(source)
    source["count"] = 99
    two = runtime.hooks.AdvancedOptionsDetailerHook({"return_by_cycle_step": False})
    combined = runtime.hooks.DetailerHookCombine(runtime.hooks.DetailerHook(), runtime.hooks.DetailerHookCombine(one, two))
    expected = {"return_by_cycle_step": False, "count": 3, "weight": .25, "label": "example"}
    actual = combined.get_detailer_options()
    assert actual == expected
    assert {key: type(value) for key, value in actual.items()} == {key: type(value) for key, value in expected.items()}
    actual["count"] = -1
    assert combined.get_detailer_options() == expected
    assert runtime.hooks.get_detailer_options(None) == {}
    assert runtime.hooks.get_detailer_options(SimpleNamespace()) == {}


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_nested_combine_preserves_custom_sampling_policy(runtime, skip, first):
    class Skip(runtime.hooks.DetailerHook):
        def get_skip_sampling(self):
            return True
    class CustomCombine(runtime.hooks.DetailerHookCombine):
        def get_skip_sampling(self):
            return skip
    child = Skip() if not skip else runtime.hooks.DetailerHook()
    custom = CustomCombine(child, child)
    custom.get_skip_sampling = Mock(wraps=custom.get_skip_sampling)
    pair = (custom, Skip()) if first else (Skip(), custom)
    combined = runtime.hooks.DetailerHookCombine(*pair)
    assert combined.get_skip_sampling() is skip
    custom.get_skip_sampling.assert_called_once_with()


@pytest.mark.parametrize("kind", ["none", "denoise", "variation", "sampler", "coreml"])
def test_disabled_options_preserve_existing_hook_outputs(runtime, state, kind):
    hook = {"none": None, "denoise": runtime.hooks.SimpleDetailerDenoiseSchedulerHook(.3),
            "variation": runtime.hooks.VariationNoiseDetailerHookProvider(987, .4),
            "sampler": runtime.hooks.CustomSamplerDetailerHookProvider(object()),
            "coreml": runtime.hooks.CoreMLHook("64x64")}[kind]
    expected = runtime.pack.DetailerForEach.do_detail(**foreach_arguments(runtime, state, hook))
    calls = state.sampler.call_args_list.copy()
    state.sampler.reset_mock()
    actual = runtime.pack.DetailerForEach.do_detail(**foreach_arguments(runtime, state, options(runtime, False, hook)))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert len(actual) == len(expected) == 6
    assert state.vae.encode.call_count == state.vae.decode.call_count == 2
    for before, after in zip(calls, state.sampler.call_args_list, strict=True):
        assert before.args[1:6] == after.args[1:6] and before.args[9] == after.args[9]
        assert before.kwargs["sampler_opt"] is after.kwargs["sampler_opt"]
        if before.kwargs["noise"] is None:
            assert after.kwargs["noise"] is None
        else:
            torch.testing.assert_close(before.kwargs["noise"], after.kwargs["noise"], rtol=0, atol=0)


@pytest.mark.parametrize("cycles", [1, 3])
@pytest.mark.parametrize("decoder", ["normal", "tiled", "fallback"])
def test_each_cycle_is_decoded_once_in_order(runtime, state, cycles, decoder):
    hook = options(runtime, hook=runtime.hooks.SimpleDetailerDenoiseSchedulerHook(.3))
    if decoder == "fallback":
        state.vae.decode.side_effect = RuntimeError("controlled decode failure")
    result = runtime.pack.DetailerForEach.do_detail(**dict(foreach_arguments(runtime, state, hook), cycle=cycles,
                                                         tiled_decode=decoder == "tiled"))
    assert len(result) == 6 and result[0].shape == (cycles, 64, 64, 3)
    for index, image in enumerate(result[0]):
        torch.testing.assert_close(image, torch.full_like(image, int((index + 1) * .125 * 255) / 255))
    assert state.sampler.call_count == cycles and state.vae.encode.call_count == 1
    decoded = state.vae.decode if decoder == "normal" else state.vae.decode_tiled
    assert decoded.call_count == cycles
    assert len(result[2]) == len(result[3]) == len(result[5][1]) == 1


def test_snapshot_does_not_alias_next_cycle_latent(runtime, state):
    class InPlaceHook(runtime.hooks.DetailerHook):
        def cycle_latent(self, latent):
            latent["samples"].zero_()
            return latent
    output = runtime.pack.DetailerForEach.do_detail(**foreach_arguments(runtime, state, options(runtime, hook=InPlaceHook())))[0]
    torch.testing.assert_close(output, torch.full_like(output, 31 / 255))


@pytest.mark.parametrize("tiled", [False, True])
def test_cycle_snapshots_preserve_masks_and_batch_indices_for_decode_hooks(runtime, state, tiled):
    observed = []
    class InPlaceMask(runtime.hooks.DetailerHook):
        def post_encode(self, latent):
            latent["batch_index"] = [0]
            return latent
        def cycle_latent(self, latent):
            latent["noise_mask"].fill_((self.cur_step + 1) / 5)
            latent["batch_index"][0] = self.cur_step
            return latent
        def pre_decode(self, latent):
            observed.append((latent["noise_mask"].mean().item(), latent["batch_index"][0]))
            latent["samples"] = latent["samples"] * latent["noise_mask"].mean()
            return latent
    images = []
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=options(runtime, hook=InPlaceMask()),
                                       vae_tiled_decode=tiled, cycle_images=images))
    assert [mask for mask, _ in observed] == pytest.approx([.2, .4, .6])
    assert [index for _, index in observed] == [0, 1, 2]
    for image, value in zip(images, [6, 25, 57], strict=True):
        torch.testing.assert_close(image, torch.full_like(image, value / 255))


@pytest.mark.parametrize("overlap", [False, True])
def test_cycle_composites_preserve_unchanged_pixels_and_final_result(runtime, state, overlap):
    boxes = [(0, 0, 32, 64), (16 if overlap else 32, 0, 48 if overlap else 64, 64)]
    segs = ((64, 64), [runtime.core.SEG(None, np.ones((64, 32), np.float32), 1., box, box, "face") for box in boxes])
    image = torch.zeros_like(state.arguments["image"])
    image[:, :, 32:] = .25
    def encode(pixels, **kwargs):
        samples = pixels[:, ::8, ::8, :].movedim(-1, 1)
        return torch.cat((samples, samples[:, :1]), dim=1)
    state.vae.encode.side_effect = encode
    args = dict(foreach_arguments(runtime, state), segs=segs, image=image)
    expected = runtime.pack.DetailerForEach.do_detail(**args)[0]
    state.vae.encode.reset_mock()
    actual = runtime.pack.DetailerForEach.do_detail(**dict(args, detailer_hook=options(runtime)))[0]
    torch.testing.assert_close(actual[-1:], expected, rtol=0, atol=0)
    second_input = torch.full((1, 64, 32, 3), 63 / 255)
    if overlap:
        second_input[:, :, :16] = 95 / 255
    torch.testing.assert_close(state.vae.encode.call_args_list[1].args[0], second_input, rtol=0, atol=0)
    for index in range(3):
        composite = image.clone()
        composite[:, :, :32] = [31, 63, 95][index] / 255
        composite[:, :, boxes[1][0]:boxes[1][2]] = torch.floor((second_input + (index + 1) * .125) * 255) / 255
        torch.testing.assert_close(actual[index:index + 1], composite, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["empty", "zero-mask", "large", "dummy", "lama"])
@pytest.mark.parametrize("auto_retry", [False, True])
def test_empty_skipped_and_dummy_paths_return_complete_batches(runtime, state, monkeypatch, kind, auto_retry):
    hook = options(runtime)
    args = foreach_arguments(runtime, state, hook)
    expected = state.arguments["image"]
    if kind == "empty":
        args["segs"] = ((64, 64), [])
    elif kind == "zero-mask":
        args["segs"][1][0].cropped_mask[:] = 0
    elif kind == "large":
        args["force_inpaint"] = False
    elif kind == "dummy":
        args["model"] = "DUMMY"
    else:
        def remover():
            return SimpleNamespace(lama_remover=lambda image, **kwargs: (torch.full_like(image, .75),))
        monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "LamaRemover", remover)
        lama = runtime.providers.LamaRemoverDetailerHookProvider().doit(250, 4, True)[0]
        args["detailer_hook"] = runtime.hooks.DetailerHookCombine(lama, runtime.hooks.DetailerHookCombine(hook, options(runtime)))
        expected = torch.full_like(expected, 191 / 255)
    cls = runtime.pack.DetailerForEachAutoRetry if auto_retry else runtime.pack.DetailerForEach
    result = cls.do_detail(**args)
    assert result[0].shape == (3, 64, 64, 3)
    torch.testing.assert_close(result[0], expected.expand(3, -1, -1, -1), rtol=0, atol=0)
    state.sampler.assert_not_called()
    state.vae.encode.assert_not_called()
    state.vae.decode.assert_not_called()


def test_retry_discards_failed_attempt_cycles(runtime, state):
    retry = runtime.providers.BlackPatchRetryHookProvider().doit(10, 5)[0]
    state.vae.decode.side_effect = [torch.full((1, 64, 64, 3), v) for v in [.0, .0, .0, .125, .25, .5]]
    output = runtime.pack.DetailerForEachAutoRetry.do_detail(**dict(
        foreach_arguments(runtime, state, options(runtime, hook=retry)), max_retries=2))[0]
    assert state.sampler.call_count == state.vae.decode.call_count == 6
    assert state.vae.encode.call_count == 2
    assert output.shape == (3, 64, 64, 3)
    for image, value in zip(output, [31, 63, 127]):
        torch.testing.assert_close(image, torch.full_like(image, value / 255))


def test_coreml_and_post_paste_apply_once_per_output(runtime, state):
    hook = runtime.hooks.CoreMLHook("64x64")
    hook.pre_decode = Mock(wraps=hook.pre_decode)
    hook.post_decode = Mock(wraps=hook.post_decode)
    hook.post_paste = Mock(side_effect=lambda image: image + .01)
    output = runtime.pack.DetailerForEach.do_detail(**foreach_arguments(runtime, state, options(runtime, hook=hook)))[0]
    assert hook.pre_decode.call_count == hook.post_decode.call_count == hook.post_paste.call_count == 3
    assert all(c.args[0].shape[0] == 1 for c in state.vae.decode.call_args_list)
    for index, image in enumerate(output):
        torch.testing.assert_close(image, torch.full_like(image, int((index + 1) * .125 * 255) / 255 + .01))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("auto_retry", [False, True])
def test_post_paste_observes_each_cycle_and_preserves_final_only_state(runtime, state, enabled, auto_retry):
    class Stamp(runtime.hooks.DetailerHook):
        def __init__(self):
            self.observed = []
        def post_paste(self, image):
            self.observed.append((self.cur_step, self.total_step))
            image = image.clone()
            image[:, 0, 0] = self.cur_step / 10
            return image
    stamp = Stamp()
    hook = options(runtime, enabled, stamp)
    cls = runtime.pack.DetailerForEachAutoRetry if auto_retry else runtime.pack.DetailerForEach
    image = cls.do_detail(**foreach_arguments(runtime, state, hook))[0]
    steps = [0, 1, 2] if enabled else [2]
    assert stamp.observed == [(step, 3) for step in steps]
    torch.testing.assert_close(image[:, 0, 0, 0], torch.tensor(steps) / 10)
    assert (stamp.cur_step, stamp.total_step) == (2, 3)


@pytest.mark.parametrize("kind,composition", [
    ("none", "direct"), ("none", "nested"), ("plain", "first"),
    ("coreml", "first"), ("coreml", "second"), ("coreml", "nested"),
    ("combined", "nested"), ("custom-combine", "nested"),
])
def test_options_preserve_face_segmentation_override(runtime, state, kind, composition):
    class Detector:
        def __init__(self, box, override=False):
            self.box = box
            self.override_bbox_by_segm = override
        def setAux(self, text):
            pass
        def detect(self, image, *args, **kwargs):
            x1, y1, x2, y2 = self.box
            seg = runtime.core.SEG(None, np.ones((y2 - y1, x2 - x1), np.float32), 1., self.box, self.box, "face")
            return ((64, 64), [seg])
    class CustomCombine(runtime.hooks.DetailerHookCombine):
        override_bbox_by_segm = False
    coreml = runtime.hooks.CoreMLHook("64x64")
    plain = runtime.hooks.DetailerHook()
    base_hook = {"none": None, "plain": plain, "coreml": coreml,
                 "combined": runtime.hooks.DetailerHookCombine(coreml, plain),
                 "custom-combine": CustomCombine(plain, plain)}[kind]
    params = saved_inputs("sampler", 51)
    params.update({key: state.arguments[key] for key in ("model", "clip", "vae", "positive", "negative", "image")})
    params.update(guide_size=64, max_size=64, cycle=1, force_inpaint=True, noise_mask_feather=0,
                  feather=0, bbox_detector=Detector((0, 0, 32, 64)),
                  segm_detector_opt=Detector((32, 0, 64, 64), True), detailer_hook=base_hook)
    expected = runtime.pack.FaceDetailer().doit(**params)
    # Preserve the legacy policy, including attribute-presence checks on hooks.
    mask = torch.zeros(64, 64)
    if kind in ("none", "coreml", "custom-combine"):
        mask[:, 32:] = 1
    torch.testing.assert_close(expected[3], mask, rtol=0, atol=0)
    advanced = options(runtime, False)
    if composition == "direct":
        hook = advanced
    elif composition == "first":
        hook = runtime.hooks.DetailerHookCombine(advanced, base_hook)
    elif composition == "second":
        hook = runtime.hooks.DetailerHookCombine(base_hook, advanced)
    else:
        inner = advanced if base_hook is None else runtime.hooks.DetailerHookCombine(advanced, base_hook)
        hook = runtime.hooks.DetailerHookCombine(inner, options(runtime, False))
    state.sampler.reset_mock()
    actual = runtime.pack.FaceDetailer().doit(**dict(params, detailer_hook=hook))
    torch.testing.assert_close(actual[3], mask, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert state.sampler.call_count == int(mask.any())


@pytest.mark.parametrize("empty", [False, True])
def test_face_batch_masks_follow_input_then_cycle_order(runtime, state, empty):
    class Detector:
        def setAux(self, text):
            pass
        def detect(self, image, *args, **kwargs):
            box = (0, 0, 32, 64) if image.mean() < .4 else (32, 0, 64, 64)
            seg = runtime.core.SEG(None, np.ones((64, 32), np.float32), 1., box, box, "face")
            return ((64, 64), [] if empty else [seg])
    params = saved_inputs("sampler", 51)
    params.update(guide_size=64, max_size=64, cycle=3, force_inpaint=True, noise_mask_feather=0, feather=0)
    params.update({key: state.arguments[key] for key in ("model", "clip", "vae", "positive", "negative")})
    params.update(image=torch.cat((state.arguments["image"], state.arguments["image"] * 2)), bbox_detector=Detector(),
                  detailer_hook=options(runtime))
    output = runtime.pack.FaceDetailer().doit(**params)
    assert output[0].shape == (6, 64, 64, 3) and output[3].shape == (6, 64, 64)
    masks = torch.zeros(6, 64, 64)
    if not empty:
        masks[:3, :, :32] = 1
        masks[3:, :, 32:] = 1
    torch.testing.assert_close(output[3], masks, rtol=0, atol=0)
    torch.testing.assert_close(output[0][:3, :, 32:], torch.full((3, 64, 32, 3), .25))
    torch.testing.assert_close(output[0][3:, :, :32], torch.full((3, 64, 32, 3), .5))
    pipe_params = {key: value for key, value in params.items()
                   if key not in ("model", "clip", "vae", "positive", "negative", "wildcard", "bbox_detector", "detailer_hook")}
    piped = runtime.pack.FaceDetailerPipe().doit(detailer_pipe=output[4], **pipe_params)
    torch.testing.assert_close(piped[0], output[0], rtol=0, atol=0)
    torch.testing.assert_close(piped[3], masks, rtol=0, atol=0)


@pytest.mark.parametrize("masked", [False, True])
def test_mask_detailer_preserves_group_then_cycle_order(runtime, state, masked):
    params = saved_inputs("cycles", 3)
    params.update(guide_size=64, max_size=64, crop_factor=1., batch_size=2, feather=0, seed=123)
    state.sampler.side_effect = lambda *a, **kw: dict(a[8], samples=torch.full_like(a[8]["samples"], (a[1] - 122) / 8))
    pipe = tuple(state.arguments[key] for key in ("model", "clip", "vae", "positive", "negative"))
    output = runtime.pack.MaskDetailerPipe().doit(image=state.arguments["image"], mask=torch.ones(1, 64, 64) if masked else None,
                                               basic_pipe=pipe, detailer_hook=options(runtime), **params)
    assert output[0].shape == (6, 64, 64, 3)
    assert [call.args[1] for call in state.sampler.call_args_list] == ([123, 124, 125, 124, 125, 126] if masked else [])
    values = [31, 63, 95, 63, 95, 127] if masked else [63.75] * 6
    expected = torch.tensor(values).view(6, 1, 1, 1).expand_as(output[0]) / 255
    torch.testing.assert_close(output[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("node_type", ["DetailerForEach", "DetailerForEachAutoRetry", "DetailerForEachTest",
                                       "DetailerForEachPipe", "DetailerForEachTestPipe"])
def test_existing_node_entry_points_forward_options(runtime, state, node_type):
    args = foreach_arguments(runtime, state, options(runtime))
    args["guide_size_for"] = args.pop("guide_size_for_bbox")
    args["wildcard"] = ""
    if "Pipe" in node_type:
        args["basic_pipe"] = tuple(args.pop(key) for key in ("model", "clip", "vae", "positive", "negative"))
    result = getattr(runtime.pack, node_type)().doit(**args)
    assert result[0].shape == (3, 64, 64, 3)
    assert state.sampler.call_count == state.vae.decode.call_count == 3
    expected = torch.tensor([31, 63, 95]).view(3, 1, 1, 1).expand_as(result[0]) / 255
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


def test_retry_failure_does_not_leak_images_into_next_execution(runtime, state):
    retry = runtime.providers.BlackPatchRetryHookProvider().doit(10, 5)[0]
    hook = options(runtime, hook=retry)
    state.vae.decode.side_effect = lambda samples: torch.zeros(1, 64, 64, 3)
    with pytest.raises(Exception, match="Max retries reached"):
        runtime.pack.DetailerForEachAutoRetry.do_detail(**dict(foreach_arguments(runtime, state, hook), max_retries=2))
    state.vae.decode.side_effect = lambda samples: torch.full((1, 64, 64, 3), .5)
    result = runtime.pack.DetailerForEachAutoRetry.do_detail(**dict(foreach_arguments(runtime, state, hook), max_retries=2))
    assert result[0].shape == (3, 64, 64, 3)
    torch.testing.assert_close(result[0], torch.full_like(result[0], 127 / 255))


@pytest.mark.parametrize("enabled", [False, True])
def test_retry_hook_does_not_inspect_skipped_patches(runtime, state, enabled):
    retry = runtime.providers.BlackPatchRetryHookProvider().doit(10, 5)[0]
    hook = options(runtime, enabled, retry)
    result = runtime.pack.DetailerForEachAutoRetry.do_detail(**dict(
        foreach_arguments(runtime, state, hook), force_inpaint=False, max_retries=2))
    expected = state.arguments["image"].expand(3 if enabled else 1, -1, -1, -1)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    state.sampler.assert_not_called()


def upscaled_arguments(runtime, state, hook, **overrides):
    # A 64x64 crop sampled at 128x128.
    return {**foreach_arguments(runtime, state, hook), "guide_size": 128, "max_size": 128, **overrides}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("auto_retry", [False, True])
def test_upscaled_crops_replace_only_the_crop_outputs(runtime, state, enabled, auto_retry):
    cls = runtime.pack.DetailerForEachAutoRetry if auto_retry else runtime.pack.DetailerForEach
    reference = cls.do_detail(**upscaled_arguments(runtime, state, None))
    reference_calls = state.sampler.call_args_list
    state.sampler.reset_mock()
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(return_upscaled_crops=enabled)[0]
    image, cropped, refined, alpha, _, (_, segs) = cls.do_detail(**upscaled_arguments(runtime, state, hook))

    size = 128 if enabled else 64
    assert [tuple(x.shape) for x in cropped] == [tuple(x.shape) for x in refined] == [(1, size, size, 3)]
    if enabled:
        # The image given to the VAE and the decoded image before it is scaled back.
        assert state.vae.encode.call_args.args[0] is cropped[0]
        assert torch.equal(refined[0], torch.full((1, 128, 128, 3), .375))
    else:
        assert torch.equal(cropped[0], reference[1][0]) and torch.equal(refined[0], reference[2][0])
    # Everything else keeps the crop size and its pixels.
    assert torch.equal(image, reference[0])
    assert torch.equal(alpha[0], reference[3][0]) and alpha[0].shape == (1, 64, 64, 4)
    assert np.array_equal(segs[0].cropped_image, reference[5][1][0].cropped_image) and segs[0].cropped_image.shape == (1, 64, 64, 3)
    assert [c.args[9] for c in state.sampler.call_args_list] == [c.args[9] for c in reference_calls]
    for call, expected in zip(state.sampler.call_args_list, reference_calls, strict=True):
        assert torch.equal(call.args[8]["samples"], expected.args[8]["samples"])


@pytest.mark.parametrize("node_type,cropped,refined", [("DetailerForEachTest", 1, 2), ("DetailerForEachTestPipe", 3, 4)])
def test_debug_nodes_return_upscaled_crops_through_existing_slots(runtime, state, node_type, cropped, refined):
    node = getattr(runtime.pack, node_type)
    names = node.RETURN_NAMES
    args = upscaled_arguments(runtime, state, runtime.providers.DetailerAdvancedOptionsHook.execute(return_upscaled_crops=True)[0])
    args["guide_size_for"] = args.pop("guide_size_for_bbox")
    args["wildcard"] = ""
    if "Pipe" in node_type:
        args["basic_pipe"] = tuple(args.pop(key) for key in ("model", "clip", "vae", "positive", "negative"))
    result = node().doit(**args)
    assert len(result) == len(names) and (names[cropped], names[refined]) == ("cropped", "cropped_refined")
    assert [tuple(x.shape) for x in result[cropped]] == [tuple(x.shape) for x in result[refined]] == [(1, 128, 128, 3)]
    assert tuple(result[names.index("cropped_refined_alpha")][0].shape) == (1, 64, 64, 4)
    assert result[0].shape == (1, 64, 64, 3)


@pytest.mark.parametrize("kind", ["skip", "dummy", "lama", "cycles", "combined"])
@pytest.mark.parametrize("auto_retry", [False, True])
def test_upscaled_crops_follow_each_processing_path(runtime, state, monkeypatch, kind, auto_retry):
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(return_by_cycle_step=kind == "cycles", return_upscaled_crops=True)[0]
    overrides = {}
    if kind == "skip":
        overrides = dict(guide_size=32, force_inpaint=False)
    elif kind == "dummy":
        overrides = dict(model="DUMMY")
    elif kind == "lama":
        monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "LamaRemover",
                            lambda: SimpleNamespace(lama_remover=lambda image, **kwargs: (torch.full_like(image, .75),)))
        hook = runtime.hooks.DetailerHookCombine(runtime.providers.LamaRemoverDetailerHookProvider().doit(250, 4, True)[0], hook)
    elif kind == "combined":
        hook = runtime.hooks.DetailerHookCombine(runtime.hooks.DetailerHook(), hook)
    cls = runtime.pack.DetailerForEachAutoRetry if auto_retry else runtime.pack.DetailerForEach
    image, cropped, refined, *_ = cls.do_detail(**upscaled_arguments(runtime, state, hook, **overrides))

    if kind in ("skip", "dummy"):
        # Nothing was upscaled, so the original crop is returned.
        assert [tuple(x.shape) for x in cropped] == [(1, 64, 64, 3)]
        assert [tuple(x.shape) for x in refined] == ([] if kind == "skip" else [(1, 64, 64, 3)])
        state.sampler.assert_not_called()
        return
    assert [tuple(x.shape) for x in cropped] == [tuple(x.shape) for x in refined] == [(1, 128, 128, 3)]
    if kind == "lama":
        assert refined[0] is cropped[0] and torch.equal(cropped[0], torch.full((1, 128, 128, 3), .75))
        state.sampler.assert_not_called()
    else:
        # With per-cycle images the refined crop is the last cycle, decoded once per cycle.
        assert torch.equal(refined[0], torch.full((1, 128, 128, 3), .375))
        assert state.vae.decode.call_count == (3 if kind == "cycles" else 1)
        assert image.shape == ((3, 64, 64, 3) if kind == "cycles" else (1, 64, 64, 3))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("auto_retry", [False, True])
def test_crop_outputs_stay_paired_per_region(runtime, state, enabled, auto_retry):
    # Two 64 px high crops, 80 and 82 px wide. At 2x both are sampled 160 px wide, the size the VAE keeps whole.
    image = torch.zeros(1, 64, 200, 3)
    image[:, :, :80] = .25
    image[:, :, 100:182] = .75
    boxes = [(0, 0, 80, 64), (100, 0, 182, 64)]
    segs = ((64, 200), [runtime.core.SEG(None, np.ones((64, x2 - x1), np.float32), 1., (x1, y1, x2, y2), (x1, y1, x2, y2), "face")
                        for x1, y1, x2, y2 in boxes])
    state.vae.encode.side_effect = lambda pixels, **kwargs: (
        pixels[:, ::8, ::8, :1].movedim(-1, 1).repeat(1, 4, 1, 1)[:, :, :pixels.shape[1] // 8, :pixels.shape[2] // 8])
    state.sampler.side_effect = lambda *a, **kw: a[8]
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(return_upscaled_crops=enabled)[0]
    cls = runtime.pack.DetailerForEachAutoRetry if auto_retry else runtime.pack.DetailerForEach
    _, cropped, refined, alpha, *_ = cls.do_detail(**dict(
        foreach_arguments(runtime, state, hook), image=image, segs=segs, guide_size=128, max_size=256,
        guide_size_for_bbox=False, cycle=1))

    def regions(images):
        return ["left" if x[..., :3].mean() < .5 else "right" for x in images]

    assert regions(cropped) == regions(refined) == regions(alpha) == ["right", "left"]
    if enabled:
        assert [tuple(x.shape) for x in refined] == [(1, 128, 160, 3), (1, 128, 160, 3)]


def test_retry_returns_sampled_size_crops_of_the_accepted_attempt(runtime, state):
    retry = runtime.providers.BlackPatchRetryHookProvider().doit(10, 5)[0]
    advanced = runtime.providers.DetailerAdvancedOptionsHook.execute(return_upscaled_crops=True)[0]
    # The first attempt decodes a black patch and is rejected.
    state.vae.decode.side_effect = [torch.zeros(1, 128, 128, 3), torch.full((1, 128, 128, 3), .5)]
    image, cropped, refined, *_ = runtime.pack.DetailerForEachAutoRetry.do_detail(
        **upscaled_arguments(runtime, state, runtime.hooks.DetailerHookCombine(retry, advanced), max_retries=2))

    assert state.vae.decode.call_count == state.vae.encode.call_count == 2
    assert [tuple(x.shape) for x in cropped] == [tuple(x.shape) for x in refined] == [(1, 128, 128, 3)]
    assert state.vae.encode.call_args_list[1].args[0] is cropped[0]
    assert torch.equal(refined[0], torch.full((1, 128, 128, 3), .5))
    torch.testing.assert_close(image, torch.full_like(image, 127 / 255))


def test_decode_hooks_observe_the_cycle_being_decoded(runtime, state):
    class Stamp(runtime.hooks.DetailerHook):
        def __init__(self):
            self.steps = []

        def pre_decode(self, latent):
            self.steps.append(("pre", self.cur_step, self.total_step))
            return latent

        def post_decode(self, pixels):
            self.steps.append(("post", self.cur_step, self.total_step))
            return pixels

    stamp = Stamp()
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=options(runtime, hook=stamp), cycle_images=[]))
    assert stamp.steps == [(name, i, 3) for i in range(3) for name in ("pre", "post")]

    # Final-only decoding keeps the state left by the sampling loop.
    stamp.steps.clear()
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=options(runtime, False, stamp)))
    assert stamp.steps == [("pre", 2, 3), ("post", 2, 3)]
