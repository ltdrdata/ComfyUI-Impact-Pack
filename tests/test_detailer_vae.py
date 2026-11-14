"""VAE selection through the real Detailer paths.

Sampling and model inference are controlled; conditioning and encode routing
use the installed ComfyUI/Impact implementations.
"""
import pytest
import torch

from test_detailer_hooks import runtime as runtime, state as state, foreach_arguments


@pytest.mark.parametrize("inpaint", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_existing_encode_switch_also_controls_inpaint(runtime, state, inpaint, tiled):
    runtime.core.enhance_detail(**dict(state.arguments, inpaint_model=inpaint, vae_tiled_encode=tiled))
    assert state.vae.encode.call_count == (0 if tiled else 1 + inpaint)
    assert state.vae.encode_tiled.call_count == (1 + inpaint if tiled else 0)


@pytest.mark.parametrize("inpaint", [False, True])
@pytest.mark.parametrize("mode,legacy,expected", [("inherit", False, False), ("inherit", True, True),
                                                   ("disabled", True, False), ("tiled", False, True)])
def test_encode_modes_and_safe_overlap(runtime, state, inpaint, mode, legacy, expected):
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(
        vae_encode_mode=mode, vae_tile_size=512, vae_tile_overlap=256)[0]
    original_encode = state.vae.encode
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, inpaint_model=inpaint,
                                       vae_tiled_encode=legacy))
    assert state.vae.encode is original_encode, "Shared VAE was mutated"
    assert state.vae.encode.call_count == (0 if expected else 1 + inpaint)
    assert state.vae.encode_tiled.call_count == (1 + inpaint if expected else 0)
    if expected:
        for call in state.vae.encode_tiled.call_args_list:
            assert call.kwargs["tile_x"] == call.kwargs["tile_y"] == 512
            assert call.kwargs["overlap"] == (64 if mode == "inherit" else 128)


@pytest.mark.parametrize("mode,legacy,expected", [("inherit", False, False), ("inherit", True, True),
                                                   ("disabled", True, False), ("tiled", False, True)])
def test_decode_mode_reaches_every_cycle(runtime, state, mode, legacy, expected):
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(
        True, vae_decode_mode=mode, vae_tile_size=256, vae_tile_overlap=0)[0]
    result = runtime.pack.DetailerForEach.do_detail(**dict(foreach_arguments(runtime, state, hook), tiled_decode=legacy))
    assert len(result[0]) == 3
    assert state.vae.decode.call_count == (0 if expected else 3)
    assert state.vae.decode_tiled.call_count == (3 if expected else 0)
    if expected:
        for call in state.vae.decode_tiled.call_args_list:
            assert call.kwargs["tile_x"] == (64 if mode == "inherit" else 32)
            assert call.kwargs["overlap"] == (8 if mode == "inherit" else 0)


@pytest.mark.parametrize("encode_mode,decode_mode", [
    ("disabled", "disabled"), ("disabled", "tiled"), ("tiled", "disabled"), ("tiled", "tiled"),
])
def test_inpaint_vae_results_reach_sampler_and_conditioning(runtime, state, encode_mode, decode_mode):
    def encode(pixels, **kwargs):
        return pixels[:, ::8, ::8, :1].movedim(-1, 1).repeat(1, 4, 1, 1)
    state.vae.encode.side_effect = encode
    state.vae.encode_tiled.side_effect = lambda pixels, **kwargs: encode(pixels) + .125
    bias = 0 if encode_mode == "disabled" else .125
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 32:] = 1
    positive, negative = [[torch.zeros(1), {"label": "positive"}]], [[torch.zeros(1), {"label": "negative"}]]
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(
        True, vae_encode_mode=encode_mode, vae_decode_mode=decode_mode)[0]
    images = []
    runtime.core.enhance_detail(**dict(state.arguments, noise_mask=mask, positive=positive, negative=negative,
                                       inpaint_model=True, detailer_hook=hook, cycle_images=images,
                                       vae_tiled_encode=True, vae_tiled_decode=True))
    # Image resizing quantizes .25 to 63/255; inpaint replaces the masked half
    # with .5 for conditioning and encodes the unmasked original separately.
    original = torch.full((1, 4, 8, 8), 63 / 255 + bias)
    conditioning = original.clone()
    conditioning[:, :, :, 4:] = .5 + bias
    for index, call in enumerate(state.sampler.call_args_list):
        torch.testing.assert_close(call.args[8]["samples"], original + index * .125)
        torch.testing.assert_close(call.args[8]["noise_mask"], mask[:, None])
        for values, label in [(call.args[6], "positive"), (call.args[7], "negative")]:
            assert values[0][1]["label"] == label
            torch.testing.assert_close(values[0][1]["concat_latent_image"], conditioning)
            torch.testing.assert_close(values[0][1]["concat_mask"], mask[:, None])
        torch.testing.assert_close(images[index], torch.full_like(images[index],
                                   int((63 / 255 + bias + (index + 1) * .125) * 255) / 255))
    assert positive[0][1] == {"label": "positive"} and negative[0][1] == {"label": "negative"}
    assert state.vae.encode.call_count == (2 if encode_mode == "disabled" else 0)
    assert state.vae.encode_tiled.call_count == (0 if encode_mode == "disabled" else 2)
    assert state.vae.decode.call_count == (3 if decode_mode == "disabled" else 0)
    assert state.vae.decode_tiled.call_count == (0 if decode_mode == "disabled" else 3)

def test_small_decode_size_difference_is_resized_without_black_padding(runtime, state):
    state.vae.decode.side_effect = lambda samples: torch.ones(1, 512, 512, 3)
    image, _ = runtime.core.enhance_detail(**dict(state.arguments, image=torch.ones(1, 520, 520, 3)))
    assert image.shape == (1, 520, 520, 3)
    assert torch.all(image == 1)


@pytest.mark.parametrize("tile,overlap", [(512, 256), (64, 64), (512, 0)])
def test_overlap_is_valid_for_real_comfy_tile_step_calculation(runtime, state, tile, overlap):
    import comfy.utils
    encode_model = state.vae.encode_tiled.side_effect
    def encode(pixels, tile_x, tile_y, overlap, **kwargs):
        # These are the three shapes used by VAE.encode_tiled_. Model inference
        # remains controlled; native step calculation caught the source PR crash.
        for x, y in [(tile_x, tile_y), (tile_x * 2, tile_y // 2), (tile_x // 2, tile_y * 2)]:
            assert comfy.utils.get_tiled_scale_steps(pixels.shape[2], pixels.shape[1], x, y, overlap) > 0
        return encode_model(pixels)
    state.vae.encode_tiled.side_effect = encode
    hook = runtime.providers.DetailerAdvancedOptionsHook.execute(
        vae_encode_mode="tiled", vae_tile_size=tile, vae_tile_overlap=overlap)[0]
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=hook, cycle=1,
                                       image=torch.zeros(1, 512, 1024, 3), bbox=(0, 0, 1024, 512),
                                       guide_size=512, max_size=1024))
    state.vae.encode_tiled.assert_called_once()


def test_default_vae_options_do_not_overwrite_explicit_combined_modes(runtime, state):
    one = runtime.providers.DetailerAdvancedOptionsHook.execute(vae_encode_mode="tiled", vae_tile_size=256)[0]
    default = runtime.providers.DetailerAdvancedOptionsHook.execute()[0]
    combined = runtime.hooks.DetailerHookCombine(one, default)
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=combined))
    assert state.vae.encode_tiled.call_args.kwargs["tile_x"] == 256
    disabled = runtime.providers.DetailerAdvancedOptionsHook.execute(vae_encode_mode="disabled")[0]
    combined = runtime.hooks.DetailerHookCombine(combined, disabled)
    runtime.core.enhance_detail(**dict(state.arguments, detailer_hook=combined, vae_tiled_encode=True))
    assert state.vae.encode.call_count == state.vae.encode_tiled.call_count == 1
