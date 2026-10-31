"""Compare real LoRA patches and conditioning tensors with core ComfyUI nodes.

Run separately from pytest's stubbed wildcard tests. Requires the same models
as e2e_lora_wildcard_inference.py. Only unused HTTP route registration is
substituted; checkpoint loading, patching and CLIP encoding use real ComfyUI.
"""
import argparse
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from lora_source_case import BASE, CHECKPOINT, FIRST_SECTION, LAST_SECTIONS, LORAS


def assert_lora_patches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, patches in actual.items():
        assert len(patches) == len(expected[key]), key
        for first, second in zip(patches, expected[key]):
            assert type(first[1]) is type(second[1]), key
            # ComfyUI adapters hold tensor payloads in .weights; older loaders
            # return tuples directly. Let PyTorch compare the nested values.
            first_values = (first[0], getattr(first[1], "weights", first[1]), *first[2:])
            second_values = (second[0], getattr(second[1], "weights", second[1]), *second[2:])
            torch.testing.assert_close(first_values, second_values, rtol=0, atol=0,
                                       msg=lambda message: f"{key}: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.comfy_root.resolve()), str(root / "modules")]
    import nodes
    import comfy.model_management
    from aiohttp import web
    from server import PromptServer
    PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef())
    from impact import config, wildcard_source

    config.cached_config = {"custom_wildcards": str(root / "tests/wildcards/model_samples")}
    with torch.inference_mode():
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(CHECKPOINT)
        result = wildcard_source.LoraWildcardSource().build(model, clip, "models", LORAS[0], "batch", 0.6, BASE, 1)
        assert [len(output) for output in result] == [2] * 6
        for index, last in enumerate(LAST_SECTIONS):
            reference_model, reference_clip = nodes.LoraLoader().load_lora(
                model, clip, LORAS[0] + ".safetensors", 0.1, 0.2)
            reference_model, reference_clip = nodes.LoraLoader().load_lora(
                reference_model, reference_clip, LORAS[index] + ".safetensors", 0.6, 0.6)
            if index == 0:
                reference_model, reference_clip = nodes.LoraLoader().load_lora(
                    reference_model, reference_clip, LORAS[1] + ".safetensors", -0.15, 0.2)
            first = nodes.CLIPTextEncode().encode(reference_clip, FIRST_SECTION)[0]
            second = nodes.CLIPTextEncode().encode(reference_clip, last)[0]
            conditioning = nodes.ConditioningConcat().concat(first, second)[0]
            torch.testing.assert_close(result[2][index], conditioning, rtol=0, atol=0)
            assert result[0][index].patches and result[1][index].patcher.patches
            assert_lora_patches_equal(result[0][index].patches, reference_model.patches)
            assert_lora_patches_equal(result[1][index].patcher.patches, reference_clip.patcher.patches)
            assert result[3][index] == FIRST_SECTION + " BREAK " + last
            print(f"PASS: entry {index}, exact conditioning and all MODEL/CLIP patch tensors and strengths", flush=True)

        for prompt, sections in [("a flower, ", ["a flower,"]),
                                 (" first,,  BREAK\nsecond, \t ", ["first,,", "second,"])]:
            plain = wildcard_source.LoraWildcardSource().build(
                model, clip, "absent", LORAS[0], "fixed", 0.6, prompt, 1)
            assert plain[0][0] is model and plain[1][0] is clip
            assert plain[3] == [prompt]
            expected = nodes.CLIPTextEncode().encode(clip, sections[0])[0]
            for section in sections[1:]:
                encoded = nodes.CLIPTextEncode().encode(clip, section)[0]
                expected = nodes.ConditioningConcat().concat(expected, encoded)[0]
            torch.testing.assert_close(plain[2][0], expected, rtol=0, atol=0)
            print(f"PASS: empty category, original text and conditioning for {prompt!r}", flush=True)
        assert model.patches == {} and clip.patcher.patches == {}, "Input models were mutated"
        comfy.model_management.unload_all_models()
        del result, plain, reference_model, reference_clip, model, clip, vae
        gc.collect()


if __name__ == "__main__":
    main()
