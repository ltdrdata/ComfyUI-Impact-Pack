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


def same_values(first, second, torch):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            same_values(a, b, torch) for a, b in zip(first, second))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same_values(first[k], second[k], torch) for k in first)
    if hasattr(first, "weights"):
        return type(first) is type(second) and same_values(first.weights, second.weights, torch)
    return first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.comfy_root.resolve()), str(root / "modules")]
    import torch
    import nodes
    import comfy.model_management
    from aiohttp import web
    from server import PromptServer
    PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef())
    from impact import config, wildcard_source
    from e2e_lora_wildcard_inference import BASE, FIRST_SECTION, LAST_SECTIONS, LORAS

    config.cached_config = {"custom_wildcards": str(root / "tests/wildcards/model_samples")}
    with torch.inference_mode():
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint("v1-5-pruned-emaonly-fp16.safetensors")
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
            assert same_values(result[2][index], conditioning, torch), f"Conditioning differs: {index}"
            assert same_values(result[0][index].patches, reference_model.patches, torch), f"MODEL patches differ: {index}"
            assert same_values(result[1][index].patcher.patches, reference_clip.patcher.patches, torch), f"CLIP patches differ: {index}"
            assert result[3][index] == FIRST_SECTION + " BREAK " + last
            print(f"PASS: entry {index}, exact conditioning and all MODEL/CLIP patch tensors and strengths", flush=True)
        assert model.patches == {} and clip.patcher.patches == {}, "Input models were mutated"
        comfy.model_management.unload_all_models()
        del result, reference_model, reference_clip, model, clip, vae
        gc.collect()


if __name__ == "__main__":
    main()
