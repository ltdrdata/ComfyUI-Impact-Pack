"""Check input validation with real ComfyUI on CPU, without models or a server.

Run separately from pytest's stubbed wildcard tests. Only the MODEL/CLIP input
provider and unused HTTP route registration are substituted.
"""
import argparse
import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.comfy_root.resolve()), str(root / "modules")]
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    import execution
    import nodes
    from aiohttp import web
    from server import PromptServer
    PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef())
    from impact import config, wildcard_source

    class Inputs:
        RETURN_TYPES = ("MODEL", "CLIP")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    nodes.NODE_CLASS_MAPPINGS["ValidationInputs"] = Inputs
    nodes.NODE_CLASS_MAPPINGS["LoraWildcardSource"] = wildcard_source.LoraWildcardSource
    cases = [
        *[(mode, {"mode": mode}, None) for mode in ("fixed", "random", "sequential", "batch")],
        ("lower bounds", {"strength": 0.0, "seed": 0}, None),
        ("upper bounds", {"strength": 2.0, "seed": 0xffffffffffffffff}, None),
        ("category alias and secondary choice", {"category": "pose", "lora_wildcard": "B"}, None),
        ("removed saved choice", {"lora_wildcard": "Removed"}, None),
        ("invalid mode", {"mode": "typo"}, "value_not_in_list"),
        ("negative strength", {"strength": -0.1}, "value_smaller_than_min"),
        ("excessive strength", {"strength": 2.1}, "value_bigger_than_max"),
        ("negative seed", {"seed": -1}, "value_smaller_than_min"),
        ("excessive seed", {"seed": 0x10000000000000000}, "value_bigger_than_max"),
        ("invalid number", {"strength": "not-a-number"}, "invalid_input_type"),
        ("invalid model link", {"model": ["1", 1]}, "return_type_mismatch"),
    ]
    with TemporaryDirectory() as temporary:
        config.cached_config = {"custom_wildcards": temporary}
        Path(temporary, "aaa.txt").write_text("portrait, <lora:A:0.7>\n", encoding="utf-8")
        Path(temporary, "poses.txt").write_text("pose, <lora:B:0.7>\n", encoding="utf-8")
        for label, changes, expected_error in cases:
            inputs = dict(model=["1", 0], clip=["1", 1], category="aaa", lora_wildcard="A",
                          mode="fixed", strength=0.8, base_prompt="base", seed=0) | changes
            prompt = {
                "1": {"class_type": "ValidationInputs", "inputs": {}},
                "2": {"class_type": "LoraWildcardSource", "inputs": inputs},
            }
            valid, errors, _ = asyncio.run(execution.validate_inputs("validation", prompt, "2", {}))
            assert valid == (expected_error is None), (label, valid, errors)
            assert [error["type"] for error in errors] == ([] if expected_error is None else [expected_error]), (label, errors)
            if expected_error:
                assert errors[0]["extra_info"]["input_name"] in changes, (label, errors)
            print(f"PASS: {label}", flush=True)
    print(f"PASS: {len(cases)} input validation cases", flush=True)


if __name__ == "__main__":
    main()
