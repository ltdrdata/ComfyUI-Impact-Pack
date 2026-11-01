"""Compare wildcard Encode outputs with independent core CLIP/LoRA execution.

Run separately from pytest's stubbed tests. Uses installed SD1.5 checkpoint and
LoRA weights on CPU; downloads nothing. Only unused HTTP routes/feedback are
substituted. Expected selections and reference text are explicit test data.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

from integration_lora_wildcard_tensors import assert_lora_patches_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.comfy_root.resolve()), str(root / "modules")]
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    torch.set_num_threads(2)
    from aiohttp import web
    from server import PromptServer
    PromptServer.instance = SimpleNamespace(
        routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
        send_sync=lambda *args: None,
    )
    import nodes
    from impact import impact_server, impact_pack

    model, clip, _ = nodes.CheckpointLoaderSimple().load_checkpoint("v1-5-pruned-emaonly-fp16.safetensors")
    lora = "SD1.5/etc/add_detail.safetensors"

    class EncodeTensorTests(unittest.TestCase):
        def compare(self, *, seed, text, expected, mode="fixed", source="", sections=None, with_lora=False):
            inputs = dict(wildcard_text=source, populated_text=text, mode=mode, seed=seed)
            prompt = {"prompt": {"1": {"class_type": "ImpactWildcardEncode", "inputs": inputs}}}
            impact_server.onprompt_populate_wildcards(prompt)
            actual_model, actual_clip, actual_conditioning, actual_text = impact_pack.ImpactWildcardEncode().doit(
                model=model, clip=clip, **inputs)
            self.assertEqual(actual_text, expected)
            reference_model, reference_clip = model, clip
            if with_lora:
                reference_model, reference_clip = nodes.LoraLoader().load_lora(model, clip, lora, 0.6, 0.4)
                self.assertTrue(actual_model.patches)
                self.assertTrue(actual_clip.patcher.patches)
            else:
                self.assertIs(actual_model, model)
                self.assertIs(actual_clip, clip)
            reference_conditioning = None
            for section in sections or [expected]:
                encoded = nodes.CLIPTextEncode().encode(reference_clip, section)[0]
                reference_conditioning = encoded if reference_conditioning is None else (
                    nodes.ConditioningConcat().concat(reference_conditioning, encoded)[0])
            torch.testing.assert_close(actual_conditioning, reference_conditioning, rtol=0, atol=0)
            assert_lora_patches_equal(actual_model.patches, reference_model.patches)
            assert_lora_patches_equal(actual_clip.patcher.patches, reference_clip.patcher.patches)
            self.assertEqual(model.patches, {})
            self.assertEqual(clip.patcher.patches, {})

        def test_plain_text(self):
            for seed in (2, 5):
                with self.subTest(seed=seed):
                    self.compare(seed=seed, text="a red flower", expected="a red flower")

        def test_populate_selections(self):
            for seed, color in ((0, "green"), (1, "green"), (2, "red"), (5, "blue")):
                with self.subTest(seed=seed):
                    self.compare(seed=seed, text="old result", source="{red|green|blue}",
                                 mode="populate", expected=color)

        def test_fixed_multiline_and_nested(self):
            for text in ("{red|\ngreen|blue}", "{red|{green}|blue}"):
                for seed, color in ((2, "red"), (5, "blue")):
                    with self.subTest(text=text, seed=seed):
                        self.compare(seed=seed, text=text, expected=color)

        def test_reproduce_ignores_source(self):
            self.compare(seed=5, text="red", source="{red|green|blue}", mode="reproduce", expected="red")

        def test_lora_and_break(self):
            tag = "<lora:SD1.5/etc/add_detail:0.6:0.4>"
            for seed, color in ((2, "red"), (5, "blue")):
                with self.subTest(seed=seed):
                    self.compare(seed=seed, text="", mode="populate",
                                 source=tag + " {red|green|blue} BREAK a flower",
                                 expected=tag + f" {color} BREAK a flower",
                                 sections=[color, "a flower"], with_lora=True)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EncodeTensorTests))
    if args.report:
        args.report.write_text(json.dumps({
            "tests_run": result.testsRun, "successful": result.wasSuccessful(),
            "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures],
            "errors": [{"test": str(test), "detail": detail} for test, detail in result.errors],
            "skipped": result.skipped,
        }, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
