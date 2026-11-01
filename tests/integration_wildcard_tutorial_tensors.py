"""Original tutorial inputs against explicit core/Inspire references on CPU.

Uses installed original SD1.5/SDXL checkpoints and LoRAs, downloading nothing.
This verifies Encode's output boundary, including CLIP reused downstream; the
missing SDXLPromptStyler and BNK plugins are not substituted or claimed tested.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import unittest

import torch

from integration_lora_wildcard_tensors import assert_lora_patches_equal
from integration_wildcard_cache_tensors import runtime
from wildcard_tutorial_cases import tutorial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--tutorials-root", type=Path, required=True)
    parser.add_argument("--inspire-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    nodes, impact_server, impact_pack = runtime(args.comfy_root)
    sys.path.insert(0, str(args.inspire_root.resolve()))
    from inspire.lora_block_weight import LoraLoaderBlockWeight
    import comfy.model_management
    nodes.NODE_CLASS_MAPPINGS["LoraLoaderBlockWeight //Inspire"] = LoraLoaderBlockWeight
    checkpoints = {}

    def original(case):
        _, workflow, target = tutorial(args.tutorials_root, case)
        checkpoint = next(n["widgets_values"][0] for n in workflow["nodes"]
                          if n["type"] == "CheckpointLoaderSimple")
        override = "IMPACT_TUTORIAL_SD15_CHECKPOINT" if case == "lbw" else "IMPACT_TUTORIAL_SDXL_CHECKPOINT"
        selected = os.environ.get(override, checkpoint)
        checkpoints[case] = {"original": checkpoint, "selected": selected}
        model, clip, _ = nodes.CheckpointLoaderSimple().load_checkpoint(selected)
        source, text, mode, _, _, seed, _ = target["widgets_values"]
        return model, clip, dict(wildcard_text=source, populated_text=text, mode=mode, seed=seed)

    class TutorialTensorTests(unittest.TestCase):
        def tearDown(self):
            comfy.model_management.unload_all_models()
            gc.collect()

        def compare(self, model, clip, inputs, reference, text, encoded_text, downstream=None):
            inputs = dict(inputs)
            impact_server.onprompt_populate_wildcards({"prompt": {
                "1": {"class_type": "ImpactWildcardEncode", "inputs": inputs}}})
            actual_model, actual_clip, conditioning, actual_text = impact_pack.ImpactWildcardEncode().doit(
                model=model, clip=clip, **inputs)
            self.assertEqual(actual_text, text)
            reference_model, reference_clip = reference
            expected = nodes.CLIPTextEncode().encode(reference_clip, encoded_text)[0]
            torch.testing.assert_close(conditioning, expected, rtol=0, atol=0)
            assert_lora_patches_equal(actual_model.patches, reference_model.patches)
            assert_lora_patches_equal(actual_clip.patcher.patches, reference_clip.patcher.patches)
            if downstream is not None:
                actual = nodes.CLIPTextEncode().encode(actual_clip, downstream)[0]
                expected = nodes.CLIPTextEncode().encode(reference_clip, downstream)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(model.patches, {})
            self.assertEqual(clip.patcher.patches, {})
            return actual_model, actual_clip

        def test_original_lbw_vector_and_strengths(self):
            model, clip, inputs = original("lbw")
            # Explicit expanded vector is independent of Impact's A/B parser.
            reference = LoraLoaderBlockWeight().doit(
                model, clip, "SD1.5/cloth/2b leotardV2.safetensors", 1.0, 1.0,
                False, 0, 1.0, 1.0, "", "0,0,0,0,0,0,0,2,2,1.5,1.5,0,0,0,0,0,0")[:2]
            self.assertTrue(reference[0].patches)
            self.compare(model, clip, inputs, reference,
                         inputs["populated_text"], ", 2b leotard, (covered eyes:1.1), hair over eyes")

        def test_sdxl_seed_changes_selected_lora_and_downstream_clip(self):
            model, clip, inputs = original("styler")
            # Trusted Main selections for the original seven-way multiline
            # source. Different seeds must change actual LoRA patches too.
            for seed, name, lora in ((3, "D.VA", "SDXL/person/d.va_xl_v1"),
                                     (2, "chunli", "SDXL/person/chun-li_xl_v1")):
                with self.subTest(seed=seed):
                    reference = nodes.LoraLoader().load_lora(model, clip, lora + ".safetensors", 1.0, 1.0)
                    self.assertTrue(reference[0].patches)
                    self.compare(model, clip, inputs | {"seed": seed}, reference,
                                 f"{name} <lora:{lora}>", name, downstream=f"a photograph, {name}")

        def test_sdxl_empty_prompt_passes_model_and_clip_through(self):
            model, clip, inputs = original("a1111")
            actual_model, actual_clip = self.compare(
                model, clip, inputs, (model, clip), "", "", downstream="a photograph")
            self.assertIs(actual_model, model)
            self.assertIs(actual_clip, clip)

    with torch.inference_mode():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(TutorialTensorTests))
    args.report.write_text(json.dumps({
        "test_groups": result.testsRun, "successful": result.wasSuccessful(),
        "checkpoints": checkpoints,
        "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures],
        "errors": [{"test": str(test), "detail": detail} for test, detail in result.errors],
        "skipped": result.skipped,
    }, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
