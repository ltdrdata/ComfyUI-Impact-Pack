"""Chrome regression for conditioning and regional-prompt combine nodes.

Requires the SD1.5/realcartoonPixar_v8.safetensors checkpoint used by the
Detailer E2E. Imports through the UI and compares against core ComfyUI nodes.
"""
import json

from e2e_ui import browser_session, options


PROMPT = {
    "model": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "SD1.5/realcartoonPixar_v8.safetensors"}},
    "first": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["model", 1], "text": "a red flower"}},
    "second": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["model", 1], "text": "a blue car"}},
    "combine": {"class_type": "ImpactCombineConditionings",
                "inputs": {"conditioning1": ["first", 0], "conditioning2": ["second", 0]}},
    "concat": {"class_type": "ImpactConcatConditionings",
               "inputs": {"conditioning1": ["first", 0], "conditioning2": ["second", 0]}},
    "core_combine": {"class_type": "ConditioningCombine",
                     "inputs": {"conditioning_1": ["first", 0], "conditioning_2": ["second", 0]}},
    "core_concat": {"class_type": "ConditioningConcat",
                    "inputs": {"conditioning_to": ["first", 0], "conditioning_from": ["second", 0]}},
    "pipe": {"class_type": "ToBasicPipe", "inputs": {"model": ["model", 0], "clip": ["model", 1],
             "vae": ["model", 2], "positive": ["first", 0], "negative": ["second", 0]}},
    "sampler": {"class_type": "KSamplerAdvancedProvider", "inputs": {"cfg": 8.0, "sampler_name": "euler",
                "scheduler": "normal", "sigma_factor": 1.0, "basic_pipe": ["pipe", 0]}},
    "mask_a": {"class_type": "SolidMask", "inputs": {"value": 0.25, "width": 16, "height": 16}},
    "mask_b": {"class_type": "SolidMask", "inputs": {"value": 0.75, "width": 16, "height": 16}},
    "region_a": {"class_type": "RegionalPrompt", "inputs": {"mask": ["mask_a", 0], "advanced_sampler": ["sampler", 0]}},
    "region_b": {"class_type": "RegionalPrompt", "inputs": {"mask": ["mask_b", 0], "advanced_sampler": ["sampler", 0]}},
    "regions": {"class_type": "CombineRegionalPrompts",
                "inputs": {"regional_prompts1": ["region_a", 0], "regional_prompts2": ["region_b", 0]}},
}
for node in ("combine", "concat", "core_combine", "core_concat", "region_a", "region_b", "regions"):
    PROMPT[f"preview_{node}"] = {"class_type": "PreviewAny", "inputs": {"source": [node, 0]}}


def main():
    args = options(__doc__)
    with browser_session(args) as ui:
        ui.new_workflow()
        ui.load_workflow({"name": "conditioning-api.json", "mimeType": "application/json",
                          "buffer": json.dumps(PROMPT).encode("utf-8")}, len(PROMPT))
        saved = ui.app.evaluate("app => app.graph.serialize()")
        expected = {
            "combine": ["conditioning1", "conditioning2", "conditioning3"],
            "concat": ["conditioning1", "conditioning2", "conditioning3"],
            "regions": ["regional_prompts1", "regional_prompts2", "regional_prompts3"],
        }
        for reload_workflow in (False, True):
            if reload_workflow:
                ui.new_workflow()
                ui.load_workflow({"name": "conditioning.json", "mimeType": "application/json",
                                  "buffer": json.dumps(saved).encode("utf-8")}, len(PROMPT))
                assert ui.app.evaluate("app => app.graph.serialize().links") == saved["links"]
            for node, names in expected.items():
                inputs = ui.app.evaluate("(app, id) => app.graph.getNodeById(id).inputs.map(s => [s.name, s.link])", node)
                assert [slot[0] for slot in inputs] == names
                assert all(slot[1] is not None for slot in inputs[:2]) and inputs[2][1] is None
            outputs = ui.run()["outputs"]
            assert outputs["preview_combine"]["text"] == outputs["preview_core_combine"]["text"]
            assert outputs["preview_concat"]["text"] == outputs["preview_core_concat"]["text"]
            first = outputs["preview_region_a"]["text"][0]
            second = outputs["preview_region_b"]["text"][0]
            assert first != second
            assert outputs["preview_regions"]["text"] == [f"[{first[1:-1]}, {second[1:-1]}]"]
        print("PASS: UI import/Run/reload, core conditioning parity, ordered regional-prompt combination")


if __name__ == "__main__":
    main()
