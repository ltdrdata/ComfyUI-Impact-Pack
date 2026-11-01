"""Check wildcard prompt hooks and ComfyUI cache keys on CPU, without models.

Run separately from the stubbed wildcard pytest suite. The actual server hook,
parser, node implementations and cache key builder are used. Only HTTP feedback
and the CLIP model are substituted; this does not test neural encoding quality.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.comfy_root.resolve()), str(root / "modules")]
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    from aiohttp import web
    from server import PromptServer
    feedback = []
    PromptServer.instance = SimpleNamespace(
        routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
        send_sync=lambda event, data: feedback.append((event, data)),
    )
    import nodes
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_string import StringConcatenate
    from execution import IsChangedCache
    # Match server startup order; importing impact_pack first creates a cycle.
    from impact import impact_server, impact_pack, wildcards

    classes = {name: getattr(impact_pack, name) for name in (
        "ImpactWildcardProcessor", "ImpactWildcardEncode",
    )}
    nodes.NODE_CLASS_MAPPINGS.update(classes | {"PreviewAny": PreviewAny})
    wildcards.wildcard_dict = {"colors": ["red", "green", "blue"]}
    wildcards._on_demand_mode = False

    # Captured from trusted Main d9bd2bf, before applying #1190. Do not derive
    # expected selections from the implementation under test.
    seeds = (2, 5, 42)
    selections = {
        "__colors__": ("red", "blue", "blue"),
        "{red|\nblue|green}": ("red", "green", "green"),
        "{red|{green}|blue}": ("red", "blue", "blue"),
        "{2$$, $$red|green|blue}": ("red, blue", "blue, green", "blue, green"),
        r"{red|blue\{sky\}|green}": ("red", "green", "green"),
        "{5::red|1::blue|1::green}": ("red", "blue", "blue"),
    }

    class Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    def prepare(name, seed, text, mode="fixed", source="", upstream=None):
        inputs = dict(wildcard_text=source, populated_text=text, seed=seed, mode=mode)
        prompt = deepcopy(upstream or {}) | {"1": {"class_type": name, "inputs": inputs}}
        # The saved native workflow must keep the user's seed and widget slots.
        widgets = [source, text, mode]
        widgets += ([seed, "increment", "Select the Wildcard to add to the text"]
                    if name == "ImpactWildcardProcessor" else [
                        "Select the LoRA to add to the text", "Select the Wildcard to add to the text", seed, "increment"])
        workflow_node = {"id": 1, "widgets_values": deepcopy(widgets)}
        data = {"prompt": prompt, "extra_data": {"extra_pnginfo": {"workflow": {"nodes": [workflow_node]}}}}
        feedback.clear()
        impact_server.onprompt_populate_wildcards(data)
        expected_widgets = deepcopy(widgets)
        if mode in ("populate", True) and isinstance(text, str):
            expected_widgets[1:3] = [inputs["populated_text"], "reproduce"]
        assert workflow_node["widgets_values"] == expected_widgets
        mode_feedback = [data["value"] for _, data in feedback if data["widget_name"] == "mode"]
        assert mode_feedback == (["populate"] if mode in ("populate", "reproduce", True) else [])
        assert all(data["widget_name"] != "seed" for _, data in feedback)
        return prompt

    def output(name, inputs, resolved_text=None, resolved_seed=None):
        values = dict(inputs)
        if resolved_text is not None:
            values["populated_text"] = resolved_text
        if resolved_seed is not None:
            values["seed"] = resolved_seed
        if name == "ImpactWildcardEncode":
            values.update(model=object(), clip=Clip())
        return classes[name]().doit(**values)[-1]

    async def cache_keys(prompt):
        name = prompt["1"]["class_type"]
        prompt = deepcopy(prompt) | {"2": {
            "class_type": "PreviewAny", "inputs": {"source": ["1", 3 if name == "ImpactWildcardEncode" else 0]},
        }}
        dynamic = DynamicPrompt(prompt)
        keys = CacheKeySetInputSignature(dynamic, prompt.keys(), IsChangedCache("test", dynamic, None))
        await keys.add_keys(prompt.keys())
        return [keys.get_data_key(node) for node in ("1", "2")]

    class WildcardCacheTests(unittest.TestCase):
        def test_populate_preserves_seeded_output_and_saved_values(self):
            for name in classes:
                for mode in ("populate", True):
                    for previous in ("", "previous result"):
                        with self.subTest(node=name, mode=mode, previous=previous):
                            actual = []
                            for seed in seeds:
                                prompt = prepare(name, seed, previous, mode, "__colors__")
                                actual.append(output(name, prompt["1"]["inputs"]))
                            self.assertEqual(tuple(actual), selections["__colors__"])

        def test_fixed_and_reproduce_keep_dynamic_semantics(self):
            for name in classes:
                for mode in ("fixed", "reproduce", False):
                    for text, expected in selections.items():
                        with self.subTest(node=name, mode=mode, text=text):
                            for seed, selection in zip(seeds, expected):
                                inputs = prepare(name, seed, text, mode)["1"]["inputs"]
                                self.assertEqual(output(name, inputs), selection)
                                self.assertEqual(inputs["seed"], seed)

        def test_linked_text_may_create_syntax(self):
            parts = dict(string_a="__", string_b="colors__", delimiter="")
            combined = StringConcatenate.execute(**parts)[0]
            self.assertEqual(combined, "__colors__")
            sources = [
                {"class_type": "StringConcatenate", "inputs": parts},
                {"class_type": "ExternalTextReader", "inputs": {"path": "prompt.txt"}},
                {"class_type": "ExternalTextGenerator", "inputs": {}},
            ]
            for name in classes:
                for source in sources:
                    with self.subTest(node=name, source=source["class_type"]):
                        for seed, selection in zip(seeds, selections["__colors__"]):
                            inputs = prepare(name, seed, ["upstream", 0], upstream={"upstream": source})["1"]["inputs"]
                            self.assertEqual(inputs["seed"], seed)
                            self.assertEqual(inputs["populated_text"], ["upstream", 0])
                            self.assertEqual(output(name, inputs, combined), selection)

        def test_linked_seed_is_never_replaced(self):
            for name in classes:
                for mode in ("populate", "fixed", "reproduce"):
                    with self.subTest(node=name, mode=mode):
                        upstream = {"seed": {"class_type": "ImpactInt", "inputs": {"value": 5}}}
                        inputs = prepare(name, ["seed", 0], "__colors__", mode, "__colors__", upstream)["1"]["inputs"]
                        self.assertEqual(inputs["seed"], ["seed", 0])
                        self.assertEqual(output(name, inputs, resolved_seed=5), "blue")

        def test_unresolved_syntax_after_population_keeps_seed(self):
            for name in classes:
                with self.subTest(node=name):
                    inputs = prepare(name, 42, "old", "populate", "__missing/path__")["1"]["inputs"]
                    self.assertEqual(inputs["populated_text"], "__missing/path__")
                    self.assertEqual(inputs["seed"], 42)

        def test_plain_text_reuses_both_cache_keys(self):
            for name in classes:
                for mode in ("populate", "fixed", "reproduce", True, False):
                    with self.subTest(node=name, mode=mode):
                        first = prepare(name, 731, "plain α", mode, "plain α")
                        second = prepare(name, 732, "plain α", mode, "plain α")
                        self.assertEqual(output(name, first["1"]["inputs"]), "plain α")
                        self.assertEqual(asyncio.run(cache_keys(first)), asyncio.run(cache_keys(second)))
                        changed = prepare(name, 733, "changed", mode, "changed")
                        self.assertNotEqual(asyncio.run(cache_keys(first)), asyncio.run(cache_keys(changed)))

        def test_populated_cache_follows_result_instead_of_original_seed(self):
            for name in classes:
                with self.subTest(node=name):
                    seen = {}
                    for seed in range(10):
                        prompt = prepare(name, seed, "old", "populate", "__colors__")
                        result = output(name, prompt["1"]["inputs"])
                        key = asyncio.run(cache_keys(prompt))
                        if result in seen:
                            self.assertEqual(key, seen[result])
                        else:
                            for other in seen.values():
                                self.assertNotEqual(key, other)
                        seen[result] = key
                    self.assertGreater(len(seen), 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WildcardCacheTests))
    if args.report:
        args.report.write_text(json.dumps({
            "test_groups": result.testsRun,
            "failures": [{"test": test.id(), "details": details} for test, details in result.failures],
            "errors": [{"test": test.id(), "details": details} for test, details in result.errors],
            "skipped": [(test.id(), reason) for test, reason in result.skipped],
            "successful": result.wasSuccessful(),
        }, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
