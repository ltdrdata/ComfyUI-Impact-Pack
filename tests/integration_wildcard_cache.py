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
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


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
    from comfy_api.latest import io
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_primitive import Float, StringMultiline
    from comfy_extras.nodes_string import StringConcatenate
    from execution import IsChangedCache, validate_prompt
    # Match server startup order; importing impact_pack first creates a cycle.
    from impact import impact_server, impact_pack, logics, wildcards

    # Text providers whose output may change while their inputs stay equal, like file loaders.
    class ChangingTextSource:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"path": ("STRING",)}}

        @classmethod
        def IS_CHANGED(cls, path):
            return float("NaN")

    class ChangingTextSourceV3(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="ChangingTextSourceV3", inputs=[io.String.Input("path")], outputs=[io.String.Output()])

        @classmethod
        def fingerprint_inputs(cls, path):
            return float("NaN")

    # Breaks ComfyUI's cache contract: its text may change without declaring IS_CHANGED.
    class UndeclaredTextSource:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"path": ("STRING",)}}

    classes = {name: getattr(impact_pack, name) for name in (
        "ImpactWildcardProcessor", "ImpactWildcardEncode",
    )}
    nodes.NODE_CLASS_MAPPINGS.update(classes | {
        "PreviewAny": PreviewAny, "PrimitiveStringMultiline": StringMultiline, "StringConcatenate": StringConcatenate,
        "ChangingTextSource": ChangingTextSource, "ChangingTextSourceV3": ChangingTextSourceV3,
        "UndeclaredTextSource": UndeclaredTextSource, "PrimitiveFloat": Float,
        "ImpactConvertDataType": logics.ImpactConvertDataType,
    })
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
        # Linked populated_text cannot be populated before execution.
        populated = mode in ("populate", True) and isinstance(text, str)
        if populated:
            expected_widgets[1:3] = [inputs["populated_text"], "reproduce"]
        assert workflow_node["widgets_values"] == expected_widgets
        mode_feedback = [data["value"] for _, data in feedback if data["widget_name"] == "mode"]
        assert mode_feedback == (["populate"] if populated or mode == "reproduce" else [])
        assert all(data["widget_name"] != "seed" for _, data in feedback)
        return prompt

    def output(name, inputs, resolved_text=None, resolved_seed=None, prompt=None):
        values = dict(inputs)
        if resolved_text is not None:
            values["populated_text"] = resolved_text
        if resolved_seed is not None:
            values["seed"] = resolved_seed
        if name == "ImpactWildcardEncode":
            values.update(model=object(), clip=Clip())
        if prompt is not None:
            # Hidden inputs supplied by ComfyUI during execution.
            values.update(prompt=prompt, unique_id="1")
        return classes[name]().doit(**values)[-1]

    def resolve(source):
        return nodes.NODE_CLASS_MAPPINGS[source["class_type"]].execute(**source["inputs"])[0]

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
        def test_file_loading_modes_preserve_seeded_output(self):
            # Exercise the real startup loader, lazy TXT lookup and preloaded
            # YAML through the hook, not just an injected wildcard dictionary.
            with TemporaryDirectory() as directory, patch.multiple(
                    wildcards, wildcards_path=directory, wildcard_dict={},
                    available_wildcards={}, loaded_wildcards={}, _on_demand_mode=False):
                Path(directory, "colors.txt").write_text("red\ngreen\nblue\n")
                Path(directory, "palette.yaml").write_text("palette:\n  colors: [red, green, blue]\n")
                for on_demand in (False, True):
                    with patch.object(wildcards.config, "get_config", return_value={
                            "custom_wildcards": "", "wildcard_cache_limit_mb": 0 if on_demand else 50}):
                        wildcards.wildcard_load()
                    self.assertEqual(wildcards.is_on_demand_mode(), on_demand)
                    if on_demand:
                        self.assertNotIn("colors", wildcards.loaded_wildcards)
                        self.assertIn("palette/colors", wildcards.loaded_wildcards)
                    for name in classes:
                        for mode in ("populate", "fixed"):
                            for text in ("__colors__", "__palette/colors__"):
                                with self.subTest(node=name, mode=mode, text=text, on_demand=on_demand):
                                    for seed, expected in ((2, "red"), (5, "blue")):
                                        previous = "old result" if mode == "populate" else text
                                        inputs = prepare(name, seed, previous, mode, text)["1"]["inputs"]
                                        self.assertEqual(output(name, inputs), expected)
                    if on_demand:
                        self.assertEqual(wildcards.loaded_wildcards["colors"], ["red", "green", "blue"])

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
                # Unregistered descriptors model opaque providers, not real nodes.
                # Their resolved output is supplied below; upstream execution is outside this test.
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

        def test_linked_static_text_reuses_learned_seed(self):
            sources = (
                {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "plain α"}},
                {"class_type": "StringConcatenate", "inputs": {"string_a": "plain", "string_b": "α", "delimiter": " "}},
            )
            for name in classes:
                for mode in ("populate", "fixed", "reproduce"):
                    for source in sources:
                        with self.subTest(node=name, mode=mode, source=source["class_type"]):
                            wildcards.static_linked_text_seeds.clear()

                            def linked(seed, upstream=source):
                                return prepare(name, seed, ["text", 0], mode, upstream={"text": upstream})

                            # Nothing is known about the text before the first run.
                            first = linked(731)
                            self.assertEqual(first["1"]["inputs"]["seed"], 731)
                            self.assertEqual(output(name, first["1"]["inputs"], resolve(source), prompt=first), resolve(source))

                            second = linked(732)
                            self.assertEqual(second["1"]["inputs"]["seed"], 731)
                            self.assertEqual(asyncio.run(cache_keys(first)), asyncio.run(cache_keys(second)))

                            changed = deepcopy(source)
                            changed["inputs"][next(iter(changed["inputs"]))] = "changed"
                            third = linked(733, changed)
                            self.assertEqual(third["1"]["inputs"]["seed"], 733)
                            self.assertNotEqual(asyncio.run(cache_keys(first)), asyncio.run(cache_keys(third)))

                            # Only the last run is kept; ComfyUI's cache drops the earlier text too.
                            output(name, third["1"]["inputs"], resolve(changed), prompt=third)
                            self.assertEqual(linked(734)["1"]["inputs"]["seed"], 734)

        def test_linked_dynamic_text_keeps_seeds_of_queued_runs(self):
            sources = (
                {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "__colors__"}},
                {"class_type": "StringConcatenate", "inputs": {"string_a": "__", "string_b": "colors__", "delimiter": ""}},
            )
            for name in classes:
                for mode in ("populate", "fixed", "reproduce"):
                    for source in sources:
                        with self.subTest(node=name, mode=mode, source=source["class_type"]):
                            wildcards.static_linked_text_seeds.clear()
                            for _ in range(2):
                                # Queue every prompt before any of them runs.
                                queued = [prepare(name, seed, ["text", 0], mode, upstream={"text": source}) for seed in seeds]
                                self.assertEqual([p["1"]["inputs"]["seed"] for p in queued], list(seeds))
                                actual = tuple(output(name, p["1"]["inputs"], resolve(source), prompt=p) for p in queued)
                                self.assertEqual(actual, selections["__colors__"])
                            self.assertEqual(wildcards.static_linked_text_seeds, {})

        def test_linked_text_from_changing_source_keeps_seed(self):
            def file_source(class_type):
                return {"class_type": class_type, "inputs": {"path": "prompt.txt"}}

            upstreams = {
                "IS_CHANGED": {"text": file_source("ChangingTextSource")},
                "fingerprint_inputs": {"text": file_source("ChangingTextSourceV3")},
                "unregistered": {"text": file_source("ExternalTextReader")},
                "nested IS_CHANGED": {
                    "text": {"class_type": "StringConcatenate", "inputs": {"string_a": ["file", 0], "string_b": "α", "delimiter": " "}},
                    "file": file_source("ChangingTextSource"),
                },
            }
            for name in classes:
                for kind, upstream in upstreams.items():
                    with self.subTest(node=name, upstream=kind):
                        wildcards.static_linked_text_seeds.clear()
                        first = prepare(name, 731, ["text", 0], upstream=upstream)
                        self.assertEqual(output(name, first["1"]["inputs"], "plain α", prompt=first), "plain α")
                        self.assertEqual(wildcards.static_linked_text_seeds, {})
                        self.assertEqual(prepare(name, 732, ["text", 0], upstream=upstream)["1"]["inputs"]["seed"], 732)

        def test_validated_upstream_values_keep_fingerprint(self):
            # The hook sees the submitted prompt, the node sees it after ComfyUI's validation
            # cast the FLOAT (JSON sends 1.0 as 1) and unwrapped the __value__ widget.
            name = "ImpactWildcardProcessor"  # Encode needs a model loader to validate
            upstream = {
                "float": {"class_type": "PrimitiveFloat", "inputs": {"value": 1}},
                "number": {"class_type": "ImpactConvertDataType", "inputs": {"value": ["float", 0]}},
                "text": {"class_type": "StringConcatenate", "inputs": {
                    "string_a": {"__value__": "plain"}, "string_b": ["number", 0], "delimiter": " "}},
                "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
            }
            for mode in ("populate", "fixed", "reproduce"):
                with self.subTest(mode=mode):
                    wildcards.static_linked_text_seeds.clear()

                    def submit(seed):
                        prompt = prepare(name, seed, ["text", 0], mode, upstream=upstream)
                        prompt["1"]["inputs"]["Select to add Wildcard"] = "Select the Wildcard to add to the text"
                        self.assertTrue(asyncio.run(validate_prompt("test", prompt, None))[0])
                        return prompt

                    first = submit(731)
                    self.assertIsInstance(first["float"]["inputs"]["value"], float)
                    self.assertEqual(first["text"]["inputs"]["string_a"], "plain")
                    self.assertEqual(output(name, first["1"]["inputs"], "plain 1.0", prompt=first), "plain 1.0")
                    self.assertEqual(submit(732)["1"]["inputs"]["seed"], 731)

        def test_list_widget_values_are_not_followed_as_links(self):
            upstream = {"text": {"class_type": "UndeclaredTextSource", "inputs": {"path": "prompt.txt", "points": [[0, 0], [1, 1]]}}}
            for name in classes:
                with self.subTest(node=name):
                    prompt = prepare(name, 731, ["text", 0], upstream=upstream)
                    self.assertIsNotNone(wildcards.linked_text_fingerprint(prompt, "1"))

        def test_recorded_seed_is_dropped_when_text_gains_syntax(self):
            upstream = {"text": {"class_type": "UndeclaredTextSource", "inputs": {"path": "prompt.txt"}}}
            for name in classes:
                with self.subTest(node=name):
                    wildcards.static_linked_text_seeds.clear()
                    first = prepare(name, 731, ["text", 0], upstream=upstream)
                    output(name, first["1"]["inputs"], "plain α", prompt=first)
                    second = prepare(name, 732, ["text", 0], upstream=upstream)
                    self.assertEqual(second["1"]["inputs"]["seed"], 731)
                    # Same inputs, different text: only a contract-breaking upstream node does this.
                    output(name, second["1"]["inputs"], "__colors__", prompt=second)
                    self.assertEqual(prepare(name, 733, ["text", 0], upstream=upstream)["1"]["inputs"]["seed"], 733)

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
