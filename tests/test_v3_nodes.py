"""Nodes rewritten on the V3 schema stay the nodes they were.

`v1_node_info.json` holds what `/object_info` returned for each of them while it was a V1 node. Before
a node is rewritten its record is added there, taken from a server that runs the V1 node, and never edited after.
The wording of a description or a tooltip is not held to the record: it may be corrected.
"""
import ast
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1 = json.loads((ROOT / "tests" / "v1_node_info.json").read_text())
# Nodes that were written on the V3 schema from the start.
BORN_V3 = {"DetailerAdvancedOptionsHook", "ImpactSAM3VideoDetectorSEGS"}
FIELDS = ("name", "display_name", "category", "input", "input_order", "is_input_list", "output", "output_name",
          "output_is_list", "output_node", "deprecated", "experimental", "dev_only", "api_node",
          "has_intermediate_output", "search_aliases", "essentials_category")
# Rewritten nodes whose result is not shared between nodes. `/object_info` does not carry this; the cache reads it from the class.
NOT_IDEMPOTENT = {"BlackPatchRetryHookProvider"}


@pytest.fixture(scope="module")
def registered():
    """The classes the pack registers, by node ID."""
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        init = ast.parse((ROOT / "__init__.py").read_text())
        modules = [importlib.import_module("impact." + node.module.rsplit(".", 1)[-1]) for node in init.body
                   if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("modules.impact.")]
        assert Path(modules[0].__file__).resolve().parents[2] == ROOT
        mapping = next(node.value for node in init.body
                       if isinstance(node, ast.Assign) and node.targets[0].id == "NODE_CLASS_MAPPINGS")
        # A later star import of __init__.py replaces an earlier name.
        yield {key.value: next(getattr(module, value.id) for module in reversed(modules) if hasattr(module, value.id))
               for key, value in zip(mapping.keys, mapping.values)}


def described(info):
    """A node description with what the V3 schema writes differently put back into the V1 form."""
    info = json.loads(json.dumps(info))
    inputs = {}
    for section, entries in info["input"].items():
        inputs[section] = {}
        for name, entry in entries.items():
            kind, options = entry[0], dict(entry[1]) if len(entry) > 1 else {}
            if kind == "COMBO":
                kind = options.pop("options")
                assert options.pop("multiselect") is False
            options.pop("tooltip", None)
            inputs[section][name] = [kind, options]
    return {**{field: info.get(field) or None for field in FIELDS}, "input": inputs}


@pytest.mark.parametrize("node_id", sorted(V1))
def test_a_rewritten_node_is_described_as_its_v1_node_was(registered, node_id):
    node = registered[node_id]
    assert hasattr(node, "define_schema"), "the V1 record of a node is kept only once the node is rewritten"
    new, old = described(node.GET_NODE_INFO_V1()), described(V1[node_id])
    assert new == old
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True), "a value changed its type, as 7 to 7.0"


@pytest.mark.parametrize("node_id", sorted(V1))
def test_a_rewritten_node_takes_the_inputs_of_its_v1_node(registered, node_id):
    # A prompt passes every input by its name and leaves out an optional input that is not connected; `doit` is also
    # called with the inputs in their order.
    node, inputs = registered[node_id], V1[node_id]["input"]
    parameters = inspect.signature(node.execute).parameters
    assert list(parameters) == list(inputs.get("required", {})) + list(inputs.get("optional", {}))
    assert all(parameters[name].default is not inspect.Parameter.empty for name in inputs.get("optional", {}))
    assert not getattr(node, "ACCEPT_ALL_INPUTS", False)  # ComfyUI has the flag from 0.11.0


def test_every_rewritten_node_has_a_v1_record(registered):
    rewritten = {node_id for node_id, node in registered.items() if hasattr(node, "define_schema")}
    assert rewritten - BORN_V3 == set(V1)


def test_a_rewritten_node_can_still_be_called_the_v1_way(registered):
    pipe = ("model", "clip", "vae", "positive", "negative")
    assert registered["EmptySegs"]().doit() == (((0, 0), []),)
    assert registered["FromBasicPipe"]().doit(pipe) == pipe
    assert registered["FromBasicPipe_v2"]().doit(basic_pipe=pipe) == (pipe, *pipe)


def test_a_node_that_did_not_return_a_tuple_as_a_v1_node_still_returns_what_it_did(registered):
    region = [1, 2, 3, 4]
    assert registered["ImpactFrom_SEG_ELT_crop_region"]().doit(region) is region
    assert registered["ImpactFrom_SEG_ELT_crop_region"].execute(region).result == (1, 2, 3, 4)
    box = registered["ImpactFrom_SEG_ELT_bbox"]().doit((1.0, 2, 3, 4))
    assert box == [1, 2, 3, 4] and all(type(value) is int for value in box)
    result = registered["ImpactFrom_SEG_ELT_bbox"].execute((1.0, 2, 3, 4)).result
    assert result == (1, 2, 3, 4) and all(type(value) is int for value in result)


def test_a_node_that_passes_on_the_results_of_a_call_gives_each_as_an_output(registered, monkeypatch):
    scale, reencode = registered["LatentPixelScale"], registered["ReencodeLatentPipe"]
    monkeypatch.setattr(sys.modules[scale.__module__].core, "latent_upscale_on_pixel_space2", lambda *args, **kwargs: ("latent", "image"))
    assert scale.doit("samples", "bilinear", 1.5, "vae", False) == ("latent", "image")
    monkeypatch.setattr(sys.modules[reencode.__module__].ReencodeLatent, "doit", lambda self, *args: (args,))
    in_pipe, out_pipe = ("model", "clip", "vae in", "positive", "negative"), ("model", "clip", "vae out", "positive", "negative")
    assert reencode.doit("samples", "None", in_pipe, out_pipe) == (("samples", "None", "vae in", "vae out"),)
    segs = ((8, 8), [])
    assert registered["BitwiseAndMaskForEach"].doit(segs, segs) == (segs,)


def test_only_a_node_whose_result_must_not_be_shared_between_nodes_says_so(registered):
    assert {node_id for node_id in V1 if registered[node_id].NOT_IDEMPOTENT} == NOT_IDEMPOTENT
