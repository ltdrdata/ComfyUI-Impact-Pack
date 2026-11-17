"""Nodes rewritten on the V3 schema return what they returned as V1 nodes.

`v1_node_runs.json` holds a summary of what each of them returned for the inputs of `v1_node_run_cases.py` while it was
a V1 node, through ComfyUI's executor and by `Node().doit(...)`. Run with IMPACT_COMFY_ROOT set.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import v1_node_run_cases as recorded

ROOT = Path(__file__).resolve().parents[1]
V1_INFO = json.loads((ROOT / "tests" / "v1_node_info.json").read_text())
V1_RUNS = json.loads(recorded.RECORD.read_text())


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        import execution
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401 - register routes before the hook import cycle
        from impact import core, hooks, util_nodes
        with recorded.without_models(core, util_nodes):
            yield SimpleNamespace(classes=recorded.node_classes(ROOT), cases=recorded.cases(core, hooks), execution=execution)


@pytest.mark.parametrize("case", sorted(V1_RUNS))
def test_a_rewritten_node_returns_what_its_v1_node_returned(runtime, case):
    node_id, index = case.rsplit("#", 1)
    node = runtime.classes[node_id]
    assert hasattr(node, "define_schema"), "the V1 record of a node is kept only once the node is rewritten"
    result = recorded.run(case, node, runtime.cases[node_id][int(index)], runtime.execution)
    for way in ("executor", "doit"):
        assert recorded.same(result[way], V1_RUNS[case][way]), f"{way}: {json.dumps(result[way])[:600]} was {json.dumps(V1_RUNS[case][way])[:600]}"


def test_every_rewritten_node_has_a_recorded_run_of_every_case(runtime):
    assert {case.rsplit("#", 1)[0] for case in V1_RUNS} == set(V1_INFO)
    assert set(V1_RUNS) == {f"{node_id}#{index}" for node_id, runs in runtime.cases.items() for index in range(len(runs))}
