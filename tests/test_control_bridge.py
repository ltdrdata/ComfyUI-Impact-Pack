import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

WORKFLOWS = Path(__file__).parent / "workflows" / "control-bridge"


def workflow(name):
    return json.loads((WORKFLOWS / f"{name}.json").read_text())


@pytest.fixture
def bridge():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        import comfy.model_management
        from comfy_execution.graph import DynamicPrompt
        from impact import core, logics

        def hidden(sent, unique_id, takers=(), shown_as=None):
            prompt = {shown_as or unique_id: {"inputs": {}}} | {taker: {"inputs": {"image": [unique_id, 0]}} for taker in takers}
            dynprompt = DynamicPrompt(prompt)
            if shown_as:
                dynprompt.add_ephemeral_node(unique_id, {"inputs": {}}, None, shown_as)
            given = {"UNIQUE_ID": unique_id, "PROMPT": prompt, "EXTRA_PNGINFO": sent and {"workflow": sent}, "DYNPROMPT": dynprompt}
            return {name: given[kind] for name, kind in logics.ImpactControlBridge.INPUT_TYPES()["hidden"].items()}

        def run(sent, unique_id, mode=False, behavior="Bypass", shown_as=None):
            value = object()
            try:
                assert logics.ImpactControlBridge().doit(value, mode, behavior, **hidden(sent, unique_id, shown_as=shown_as)) == (value,)
            except comfy.model_management.InterruptProcessingException:
                return server.send_sync.call_args_list, True
            return server.send_sync.call_args_list, False

        def asks_for(sent, unique_id, mode=False, behavior="Bypass", takers=(), shown_as=None):
            return logics.ImpactControlBridge().check_lazy_status(None, mode, behavior, **hidden(sent, unique_id, takers, shown_as))

        def is_changed(sent, unique_id, in_prompt=True, takers=()):
            prompt = {taker: {"inputs": {"image": [unique_id, 0]}} for taker in takers} | ({unique_id: {"inputs": {}}} if in_prompt else {})
            patch.setattr(core, "current_prompt", {"prompt": prompt, "extra_data": {"extra_pnginfo": {"workflow": sent}}})
            return logics.ImpactControlBridge.IS_CHANGED(None, False, "Bypass", unique_id=unique_id)

        yield SimpleNamespace(run=run, is_changed=is_changed, asks_for=asks_for)


@pytest.mark.parametrize("behavior,change", [("Bypass", "bypasses"), ("Mute", "mutes")])
def test_a_bridge_in_a_subgraph_changes_the_node_behind_it_in_that_subgraph(bridge, behavior, change):
    sent, interrupted = bridge.run(workflow("in-subgraph"), "22:21", behavior=behavior)
    assert sent == [call("impact-bridge-continue", {"node_id": "22:21", change: ["22:12"]})]
    assert interrupted


def test_an_active_bridge_in_a_subgraph_makes_the_bypassed_node_behind_it_active(bridge):
    bypassed = workflow("in-subgraph")
    bypassed["definitions"]["subgraphs"][0]["nodes"][1]["mode"] = 4
    sent, interrupted = bridge.run(bypassed, "22:21", mode=True)
    assert sent == [call("impact-bridge-continue", {"node_id": "22:21", "actives": ["22:12"]})]
    assert interrupted


def test_a_bridge_whose_node_behind_is_in_the_wanted_mode_passes_its_value(bridge):
    assert bridge.run(workflow("in-subgraph"), "22:21", mode=True) == ([], False)


def test_the_last_node_of_two_subgraphs_changes_the_node_behind_the_outer_subgraph(bridge):
    sent, _ = bridge.run(workflow("last-two-subgraphs-deep"), "23:22:21")
    assert sent == [call("impact-bridge-continue", {"node_id": "23:22:21", "bypasses": ["5"]})]


@pytest.mark.parametrize("unique_id,behind", [("22:21", ["23", "24"]), ("23:21", ["5"])])
def test_a_subgraph_used_twice_changes_the_nodes_behind_the_one_that_ran(bridge, unique_id, behind):
    sent, _ = bridge.run(workflow("subgraph-twice"), unique_id)
    assert sent == [call("impact-bridge-continue", {"node_id": unique_id, "bypasses": behind})]


def test_a_copy_that_a_loop_made_acts_as_the_node_of_the_workflow(bridge):
    sent, interrupted = bridge.run(workflow("loop"), "6.0.0.3", shown_as="3")
    assert sent == [call("impact-bridge-continue", {"node_id": "3", "bypasses": ["4"]})]
    assert interrupted


def test_a_reroute_behind_the_bridge_is_passed_through(bridge):
    sent, _ = bridge.run(workflow("reroute"), "21")
    assert sent == [call("impact-bridge-continue", {"node_id": "21", "bypasses": ["12"]})]


def test_a_bridge_that_nothing_follows_passes_its_value(bridge):
    assert bridge.run(workflow("nothing-behind"), "2") == ([], False)


def test_the_cache_key_names_the_nodes_behind_and_whether_the_value_is_taken_and_makes_a_copy_of_a_loop_run(bridge):
    assert bridge.is_changed(workflow("in-subgraph"), "22:21", takers=["22:12"]) == (["22:12"], True)
    assert bridge.is_changed(workflow("in-subgraph"), "22:21") == (["22:12"], False)
    assert bridge.is_changed(workflow("nothing-behind"), "2") == ([], False)
    assert math.isnan(bridge.is_changed(workflow("loop"), "6.0.0.3", in_prompt=False))


def behind_in_mode(mode):
    sent = workflow("in-subgraph")
    sent["definitions"]["subgraphs"][0]["nodes"][1]["mode"] = mode
    return sent


@pytest.mark.parametrize("mode,asked", [(False, None), (True, ["value"])])
def test_a_bridge_with_stop_asks_for_its_value_only_when_it_is_active(bridge, mode, asked):
    assert bridge.asks_for(workflow("in-subgraph"), "22:21", mode, "Stop", takers=["22:12"]) == asked


@pytest.mark.parametrize("behavior", ["Mute", "Bypass"])
def test_a_bridge_sent_without_its_workflow_asks_for_its_value(bridge, behavior):
    assert bridge.asks_for(None, "3", behavior=behavior) == ["value"]


@pytest.mark.parametrize("mode,behavior,mode_behind", [(False, "Mute", 0), (False, "Bypass", 0), (False, "Mute", 4), (True, "Mute", 2), (True, "Bypass", 4)])
def test_a_bridge_that_is_going_to_change_the_node_behind_it_does_not_ask_for_its_value(bridge, mode, behavior, mode_behind):
    assert bridge.asks_for(behind_in_mode(mode_behind), "22:21", mode, behavior, takers=["22:12"]) is None


def test_a_copy_that_a_loop_made_does_not_ask_for_its_value_before_it_changes_the_node_behind_it(bridge):
    assert bridge.asks_for(workflow("loop"), "6.0.0.3", shown_as="3") is None


@pytest.mark.parametrize("name,unique_id,takers", [("in-subgraph", "22:21", ["22:12"]), ("nothing-behind", "2", [])])
def test_an_active_bridge_with_nothing_to_change_asks_for_its_value(bridge, name, unique_id, takers):
    assert bridge.asks_for(workflow(name), unique_id, mode=True, takers=takers) == ["value"]


@pytest.mark.parametrize("behavior,mode_behind", [("Mute", 2), ("Bypass", 4)])
def test_a_bridge_that_is_off_asks_for_its_value_only_when_a_node_of_the_prompt_takes_it(bridge, behavior, mode_behind):
    assert bridge.asks_for(behind_in_mode(mode_behind), "22:21", behavior=behavior) is None
    assert bridge.asks_for(behind_in_mode(mode_behind), "22:21", behavior=behavior, takers=["22:30"]) == ["value"]
