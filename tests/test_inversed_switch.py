import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def inversed_switch():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from comfy_execution.graph import DynamicPrompt, ExecutionBlocker
        from impact import util_nodes

        def outputs(dynprompt, unique_id, select):
            routed = util_nodes.GeneralInversedSwitch().doit(select=select, dynprompt=dynprompt, unique_id=unique_id, input="routed")
            return ["blocked" if isinstance(value, ExecutionBlocker) else value for value in routed]

        yield SimpleNamespace(outputs=outputs, prompt=DynamicPrompt)


SWITCH = {"class_type": "ImpactInversedSwitch", "inputs": {}}


def test_the_selected_output_gets_the_input_and_the_other_linked_outputs_are_blocked(inversed_switch):
    prompt = inversed_switch.prompt({"5": SWITCH, "6": {"inputs": {"a": ["5", 0]}}, "7": {"inputs": {"a": ["5", 2]}}})
    assert inversed_switch.outputs(prompt, "5", select=3) == ["blocked", "blocked", "routed"]
    assert inversed_switch.outputs(prompt, "5", select=1) == ["routed", "blocked", "blocked"]


def test_a_copy_that_a_loop_made_gives_the_outputs_its_own_followers_take(inversed_switch):
    prompt = inversed_switch.prompt({"5": SWITCH, "6": {"inputs": {"a": ["5", 1]}}})
    prompt.add_ephemeral_node("9.0.0.5", SWITCH, "9", "5")
    prompt.add_ephemeral_node("9.0.0.6", {"inputs": {"a": ["9.0.0.5", 1]}}, "9", "6")
    assert inversed_switch.outputs(prompt, "9.0.0.5", select=2) == ["blocked", "routed"]
