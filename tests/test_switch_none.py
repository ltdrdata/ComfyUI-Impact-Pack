import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def switch():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import impact_server  # noqa: F401
        from impact import util_nodes
        yield SimpleNamespace(doit=util_nodes.GeneralSwitch.doit, sent=server.send_sync)


def test_a_select_without_a_link_gives_none_and_says_which_switch(switch, caplog):
    caplog.set_level(logging.INFO)
    result = switch.doit(select=9, sel_mode=False, input1=None, input2=None, unique_id="12", extra_pnginfo=None)
    assert result == (None, "", 9)
    assert "Switch (Any) 12: select is 9 and input9 has no link. It returns None. Linked inputs: input1, input2." in caplog.text
    switch.sent.assert_called_once_with("impact-switch-none", {"node_id": "12", "select": 9}, "page")


def test_a_selected_input_that_gave_none_is_passed_on_and_reported(switch, caplog):
    caplog.set_level(logging.INFO)
    assert switch.doit(select=2, sel_mode=False, input1=None, input2=None, unique_id="12", extra_pnginfo=None) == (None, "input2", 2)
    assert "Switch (Any) 12: select is 2 and input2 gave None. It returns None." in caplog.text
    switch.sent.assert_called_once_with("impact-switch-none", {"node_id": "12", "select": 2}, "page")


def test_a_select_with_a_link_gives_its_value_and_says_nothing(switch, caplog):
    caplog.set_level(logging.INFO)
    value = object()
    assert switch.doit(select=2, sel_mode=False, input1=None, input2=value, unique_id="12", extra_pnginfo=None) == (value, "input2", 2)
    assert "returns None" not in caplog.text
    switch.sent.assert_not_called()
