import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def remote():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import impact_server
        yield SimpleNamespace(apply=impact_server.onprompt_for_remote, sent=server.send_sync)


def test_a_remote_node_aimed_at_a_widget_of_another_type_does_not_stop_the_remote_nodes_after_it(remote):
    prompt = {"1": {"class_type": "ImpactRemoteInt", "inputs": {"node_id": 3, "widget_name": "value", "value": 7}},
              "2": {"class_type": "ImpactRemoteInt", "inputs": {"node_id": 4, "widget_name": "width", "value": 128}},
              "3": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a text"}},
              "4": {"class_type": "EmptyImage", "inputs": {"width": 512, "height": 512, "batch_size": 1, "color": 0}}}
    remote.apply({"prompt": prompt})
    assert prompt["3"]["inputs"]["value"] == "a text"
    assert prompt["4"]["inputs"]["width"] == 128
    remote.sent.assert_called_once_with("impact-node-feedback", {"node_id": "4", "widget_name": "width", "type": "INT", "value": 128})
