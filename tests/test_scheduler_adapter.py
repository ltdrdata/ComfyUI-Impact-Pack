import asyncio
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock()), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import impact_pack
        import comfy.samplers
        import execution
        import nodes
        yield SimpleNamespace(pack=impact_pack, samplers=comfy.samplers, execution=execution, nodes=nodes)


class AnyTypeOutput:
    RETURN_TYPES = ("*",)
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


def test_the_adapter_links_into_a_detailer_after_another_nodepack_adds_a_scheduler(runtime, monkeypatch):
    for name, node in (("AnyTypeOutput", AnyTypeOutput), ("ImpactSchedulerAdapter", runtime.pack.ImpactSchedulerAdapter), ("FaceDetailer", runtime.pack.FaceDetailer)):
        monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, name, node)
    monkeypatch.setitem(runtime.samplers.SCHEDULER_HANDLERS, "added_later", runtime.samplers.SCHEDULER_HANDLERS["normal"])

    detailer = {"wildcard": "", "scheduler": ["adapter", 0]}
    for name, (kind, *options) in runtime.pack.FaceDetailer.INPUT_TYPES()["required"].items():
        if name not in detailer:
            detailer[name] = kind[0] if isinstance(kind, list) else options[0]["default"] if options and "default" in options[0] else ["any", 0]
    prompt = {
        "any": {"class_type": "AnyTypeOutput", "inputs": {}},
        "adapter": {"class_type": "ImpactSchedulerAdapter", "inputs": {"scheduler": "normal", "extra_scheduler": "None"}},
        "detailer": {"class_type": "FaceDetailer", "inputs": detailer},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["detailer", 0]}},
    }
    valid, error, outputs, node_errors = asyncio.run(runtime.execution.validate_prompt("late-scheduler", prompt, None))
    assert valid and node_errors == {}, node_errors
