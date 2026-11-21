import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

NEW_NODE_WIDGETS = {node["type"]: {widget["name"]: widget["value"] for widget in node["widgets"]}
                    for node in json.loads((Path(__file__).parent / "workflows" / "all-impact-nodes.widgets.json").read_text())["nodes"]}


@pytest.fixture(scope="module")
def impact_pack():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import impact_pack
        yield impact_pack


@pytest.mark.parametrize("node_type", ["MaskRectArea", "MaskRectAreaAdvanced"])
def test_the_served_defaults_are_the_values_of_a_new_node(impact_pack, node_type):
    served = {name: spec[1]["default"] for name, spec in getattr(impact_pack, node_type).INPUT_TYPES()["required"].items()}

    assert served == {name: NEW_NODE_WIDGETS[node_type][name] for name in served}


def test_a_prompt_with_the_served_defaults_gives_the_mask_of_a_new_node(impact_pack):
    node = impact_pack.MaskRectAreaAdvanced
    served = {name: spec[1]["default"] for name, spec in node.INPUT_TYPES()["required"].items()}

    mask = node().create_mask_advanced(**served, extra_pnginfo=None, unique_id="1")[0]

    assert mask.shape == (1, 512, 512)
    assert mask[0, :256, :256].min() == 1.0 and mask.sum() == 256 * 256
