"""Native import/Export/reload of pinned hook tutorials; no model inference.

IMPACT_TUTORIALS_DIR and an isolated IMPACT_E2E_BASE_URL are required.
Unknown unrelated tutorial plugins are tolerated; every target must be registered.
"""
from argparse import Namespace
import json
import os
from pathlib import Path

import pytest

from detailer_hook_cases import CASES, expected_widgets, targets, tutorial
from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, full_zoom, setting, show_node, widget_state


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("case", CASES)
def test_original_hook_tutorial_roundtrip(tmp_path, case, restore_by_name):
    source, original = tutorial(case)
    nodes = targets(original)
    assert nodes, case
    ids = {str(n["id"]) for n in nodes}
    hook_ids = {str(n["id"]) for n in nodes if "Hook" in n["type"]}
    expected = {str(n["id"]): expected_widgets(n) for n in nodes}
    # Legacy workflow/MAKE_BASIC_PIPE components may be unavailable. Their
    # model wiring is outside this loading test; all hook edges remain required.
    def hook_links(graph):
        return sorted(link for link in graph["links"] if str(link[1]) in hook_ids or str(link[3]) in hook_ids
                      or link[5] in ("DETAILER_HOOK", "PK_HOOK", "UPSCALER_HOOK"))
    links = hook_links(original)
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / f"{case}-{restore_by_name}")
    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        for roundtrip in range(2):
            ui.new_workflow()
            ui.load_workflow(source, len(original["nodes"]), check_node_errors=False)
            full_zoom(ui)
            states = []
            for node in nodes:
                node_id = str(node["id"])
                show_node(ui, node_id)
                state = widget_state(ui, node_id)
                assert state["type"] == node["type"]
                assert json.dumps(state["widgets"], sort_keys=True) == json.dumps(expected[node_id], sort_keys=True)
                states.append(state)
            (args.artifacts / f"widgets-{roundtrip}.json").write_text(json.dumps(states, indent=2) + "\n")
            source = args.artifacts / f"export-{roundtrip}.json"
            export_workflow(ui, source)
            exported = json.loads(source.read_text())
            for node in exported["nodes"]:
                if str(node["id"]) in ids:
                    values = [w["value"] for w in expected[str(node["id"])]]
                    assert json.dumps(node.get("widgets_values") or []) == json.dumps(values)
            actual_links = hook_links(exported)
            assert actual_links == links, "Hook wiring or consumer inputs changed during native loading"
