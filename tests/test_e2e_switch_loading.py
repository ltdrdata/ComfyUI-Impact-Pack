"""Preserve Switch selection when loading workflows with unused inputs.

These Chrome checks pass on Main and detect the automatic selection changes in
PR #1252. The existing catalog and its fixed widget snapshot remain read-only.
"""
import json
from pathlib import Path

import pytest

from test_e2e_switches import ui as ui
from test_e2e_widget_catalog import export_workflow, full_zoom, setting, show_node, widget_state


WORKFLOWS = Path(__file__).parent / "workflows"
SWITCH_TYPES = {"ImpactSwitch", "LatentSwitch", "SEGSSwitch"}


def assert_same_values(actual, expected):
    # Unlike Python equality, JSON distinguishes True from 1 and False from 0.
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_unconnected_switch_widgets_survive_native_round_trip(ui, restore_by_name):
    browser, args = ui
    path = WORKFLOWS / "all-impact-nodes.json"
    workflow = json.loads(path.read_text())
    snapshot = json.loads((WORKFLOWS / "all-impact-nodes.widgets.json").read_text())
    expected = [node for node in snapshot["nodes"] if node["type"] in SWITCH_TYPES]
    assert {node["type"] for node in expected} == SWITCH_TYPES
    original = {str(node["id"]): node["widgets_values"] for node in workflow["nodes"]
                if node["type"] in SWITCH_TYPES}
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    exported_path = args.artifacts / "exported.json"
    observations = []
    for source in (path, exported_path):
        browser.new_workflow()
        browser.load_workflow(source, len(workflow["nodes"]), check_node_errors=False)
        full_zoom(browser)
        actual = []
        for node in expected:
            show_node(browser, node["id"])
            actual.append(widget_state(browser, node["id"]))
        observations.append(actual)
        if source == path:
            export_workflow(browser, exported_path)
    (args.artifacts / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    exported = json.loads(exported_path.read_text())
    saved = {str(node["id"]): node["widgets_values"] for node in exported["nodes"]
             if node["type"] in SWITCH_TYPES}
    assert_same_values(observations, [expected, expected])
    assert_same_values(saved, original)


@pytest.mark.parametrize("fixture,selection,expected_text", [
    ("switch-unused-inputs-linked.json", 2, ["second"]),
    ("switch-unused-inputs-selected-tail.json", 5, ["None"]),
    ("switch-unused-inputs-selected-middle.json", 2, ["None"]),
], ids=["linked", "selected_empty_tail", "selected_empty_middle"])
@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_unused_switch_inputs_preserve_selection_and_output(ui, fixture, selection, expected_text,
                                                           restore_by_name):
    browser, args = ui
    path = WORKFLOWS / fixture
    workflow = json.loads(path.read_text())
    target = next(node for node in workflow["nodes"] if node["type"] == "ImpactSwitch")
    assert_same_values(target["widgets_values"], [selection, False])
    assert len([slot for slot in target["inputs"] if slot["name"].startswith("input")]) == 16
    expected_widgets = [{"name": "select", "value": selection}, {"name": "sel_mode", "value": False}]
    expected_links = {slot["name"]: slot["link"] for slot in target["inputs"] if slot["link"] is not None}
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    exported_path = args.artifacts / "exported.json"
    observations = []
    for source in (path, exported_path):
        browser.new_workflow()
        browser.load_workflow(source, len(workflow["nodes"]))
        show_node(browser, str(target["id"]))
        state = widget_state(browser, str(target["id"]))
        state["links"] = browser.app.evaluate("""(app, id) => Object.fromEntries(
            app.graph.getNodeById(id).inputs.filter(slot => slot.link != null)
                .map(slot => [slot.name, slot.link]))""", target["id"])
        state["text"] = browser.run()["outputs"]["20"]["text"]
        observations.append(state)
        if source == path:
            export_workflow(browser, exported_path)
    (args.artifacts / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    exported = json.loads(exported_path.read_text())
    saved = next(node for node in exported["nodes"] if node["id"] == target["id"])
    for state in observations:
        assert_same_values(state["widgets"], expected_widgets)
        assert state["links"] == expected_links
        assert state["text"] == expected_text
    assert_same_values(saved["widgets_values"], target["widgets_values"])
    # A safe cleanup may remove unused sockets; connected slots and widget values
    # must retain their meaning. Do not require the bloated socket count to persist.
    assert_same_values(observations[0], observations[1])
