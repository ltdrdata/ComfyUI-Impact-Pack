"""Import original tutorial PNGs, then use native Export and reload in Chrome.

Requires IMPACT_TUTORIALS_DIR. This checks the affected registered node even
when unrelated tutorial plugins are unavailable; it does not execute those
incomplete graphs. The full LBW inference check additionally needs a trusted
Main output in IMPACT_E2E_REFERENCE_IMAGE and the original models/Inspire Pack.
"""
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from test_e2e_wildcard_cache import ui as ui  # shared browser fixture
from test_e2e_widget_catalog import ROOT, export_workflow, setting, show_node, widget_state
from wildcard_tutorial_cases import CASES, tutorial


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("case", CASES)
def test_native_tutorial_widgets(ui, case, restore_by_name):
    path, original, target = tutorial(os.environ["IMPACT_TUTORIALS_DIR"], case)
    node_id = str(target["id"])
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    ui.load_workflow(path, len(original["nodes"]), check_node_errors=False)
    show_node(ui, node_id)
    before = widget_state(ui, node_id)
    catalog = json.loads((ROOT / "tests/workflows/all-impact-nodes.widgets.json").read_text())
    reference = next(node for node in catalog["nodes"] if node["type"] == target["type"])
    assert [w["name"] for w in before["widgets"]] == [w["name"] for w in reference["widgets"]]

    saved = target["widgets_values"]
    # Explicit legacy behavior, checked against Main. "Fixed" remains invalid
    # for execution in the oldest tutorial; do not silently bless a migration.
    mode = ("populate" if saved[2] else "fixed") if isinstance(saved[2], bool) else saved[2]
    expected = [widget["value"] for widget in reference["widgets"]]
    expected[:3] = [*saved[:2], mode]
    if target["type"] == "ImpactWildcardEncode":
        expected[5:] = saved[5:]
    # JSON equality distinguishes false from zero, unlike Python list equality.
    expected_values = json.dumps(expected)
    assert json.dumps([w["value"] for w in before["widgets"]]) == expected_values

    first = ui.artifacts / "export.json"
    export_workflow(ui, first)
    exported = json.loads(first.read_text())
    exported_target = next(n for n in exported["nodes"] if str(n["id"]) == node_id)
    assert json.dumps(exported_target["widgets_values"]) == expected_values
    ui.new_workflow()
    ui.load_workflow(first, len(original["nodes"]), check_node_errors=False)
    show_node(ui, node_id)
    assert json.dumps(widget_state(ui, node_id), sort_keys=True) == json.dumps(before, sort_keys=True)
    second = ui.artifacts / "roundtrip.json"
    export_workflow(ui, second)
    restored = json.loads(second.read_text())
    restored_target = next(n for n in restored["nodes"] if str(n["id"]) == node_id)
    assert json.dumps(restored_target["widgets_values"]) == expected_values
    assert sorted(restored["links"]) == sorted(exported["links"])
    original_links = sorted(link for link in original["links"] if node_id in (str(link[1]), str(link[3])))
    for graph in (exported, restored):
        assert sorted(link for link in graph["links"] if node_id in (str(link[1]), str(link[3]))) == original_links
    (ui.artifacts / "widget-state.json").write_text(json.dumps(before, indent=2) + "\n")


def test_lbw_tutorial_inference(ui):
    reference = Path(os.environ["IMPACT_E2E_REFERENCE_IMAGE"])
    assert reference.is_file(), "Capture the original PNG's output with trusted Main first"
    path, workflow, target = tutorial(os.environ["IMPACT_TUTORIALS_DIR"], "lbw")
    ui.load_workflow(path, len(workflow["nodes"]))
    ui.timeout = 900
    entry = ui.run()
    (ui.artifacts / "history.json").write_text(json.dumps(entry, indent=2) + "\n")
    actual = ui.images(entry)["9"][0]
    actual.save(ui.artifacts / "output.png")
    with Image.open(reference) as image:
        assert actual.size == image.size
        assert actual.tobytes() == image.convert("RGB").tobytes()
    saved = next(n for n in entry["prompt"][3]["extra_pnginfo"]["workflow"]["nodes"]
                 if str(n["id"]) == str(target["id"]))
    assert saved["widgets_values"][5] == target["widgets_values"][5]
