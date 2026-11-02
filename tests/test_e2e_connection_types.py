"""Native loading of consumers created before their upstream dynamic outputs."""
import json

import pytest

from test_e2e_switches import ui as ui
from test_e2e_widget_catalog import export_workflow, rendered, setting, show_node, widget_state


TARGETS = [
    ("ImpactControlBridge", "Control Bridge"),
    ("ImpactConditionalBranch", "ImpactConditionalBranch"),
    ("ImpactConditionalBranchSelMode", "ImpactConditionalBranchSelMode"),
    ("ImpactCompare", "ImpactCompare"),
    ("ImpactSelectNthItemOfAnyList", "Select Nth Item (Any list)"),
]


def widgets(browser, node_ids):
    result = []
    for node_id in node_ids:
        show_node(browser, node_id)
        result.append(widget_state(browser, node_id))
    return result


@pytest.mark.parametrize("node_type,title", TARGETS, ids=[node[0] for node in TARGETS])
@pytest.mark.parametrize("source_first", [False, True], ids=["consumer-first", "source-first"])
@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_dynamic_output_type_survives_native_loading(ui, node_type, title, source_first, restore_by_name):
    browser, args = ui
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    browser.new_workflow()
    if not source_first:
        target = browser.add_node(title, node_type, 1000, 200)
    inverse = browser.add_node("Inversed Switch (Any)", "ImpactInversedSwitch", 580, 300)
    source = browser.add_node("ImpactInt", "ImpactInt", 200, 300)
    if source_first:
        target = browser.add_node(title, node_type, 1000, 200)
    browser.add_node("Preview as Text", "PreviewAny", 1000, 550)
    browser.set_number("ImpactInt", "value", 42)
    browser.connect("ImpactInt", "ImpactInversedSwitch")
    browser.connect("ImpactInversedSwitch", "PreviewAny")
    browser.connect("ImpactInversedSwitch", node_type, source_slot=1)
    rendered(browser)
    browser.set_number("ImpactInversedSwitch", "select", 2)
    if node_type == "ImpactSelectNthItemOfAnyList":
        browser.set_number(node_type, "index", -1)
    elif node_type == "ImpactCompare":
        browser.choose(node_type, "cmp", "a > b")
    elif node_type.startswith("ImpactConditionalBranch"):
        browser.page.mouse.click(*browser.point(node_type, widget="cond"))

    node_ids = [target, inverse, source]
    before = widgets(browser, node_ids)
    original_path = args.artifacts / "original.json"
    exported_path = args.artifacts / "exported.json"
    export_workflow(browser, original_path)
    original = json.loads(original_path.read_text())
    order = [str(node["id"]) for node in original["nodes"]]
    assert (order.index(source) < order.index(target)) == source_first
    connection = next(link for link in original["links"] if str(link[3]) == target)
    assert str(connection[1]) == inverse and connection[2] == 1 and connection[5] == "INT"

    observations = []
    for path in (original_path, exported_path):
        browser.new_workflow()
        browser.load_workflow(path, len(original["nodes"]))
        after = widgets(browser, node_ids)
        observations.append(after)
        assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
        types = browser.app.evaluate("""(app, id) => {
            const node = app.graph.getNodeById(id);
            return {inputs: node.inputs.map(slot => slot.type ?? null),
                    outputs: node.outputs.map(slot => slot.type ?? null)};
        }""", target)
        assert types["inputs"][0] == "INT"
        assert types["outputs"][0] == ("BOOLEAN" if node_type == "ImpactCompare" else "INT")
        assert None not in types["inputs"] + types["outputs"]
        if path == original_path:
            export_workflow(browser, exported_path)
    (args.artifacts / "widgets.json").write_text(json.dumps([before, *observations], indent=2) + "\n")
    exported = json.loads(exported_path.read_text())
    for before_node, after_node in zip(original["nodes"], exported["nodes"], strict=True):
        assert before_node["id"] == after_node["id"]
        for key in ("widgets_values", "widgets_values_named"):
            assert json.dumps(before_node.get(key), sort_keys=True) == json.dumps(after_node.get(key), sort_keys=True)
    assert original["links"] == exported["links"]
