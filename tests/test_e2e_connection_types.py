"""Native loading of consumers created before their upstream dynamic outputs."""
import json

import pytest
from playwright.sync_api import expect

from e2e_ui import ui as ui
from test_e2e_widget_catalog import export_workflow, rendered, setting, show_node, widget_state


TARGETS = [
    ("ImpactControlBridge", "Control Bridge", [("mode", False), ("behavior", "Mute")]),
    ("ImpactConditionalBranch", "ImpactConditionalBranch", [("cond", True)]),
    ("ImpactConditionalBranchSelMode", "ImpactConditionalBranchSelMode", [("cond", True)]),
    ("ImpactCompare", "ImpactCompare", [("cmp", "a > b")]),
    ("ImpactSelectNthItemOfAnyList", "Select Nth Item (Any list)", [("index", -1)]),
]


def widgets(browser, node_ids):
    result = []
    for node_id in node_ids:
        show_node(browser, node_id)
        result.append(widget_state(browser, node_id))
    return result


def bridge_transition(browser, preview, mode, artifacts, label):
    """Observe the real browser continuation when Bridge changes a downstream mode."""
    def submitted_mode(response, expected):
        if response.request.method != "POST" or not response.url.endswith("/prompt"):
            return False
        workflow = response.request.post_data_json["extra_data"]["extra_pnginfo"]["workflow"]
        return any(str(node["id"]) == preview and node["mode"] == expected for node in workflow["nodes"])

    previous = browser.app.evaluate("(app, id) => app.graph.getNodeById(id).mode", preview)
    assert previous != mode
    with browser.page.expect_response(lambda response: submitted_mode(response, mode)) as continued:
        with browser.page.expect_response(lambda response: submitted_mode(response, previous)) as first:
            browser.page.get_by_test_id("queue-button").click()
    for phase, response in [("initial", first.value), ("continued", continued.value)]:
        assert response.ok, response.text()
        prompt_id = response.json()["prompt_id"]
        entry = browser.page.wait_for_function("""async ({url, id}) => {
            const response = await fetch(`${url}/history/${id}`);
            if (!response.ok) throw new Error(`History request failed: ${response.status}`);
            return (await response.json())[id] ?? false;
        }""", arg={"url": browser.base_url, "id": prompt_id}).json_value()
        (artifacts / f"{label}-{phase}.json").write_text(json.dumps(entry, indent=2) + "\n")
        events = [event for event, _ in entry["status"]["messages"]]
        assert "execution_error" not in events, entry["status"]
        if phase == "continued":
            assert entry["status"]["status_str"] == "success", entry["status"]
            actual = {node: output["text"] for node, output in entry["outputs"].items() if "text" in output}
            assert actual == ({preview: ["42"]} if mode == 0 else {})
    assert browser.app.evaluate("(app, id) => app.graph.getNodeById(id).mode", preview) == mode


@pytest.mark.parametrize("node_type,title,target_widgets", TARGETS, ids=[node[0] for node in TARGETS])
@pytest.mark.parametrize("source_first", [False, True], ids=["consumer-first", "source-first"])
@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_dynamic_output_type_survives_native_loading(ui, node_type, title, target_widgets,
                                                   source_first, restore_by_name):
    browser, artifacts = ui
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
    elif node_type == "ImpactControlBridge":
        browser.choose(node_type, "behavior", "Mute")
        browser.page.mouse.click(*browser.point(node_type, widget="mode"))

    node_ids = [target, inverse, source]
    expected = [{"id": node_id, "type": kind,
                 "widgets": [{"name": name, "value": value} for name, value in values]}
                for node_id, kind, values in [
                    (target, node_type, target_widgets),
                    (inverse, "ImpactInversedSwitch", [("select", 2), ("sel_mode", False)]),
                    (source, "ImpactInt", [("value", 42)]),
                ]]
    before = widgets(browser, node_ids)
    # Anchor the round trip to the intended non-default values and widget order.
    assert json.dumps(before, sort_keys=True) == json.dumps(expected, sort_keys=True)
    original_path = artifacts / "original.json"
    exported_path = artifacts / "exported.json"
    export_workflow(browser, original_path)
    original = json.loads(original_path.read_text())
    order = [str(node["id"]) for node in original["nodes"]]
    assert (order.index(inverse) < order.index(target)) == source_first
    connection = next(link for link in original["links"] if str(link[3]) == target)
    assert str(connection[1]) == inverse and connection[2] == 1 and connection[5] == "INT"

    observations = [before]
    for path in (original_path, exported_path):
        browser.new_workflow()
        browser.load_workflow(path, len(original["nodes"]))
        after = widgets(browser, node_ids)
        observations.append(after)
        (artifacts / "widgets.json").write_text(json.dumps(observations, indent=2) + "\n")
        assert json.dumps(after, sort_keys=True) == json.dumps(expected, sort_keys=True)
        types = browser.app.evaluate("""(app, id) => {
            const node = app.graph.getNodeById(id);
            return {inputs: node.inputs.map(slot => slot.type ?? null),
                    outputs: node.outputs.map(slot => slot.type ?? null)};
        }""", target)
        input_count = 2 if node_type == "ImpactCompare" or node_type.startswith("ImpactConditionalBranch") else 1
        # The frontend may append widget-backed inputs after the data inputs.
        assert types["inputs"][:input_count] == ["INT"] * input_count
        assert types["outputs"] == ["BOOLEAN" if node_type == "ImpactCompare" else "INT"]
        if path == original_path:
            export_workflow(browser, exported_path)
    exported = json.loads(exported_path.read_text())
    for workflow in (original, exported):
        saved = {str(node["id"]): node for node in workflow["nodes"]}
        for node in expected:
            values = {widget["name"]: widget["value"] for widget in node["widgets"]}
            assert json.dumps(saved[node["id"]]["widgets_values"]) == json.dumps(list(values.values()))
            assert json.dumps(saved[node["id"]]["widgets_values_named"], sort_keys=True) == json.dumps(values, sort_keys=True)
    assert original["links"] == exported["links"]


@pytest.mark.parametrize("node_type,title,target_widgets", TARGETS, ids=[node[0] for node in TARGETS])
@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_restored_dynamic_output_consumer_executes(ui, node_type, title, target_widgets, restore_by_name):
    browser, artifacts = ui
    browser.page.set_viewport_size({"width": 1900, "height": 1100})
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    browser.new_workflow()
    target = browser.add_node(title, node_type, 1000, 180)
    inverse = browser.add_node("Inversed Switch (Any)", "ImpactInversedSwitch", 620, 350)
    source = browser.add_node("ImpactInt", "ImpactInt", 180, 350)
    # Keep output1 connected so the consumer really uses dynamic output2 on reload.
    browser.add_node("ImpactConvertDataType", "ImpactConvertDataType", 1000, 650)
    preview = browser.add_node("Preview as Text", "PreviewAny", 1450, 250)
    browser.set_number("ImpactInt", "value", 42)
    if node_type != "ImpactControlBridge":
        browser.add_node("Int", "PrimitiveInt", 180, 650)
        browser.set_number("PrimitiveInt", "value", 7)
    if node_type == "ImpactSelectNthItemOfAnyList":
        browser.add_node("Make List (Any)", "ImpactMakeAnyList", 620, 650)
        browser.connect("ImpactInt", "ImpactMakeAnyList")
        browser.connect("PrimitiveInt", "ImpactMakeAnyList", 1)
        browser.connect("ImpactMakeAnyList", "ImpactInversedSwitch")
        browser.set_number(node_type, "index", -1)
    else:
        browser.connect("ImpactInt", "ImpactInversedSwitch")
    browser.connect("ImpactInversedSwitch", "ImpactConvertDataType")
    browser.connect("ImpactInversedSwitch", node_type, source_slot=1)
    browser.connect(node_type, "PreviewAny")
    browser.set_number("ImpactInversedSwitch", "select", 2)
    if node_type == "ImpactCompare":
        browser.connect("PrimitiveInt", node_type, 1)
        browser.choose(node_type, "cmp", "a > b")
    elif node_type.startswith("ImpactConditionalBranch"):
        browser.connect("PrimitiveInt", node_type, 1)
        browser.page.mouse.click(*browser.point(node_type, widget="cond"))
    elif node_type == "ImpactControlBridge":
        target_widgets = [("mode", True), ("behavior", "Stop")]

    expected = [{"id": node_id, "type": kind,
                 "widgets": [{"name": name, "value": value} for name, value in values]}
                for node_id, kind, values in [
                    (target, node_type, target_widgets),
                    (inverse, "ImpactInversedSwitch", [("select", 2), ("sel_mode", False)]),
                    (source, "ImpactInt", [("value", 42)]),
                ]]
    node_ids = [target, inverse, source]
    assert json.dumps(widgets(browser, node_ids), sort_keys=True) == json.dumps(expected, sort_keys=True)
    original_path, exported_path = artifacts / "original.json", artifacts / "exported.json"
    export_workflow(browser, original_path)
    original = json.loads(original_path.read_text())
    order = [str(node["id"]) for node in original["nodes"]]
    assert order.index(target) < order.index(inverse)
    connection = next(link for link in original["links"] if str(link[3]) == target and link[4] == 0)
    assert connection[1:3] == [int(inverse), 1] and connection[5] == "INT"

    def run_and_check(label, text):
        entry = browser.run()
        (artifacts / f"{label}.json").write_text(json.dumps(entry, indent=2) + "\n")
        actual = {node: output["text"] for node, output in entry["outputs"].items() if "text" in output}
        assert actual == ({} if text is None else {preview: [text]})
        if text is not None:
            expect(browser.page.get_by_role("textbox", name="preview_text", exact=True)).to_have_value(text)

    for path in (original_path, exported_path):
        browser.new_workflow()
        browser.load_workflow(path, len(original["nodes"]))
        assert json.dumps(widgets(browser, node_ids), sort_keys=True) == json.dumps(expected, sort_keys=True)
        if path == original_path:
            export_workflow(browser, exported_path)
        run_and_check(path.stem + "-initial", "True" if node_type == "ImpactCompare" else
                      "7" if node_type == "ImpactSelectNthItemOfAnyList" else "42")
        show_node(browser, target)
        if node_type == "ImpactCompare":
            browser.choose(node_type, "cmp", "a < b")
            changed = "False"
        elif node_type.startswith("ImpactConditionalBranch"):
            browser.page.mouse.click(*browser.point(node_type, widget="cond"))
            changed = "7"
        elif node_type == "ImpactSelectNthItemOfAnyList":
            browser.set_number(node_type, "index", 0)
            changed = "42"
        else:
            browser.page.mouse.click(*browser.point(node_type, widget="mode"))
            changed = None  # Stop must block the downstream preview, not pass 42 through.
        run_and_check(path.stem + "-changed", changed)
        if node_type == "ImpactControlBridge":
            browser.page.mouse.click(*browser.point(node_type, widget="mode"))
            run_and_check(path.stem + "-resumed", "42")
            for behavior, mode in [("Mute", 2), ("Bypass", 4)]:
                browser.choose(node_type, "behavior", behavior)
                browser.page.mouse.click(*browser.point(node_type, widget="mode"))
                bridge_transition(browser, preview, mode, artifacts, path.stem + "-" + behavior)
                browser.page.mouse.click(*browser.point(node_type, widget="mode"))
                bridge_transition(browser, preview, 0, artifacts, path.stem + "-" + behavior + "-active")

    exported = json.loads(exported_path.read_text())
    assert original["links"] == exported["links"]
    for key in ("widgets_values", "widgets_values_named"):
        assert json.dumps({n["id"]: n.get(key) for n in original["nodes"]}, sort_keys=True) == json.dumps(
            {n["id"]: n.get(key) for n in exported["nodes"]}, sort_keys=True)
