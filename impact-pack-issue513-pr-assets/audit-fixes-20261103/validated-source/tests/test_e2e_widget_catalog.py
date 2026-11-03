"""Native workflow/widget compatibility for every registered Impact Pack node.

Run this file directly to extend the fixtures through Chrome's UI. Pytest only
reads the checked-in fixtures; it never updates the expected widget values.
"""
from argparse import Namespace
import ast
from collections import Counter
import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Error as PlaywrightError, expect

from e2e_ui import browser_session, options


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tests/workflows/all-impact-nodes.json"
WIDGETS = ROOT / "tests/workflows/all-impact-nodes.widgets.json"


def registered_types():
    module = ast.parse((ROOT / "__init__.py").read_text())
    mapping = next(statement.value for statement in module.body
                   if isinstance(statement, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                       for target in statement.targets))
    return [ast.literal_eval(key) for key in mapping.keys]


def node_definitions(ui):
    response = ui.page.request.get(f"{ui.base_url}/object_info")
    assert response.ok
    definitions = response.json()
    module = definitions["ImpactSwitch"]["python_module"]
    impact = {name: info for name, info in definitions.items() if info["python_module"] == module}
    assert set(registered_types()) == impact.keys(), "Server and source node registrations differ"
    return impact


def setting(ui, search, setting_id, enabled):
    page = ui.page
    page.get_by_role("button", name="Settings (Ctrl + ,)", exact=True).click()
    page.get_by_placeholder("Search Settings...", exact=True).fill(search)
    switch = page.locator(f'[id="{setting_id}"]')
    expect(switch).to_be_visible()
    if switch.get_attribute("aria-checked") != str(enabled).lower():
        switch.click()
    expect(switch).to_have_attribute("aria-checked", str(enabled).lower())
    page.keyboard.press("Escape")


def full_zoom(ui):
    page = ui.page
    page.get_by_test_id("zoom-controls-button").click()
    zoom = page.get_by_test_id("zoom-percentage-input").locator("input")
    # The number field can restore its previous value after a synthetic fill.
    # Type through its keyboard handlers before committing with Enter.
    zoom.press("ControlOrMeta+a")
    zoom.press_sequentially("100", delay=100)
    zoom.press("Enter")
    page.get_by_test_id("zoom-controls-button").click()
    page.wait_for_function("app => app.canvas.ds.scale === 1", arg=ui.app)


def rendered(ui):
    ui.page.evaluate("""async () => {
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
    }""")


def pan(ui, dx, dy):
    # Middle-drag also works over nodes and cannot type into a focused widget.
    page = ui.page
    while abs(dx) > 0.5 or abs(dy) > 0.5:
        step_x, step_y = max(-600, min(600, dx)), max(-400, min(400, dy))
        page.mouse.move(850, 500)
        page.mouse.down(button="middle")
        page.mouse.move(850 + step_x, 500 + step_y, steps=4)
        page.mouse.up(button="middle")
        rendered(ui)
        dx -= step_x
        dy -= step_y


def show_node(ui, node_id):
    x, y = ui.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        return app.canvasPosToClientPos(node.pos);
    }""", node_id)
    pan(ui, 250 - x, 130 - y)
    ui.page.wait_for_function("""({app, id}) =>
        app.canvas.visible_nodes.includes(app.graph.getNodeById(id))
    """, arg={"app": ui.app, "id": node_id})
    rendered(ui)


def widget_state(ui, node_id):
    return ui.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        const registered = LiteGraph.registered_node_types[node.type];
        if (!registered || !(node instanceof registered)) throw new Error(`Unregistered node: ${node.type}`);
        if (!app.canvas.visible_nodes.includes(node)) throw new Error(`Node ${id} was not rendered`);
        return {id: String(node.id), type: node.type, widgets: (node.widgets ?? []).map(widget => {
            if (!Number.isFinite(widget.last_y)) throw new Error(`${node.type}.${widget.name} was not drawn`);
            if (widget.inputEl && widget.inputEl.value !== String(widget.value))
                throw new Error(`${node.type}.${widget.name}: DOM and widget values differ`);
            return {name: widget.name, value: widget.value ?? null};
        })};
    }""", node_id)


def export_workflow(ui, path):
    page = ui.page
    page.locator('button[aria-label$="workflow actions" i]').click()
    page.get_by_text("Export", exact=True).filter(visible=True).click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("textbox").fill(path.stem)
    with page.expect_download() as download:
        dialog.get_by_role("button", name="Confirm", exact=True).click()
    download.value.save_as(path)


def assert_catalog(ui, expected):
    actual_nodes = ui.app.evaluate("""app => app.graph._nodes.map(node =>
        ({id: String(node.id), type: node.type}))""")
    assert actual_nodes == [{"id": node["id"], "type": node["type"]} for node in expected]
    full_zoom(ui)
    actual = []
    for node in expected:
        show_node(ui, node["id"])
        state = widget_state(ui, node["id"])
        # Python considers True == 1 and False == 0; JSON preserves that distinction.
        assert json.dumps(state, sort_keys=True) == json.dumps(node, sort_keys=True), (
            f"Widget mismatch for {node['type']}: {state['widgets']} != {node['widgets']}")
        actual.append(state)
    return actual


def check_catalog_load(ui, workflow, expected, restore_by_name):
    response = ui.page.request.get(f"{ui.base_url}/impact/wildcards/list/loaded")
    assert response.ok, f"Cannot check wildcard cache mode: HTTP {response.status}"
    assert response.json()["on_demand_mode"] is False, (
        "Catalog requires Full Cache mode. Set wildcard_cache_limit_mb above the total "
        "wildcard file size in the test copy's impact-pack.ini and restart the server. "
        "Keep the trusted workflow and snapshot unchanged.")
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    ui.new_workflow()
    # has_errors also flags unavailable models. This unconnected catalog
    # checks loading/rendering without requiring inference assets.
    ui.load_workflow(workflow, len(expected), check_node_errors=False)
    return assert_catalog(ui, expected)


def record_catalog(args):
    # Read the trusted baseline before opening ComfyUI. Missing fixtures must
    # be restored, never recreated from a potentially broken runtime.
    baseline = json.loads(WIDGETS.read_text())
    assert WORKFLOW.is_file(), f"Restore the trusted workflow: {WORKFLOW}"
    nodes = baseline["nodes"].copy()
    existing_types = {node["type"] for node in nodes}
    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        definitions = node_definitions(ui)
        types = registered_types()
        assert existing_types <= definitions.keys(), "Existing catalog nodes are no longer registered"
        for restore_by_name in (False, True):
            check_catalog_load(ui, WORKFLOW, nodes, restore_by_name)
        print("Existing catalog matches the trusted baseline in both restoration modes", flush=True)
        additions = [node_type for node_type in types if node_type not in existing_types]
        if not additions:
            print("No new node types; fixture files were not changed", flush=True)
            return
        setting(ui, "deprecated", "Comfy.Node.ShowDeprecated", True)
        for index, node_type in enumerate(additions, start=len(nodes)):
            # Continue the existing catalog's 14-column grid.
            row, column = divmod(index, 14)
            pan(ui, 0 if column == 0 else (800 if row % 2 else -800), -1500 if column == 0 else 0)
            node_id = ui.add_node(definitions[node_type]["display_name"], node_type, 450, 180)
            rendered(ui)
            nodes.append(widget_state(ui, node_id))
            print(f"Created {index + 1}/{len(types)}: {node_type}", flush=True)
        assert Counter(node["type"] for node in nodes) == Counter(types)
        candidate = args.artifacts / "candidate-workflow.json"
        export_workflow(ui, candidate)
        (args.artifacts / "created-widgets.json").write_text(json.dumps(nodes, indent=2) + "\n")
        for restore_by_name in (False, True):
            check_catalog_load(ui, candidate, nodes, restore_by_name)
        data = {"frontend_version": json.loads(candidate.read_text())["extra"]["frontendVersion"],
                "created_via": "Chrome node search and native workflow Export",
                "nodes": nodes}
    # Browser error checks and artifact saving must also succeed before publication.
    WORKFLOW.write_bytes(candidate.read_bytes())
    WIDGETS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def test_catalog_contains_every_registered_node():
    workflow = json.loads(WORKFLOW.read_text())
    expected = json.loads(WIDGETS.read_text())["nodes"]
    types = Counter(registered_types())
    assert all(count == 1 for count in types.values())
    assert Counter(node["type"] for node in workflow["nodes"]) == types
    assert Counter(node["type"] for node in expected) == types
    assert [(str(node["id"]), node["type"]) for node in workflow["nodes"]] == [
        (node["id"], node["type"]) for node in expected]
    for node in expected:
        for widget in node["widgets"]:
            assert set(widget) == {"name", "value"}
            assert isinstance(widget["name"], str)
    label_hook = next(node for node in workflow["nodes"] if node["type"] == "SEGSLabelFilterDetailerHookProvider")
    assert label_hook["inputs"] == [], "Recreate the label-filter hook to remove its obsolete SEGS input"


@pytest.fixture
def browser_args(tmp_path, request):
    artifacts = Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name
    return Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30, artifacts=artifacts)


@pytest.fixture
def ui(browser_args):
    with browser_session(browser_args) as browser:
        browser.page.keyboard.press("Escape")
        yield browser, browser_args


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_all_widgets_survive_native_workflow_load(ui, restore_by_name):
    browser, args = ui
    expected = json.loads(WIDGETS.read_text())["nodes"]
    assert {node["type"] for node in expected} == node_definitions(browser).keys()
    actual = check_catalog_load(browser, WORKFLOW, expected, restore_by_name)
    (args.artifacts / "loaded-widgets.json").write_text(json.dumps(actual, indent=2) + "\n")


@pytest.mark.parametrize("node_type", ["SEGSLabelFilterDetailerHookProvider", "ImpactSEGSLabelFilter"])
def test_label_preset_initial_value_and_selection(ui, node_type):
    browser, args = ui
    definitions = node_definitions(browser)
    browser.new_workflow()
    full_zoom(browser)
    node_id = browser.add_node(definitions[node_type]["display_name"], node_type, 450, 180)
    rendered(browser)
    assert widget_state(browser, node_id)["widgets"] == [
        {"name": "preset", "value": "all"}, {"name": "labels", "value": ""}]
    browser.choose(node_type, "preset", "person")
    assert widget_state(browser, node_id)["widgets"] == [
        {"name": "preset", "value": "person"}, {"name": "labels", "value": "person"}]
    before = widget_state(browser, node_id)
    path = args.artifacts / "label-preset.json"
    export_workflow(browser, path)
    browser.new_workflow()
    browser.load_workflow(path, 1)
    assert_catalog(browser, [before])


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_label_filter_hook_executes_after_native_load(ui, restore_by_name):
    browser, args = ui
    node_type = "SEGSLabelFilterDetailerHookProvider"
    expected_widgets = [{"name": "preset", "value": "person"},
                        {"name": "labels", "value": "person, dog"}]
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    browser.new_workflow()
    full_zoom(browser)
    definitions = node_definitions(browser)
    node_id = browser.add_node(definitions[node_type]["display_name"], node_type, 250, 180)
    browser.choose(node_type, "preset", "person")
    browser.page.get_by_placeholder(
        "List the types of segments to be allowed, separated by commas", exact=True
    ).fill("person, dog")
    assert "segs" not in browser.app.evaluate("""(app, id) =>
        app.graph.getNodeById(id).inputs.map(input => input.name)
    """, node_id)
    browser.add_node("Preview as Text", "PreviewAny", 850, 180)
    browser.connect(node_type, "PreviewAny")

    path = args.artifacts / "label-filter-hook.json"
    for reloaded in (False, True):
        if reloaded:
            browser.new_workflow()
            browser.load_workflow(path, 2)
        full_zoom(browser)
        show_node(browser, node_id)
        assert widget_state(browser, node_id)["widgets"] == expected_widgets
        entry = browser.run()
        text = next(output["text"] for output in entry["outputs"].values() if "text" in output)
        assert len(text) == 1 and "SEGSLabelFilterDetailerHook object" in text[0]
        if not reloaded:
            export_workflow(browser, path)
            saved = next(node for node in json.loads(path.read_text())["nodes"] if node["type"] == node_type)
            assert saved["widgets_values"] == ["person", "person, dog"]
            assert saved["widgets_values_named"] == {"preset": "person", "labels": "person, dog"}


@pytest.mark.parametrize("mutation", [
    "this.widgets[0].name = 'guide_size_broken';",
    "[this.widgets[0], this.widgets[2]] = [this.widgets[2], this.widgets[0]];",
    "this.widgets[0].value = 512;",
    "this.widgets[1].value = 1;",
], ids=["name", "order", "value", "value_type"])
def test_recorder_preserves_trusted_fixtures_on_failure(browser_args, tmp_path, monkeypatch, mutation):
    workflow_bytes, snapshot_bytes = WORKFLOW.read_bytes(), WIDGETS.read_bytes()
    workflow, snapshot = tmp_path / WORKFLOW.name, tmp_path / WIDGETS.name
    workflow.write_bytes(workflow_bytes)
    snapshot.write_bytes(snapshot_bytes)
    monkeypatch.setitem(globals(), "WORKFLOW", workflow)
    monkeypatch.setitem(globals(), "WIDGETS", snapshot)
    new_page = BrowserContext.new_page

    def broken_frontend(context):
        page = new_page(context)

        def serve(route):
            response = route.fetch()
            # Simulate a runtime regression, without editing either fixture.
            script = """
app.registerExtension({
    name: 'impact.catalog.corruptionTest',
    nodeCreated(node) {
        if (node.comfyClass !== 'FaceDetailer') return;
        const configure = node.onConfigure;
        node.onConfigure = function(...args) {
            const result = configure?.apply(this, args);
            MUTATION
            return result;
        };
    }
});
""".replace("MUTATION", mutation)
            route.fulfill(response=response, body=response.text() + script)

        page.route("**/extensions/*/impact-pack.js", serve)
        return page

    monkeypatch.setattr(BrowserContext, "new_page", broken_frontend)
    with pytest.raises(AssertionError, match="Widget mismatch for FaceDetailer"):
        record_catalog(browser_args)
    assert workflow.read_bytes() == workflow_bytes
    assert snapshot.read_bytes() == snapshot_bytes
    assert not (browser_args.artifacts / "candidate-workflow.json").exists()


@pytest.mark.parametrize("failure", [None, "pageerror", "trace"], ids=["success", "pageerror", "trace"])
def test_recorder_publishes_only_after_successful_session(browser_args, tmp_path, monkeypatch, failure):
    baseline = json.loads(WIDGETS.read_text())
    expected = next(node for node in baseline["nodes"] if node["type"] == "ImpactInt").copy()
    workflow, snapshot = tmp_path / WORKFLOW.name, tmp_path / WIDGETS.name
    setup_args = Namespace(**(vars(browser_args) | {"artifacts": browser_args.artifacts / "setup"}))
    with browser_session(setup_args) as browser:
        browser.page.keyboard.press("Escape")
        definitions = node_definitions(browser)
        browser.new_workflow()
        full_zoom(browser)
        expected["id"] = browser.add_node(definitions["ImpactInt"]["display_name"], "ImpactInt", 450, 180)
        browser.set_number("ImpactInt", "value", expected["widgets"][0]["value"])
        assert_catalog(browser, [expected])
        export_workflow(browser, workflow)
        tracing_type = type(browser.page.context.tracing)
    snapshot.write_text(json.dumps(baseline | {"nodes": [expected]}))
    workflow_bytes, snapshot_bytes = workflow.read_bytes(), snapshot.read_bytes()
    monkeypatch.setitem(globals(), "WORKFLOW", workflow)
    monkeypatch.setitem(globals(), "WIDGETS", snapshot)
    # Exercise a real UI addition with a small catalog; full inventory coverage
    # and both restoration modes are checked separately above.
    definitions = {name: definitions[name] for name in ("ImpactInt", "ImpactBoolean")}
    monkeypatch.setitem(globals(), "registered_types", lambda: list(definitions))
    monkeypatch.setitem(globals(), "node_definitions", lambda ui: definitions)
    if failure == "pageerror":
        export = export_workflow

        def export_with_error(ui, path):
            export(ui, path)
            with ui.page.expect_event("pageerror"):
                ui.page.evaluate("() => { setTimeout(() => { throw new Error('catalog async error'); }, 0); }")

        monkeypatch.setitem(globals(), "export_workflow", export_with_error)
        error, message = AssertionError, "catalog async error"
    elif failure == "trace":
        stop = tracing_type.stop

        def fail_trace(tracing, **kwargs):
            stop(tracing, **kwargs)
            raise PlaywrightError("catalog trace error")

        monkeypatch.setattr(tracing_type, "stop", fail_trace)
        error, message = PlaywrightError, "catalog trace error"
    if failure:
        with pytest.raises(error, match=message):
            record_catalog(browser_args)
        assert workflow.read_bytes() == workflow_bytes
        assert snapshot.read_bytes() == snapshot_bytes
    else:
        record_catalog(browser_args)
        assert json.loads(snapshot.read_text())["nodes"] == [expected, {
            "id": "2", "type": "ImpactBoolean", "widgets": [{"name": "value", "value": False}]}]
        assert workflow.read_bytes() == (browser_args.artifacts / "candidate-workflow.json").read_bytes()
    assert len(json.loads((browser_args.artifacts / "candidate-workflow.json").read_text())["nodes"]) == 2


if __name__ == "__main__":
    args = options("Verify the trusted catalog, then add new nodes through Chrome UI and native Export")
    if args.artifacts is None:
        raise SystemExit("--artifacts is required to retain the recording trace and candidate workflow")
    record_catalog(args)
