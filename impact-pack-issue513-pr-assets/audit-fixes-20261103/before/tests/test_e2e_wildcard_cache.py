"""Real queue/cache behavior and native wildcard widget restoration in Chrome.

Use an isolated ComfyUI server. Encode cases use an installed checkpoint,
defaulting to v1-5-pruned-emaonly-fp16.safetensors (no downloads).
"""
from argparse import Namespace
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from playwright.sync_api import expect

from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, full_zoom, rendered, setting, show_node, widget_state


NODE_TYPES = ["ImpactWildcardProcessor", "ImpactWildcardEncode"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ui(tmp_path, request):
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=120,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name)
    with browser_session(args) as browser:
        browser.artifacts = args.artifacts
        browser.page.keyboard.press("Escape")
        browser.new_workflow()
        full_zoom(browser)
        yield browser


def graph(ui, node_type):
    node_id = ui.add_node(node_type, node_type, 550, 170)
    preview_id = ui.add_node("Preview as Text", "PreviewAny", 1150, 450)
    ui.connect(node_type, "PreviewAny", source_slot=3 if node_type == "ImpactWildcardEncode" else 0)
    if node_type == "ImpactWildcardEncode":
        ui.add_node("Load Checkpoint", "CheckpointLoaderSimple", 100, 500)
        checkpoint = os.environ.get("IMPACT_E2E_CHECKPOINT", "v1-5-pruned-emaonly-fp16.safetensors")
        ui.choose("CheckpointLoaderSimple", "ckpt_name", checkpoint)
        ui.connect("CheckpointLoaderSimple", node_type)
        ui.connect("CheckpointLoaderSimple", node_type, 1, source_slot=1)
    show_node(ui, node_id)
    ui.choose(node_type, "control_after_generate", "fixed")
    return node_id, preview_id


def cached(entry):
    return {str(node) for event, data in entry["status"]["messages"]
            if event == "execution_cached" for node in data["nodes"]}


def run(ui, preview_id, expected, label):
    entry = ui.run()
    (ui.artifacts / f"{label}.json").write_text(json.dumps(entry, indent=2) + "\n")
    assert entry["outputs"][preview_id]["text"][0] == expected
    expect(ui.page.get_by_role("textbox", name="preview_text", exact=True)).to_have_value(expected)
    return entry


def run_submitted(ui, preview_id, expected, label):
    # History contains the backend-normalized seed. Observe the real POST to
    # distinguish before/after frontend control timing without copying core RNG.
    with ui.page.expect_request(lambda req: req.method == "POST" and urlsplit(req.url).path in ("/prompt", "/api/prompt")) as queued:
        entry = run(ui, preview_id, expected, label)
    request = queued.value.post_data_json
    (ui.artifacts / f"{label}-request.json").write_text(json.dumps(request, indent=2) + "\n")
    return entry, request["prompt"]


def native_roundtrip(ui, node_id, *, expected_widget_names=None):
    """Compare rendered widget names/order/values and the native serialization."""
    show_node(ui, node_id)
    snapshot = widget_state if expected_widget_names is None else promoted_widget_state
    before = snapshot(ui, node_id)
    if expected_widget_names is None:
        trusted = json.loads((ROOT / "tests/workflows/all-impact-nodes.widgets.json").read_text())
        reference = next(node for node in trusted["nodes"] if node["type"] == before["type"])
        expected_widget_names = [w["name"] for w in reference["widgets"]]
    assert [w["name"] for w in before["widgets"]] == expected_widget_names
    workflow = ui.artifacts / "saved.json"
    export_workflow(ui, workflow)
    saved = json.loads(workflow.read_text())
    ui.new_workflow()
    ui.load_workflow(workflow, len(saved["nodes"]))
    full_zoom(ui)
    show_node(ui, node_id)
    assert snapshot(ui, node_id) == before
    roundtrip = ui.artifacts / "roundtrip.json"
    export_workflow(ui, roundtrip)
    restored = json.loads(roundtrip.read_text())
    assert {str(n["id"]): n.get("widgets_values", []) for n in restored["nodes"]} == {
        str(n["id"]): n.get("widgets_values", []) for n in saved["nodes"]}
    # Native loading can reorder link records; their IDs and endpoints must
    # remain identical. Widget arrays above deliberately retain their order.
    assert sorted(restored["links"]) == sorted(saved["links"])
    # Subgraph link records can reorder too; compare complete records by ID.
    # Keep every node/widget array and all other definition fields ordered.
    def ordered_links(definitions):
        if definitions is None:
            return None
        return {**definitions, "subgraphs": [
            {**subgraph, "links": sorted(subgraph["links"], key=lambda link: link["id"])}
            for subgraph in definitions["subgraphs"]]}
    assert ordered_links(restored.get("definitions")) == ordered_links(saved.get("definitions"))
    (ui.artifacts / "widget-state.json").write_text(json.dumps(before, indent=2) + "\n")
    return before


def promoted_widget_state(ui, node_id):
    # Host widget projections can replace their transient last_y after a queue.
    # Check rendered visibility and stable name/value pairs, not that cache field.
    return ui.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        if (!app.canvas.visible_nodes.includes(node)) throw new Error('Host was not rendered');
        return {id: String(node.id), type: node.type,
            widgets: node.widgets.map(w => ({name: w.name, value: w.value ?? null}))};
    }""", node_id)


def input_slot(ui, node_type, name):
    return ui.app.evaluate("""(app, {type, name}) =>
        app.graph._nodes.find(n => n.type === type).inputs.findIndex(i => i.name === name)
    """, {"type": node_type, "name": name})


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_linked_inputs_and_subgraph_native_queue(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    source_id = ui.add_node("Concatenate Text", "StringConcatenate", 100, 650)
    show_node(ui, source_id)
    # Neither fragment contains a complete choice, so scanning upstream input
    # strings cannot determine the seed dependence of their combined output.
    ui.page.get_by_placeholder("string_a", exact=True).fill("{red|")
    ui.page.get_by_placeholder("string_b", exact=True).fill("green|blue}")
    show_node(ui, node_id)
    ui.connect("StringConcatenate", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    for seed, expected in ((2, "red"), (5, "blue")):
        ui.set_number(node_type, "seed", seed)
        entry = run(ui, preview_id, expected, f"linked-text-{seed}")
        assert entry["prompt"][2][node_id]["inputs"]["populated_text"] == [source_id, 0]
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == seed

    seed_id = ui.add_node("ImpactInt", "ImpactInt", 850, 180)
    show_node(ui, seed_id)
    ui.set_number("ImpactInt", "value", 2)
    show_node(ui, node_id)
    ui.connect("ImpactInt", node_type, target_slot=input_slot(ui, node_type, "seed"))
    entry = run(ui, preview_id, "red", "linked-seed")
    assert entry["prompt"][2][node_id]["inputs"]["seed"] == [seed_id, 0]

    # Convert the real upstream node using the frontend menu. No graph mutation
    # or hand-written workflow JSON is used to construct the subgraph.
    show_node(ui, source_id)
    title = ui.page.evaluate("""({app, id}) => {
        const n = app.graph.getNodeById(id);
        return app.canvasPosToClientPos([n.pos[0] + 40, n.pos[1] - 15]);
    }""", {"app": ui.app, "id": source_id})
    ui.page.mouse.click(*title)
    ui.page.mouse.click(*title, button="right")
    ui.page.get_by_role("menuitem", name="Convert to Subgraph", exact=True).click()
    ui.page.wait_for_function("""({app, id}) => !app.graph.getNodeById(id)
    """, arg={"app": ui.app, "id": source_id})
    native_roundtrip(ui, node_id)
    saved = json.loads((ui.artifacts / "saved.json").read_text())
    assert any(n["type"] == "StringConcatenate"
               for subgraph in saved["definitions"]["subgraphs"] for n in subgraph["nodes"])
    for seed, expected in ((5, "blue"), (2, "red")):
        show_node(ui, seed_id)
        ui.set_number("ImpactInt", "value", seed)
        entry = run(ui, preview_id, expected, f"restored-subgraph-{seed}")
        inputs = entry["prompt"][2][node_id]["inputs"]
        assert inputs["seed"] == [seed_id, 0]
        assert isinstance(inputs["populated_text"], list)
        upstream_id = inputs["populated_text"][0]
        assert entry["prompt"][2][upstream_id]["class_type"] == "StringConcatenate"


@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_populate_reuses_cache_only_for_same_expansion(ui, node_type):
    node_id, preview_id = graph(ui, node_type)
    prefix = "populate cache "
    # The source participates in the cache key. A unique comment isolates this
    # case even on a warm LRU server, while keeping expected output deterministic.
    source = f"# cache case {uuid4().hex}\n" + prefix + "{red|green|blue}"
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(source)
    # Trusted pre-change selections: 0 and 1 -> green; 2 -> red.
    for seed, color in ((0, "green"), (1, "green"), (2, "red")):
        ui.set_number(node_type, "seed", seed)
        entry = run(ui, preview_id, prefix + color, f"populate-cache-{seed}")
        expect(ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True)).to_have_value(
            prefix + color)
        if seed == 1:
            assert {node_id, preview_id} <= cached(entry)
        else:
            assert {node_id, preview_id}.isdisjoint(cached(entry))
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == 0
        assert next(w["value"] for w in widget_state(ui, node_id)["widgets"] if w["name"] == "seed") == seed
        saved_node = next(n for n in entry["prompt"][3]["extra_pnginfo"]["workflow"]["nodes"]
                          if str(n["id"]) == node_id)
        assert saved_node["widgets_values"][:3] == [source, prefix + color, "reproduce"]
        seed_index = 3 if node_type == "ImpactWildcardProcessor" else 5
        assert saved_node["widgets_values"][seed_index] == seed


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_reproduce_native_workflow_then_populate(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id = graph(ui, node_type)
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill("{red|green|blue}")
    # Set a known saved expansion through the UI so a population bug cannot
    # prevent this test from reaching the reproduce/load behavior it protects.
    ui.choose(node_type, "mode", "fixed")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("red")
    ui.choose(node_type, "mode", "reproduce")
    ui.set_number(node_type, "seed", 5)
    before = native_roundtrip(ui, node_id)
    values = {w["name"]: w["value"] for w in before["widgets"]}
    assert (values["wildcard_text"], values["populated_text"], values["mode"], values["seed"]) == (
        "{red|green|blue}", "red", "reproduce", 5)
    run(ui, preview_id, "red", "restored-reproduce")
    ui.page.wait_for_function("""({app, id}) =>
        app.graph.getNodeById(id).widgets.find(w => w.name === 'mode').value === 'populate'
    """, arg={"app": ui.app, "id": node_id})
    run(ui, preview_id, "blue", "next-populate")
    assert widget_state(ui, node_id)["widgets"] == [
        dict(w, value="populate") if w["name"] == "mode" else
        dict(w, value="blue") if w["name"] == "populated_text" else w for w in before["widgets"]
    ]


def control_timing(ui, value):
    ui.page.get_by_role("button", name="Settings (Ctrl + ,)", exact=True).click()
    ui.page.get_by_placeholder("Search Settings...", exact=True).fill("Widget control mode")
    ui.page.locator('[id="Comfy.WidgetControlMode"]').click()
    ui.page.get_by_role("option", name=value, exact=True).click()
    ui.page.get_by_role("dialog").press("Escape")
    expect(ui.page.get_by_role("dialog")).to_be_hidden()


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("text_case, control, timing", [
    ("plain", "increment", "after"),
    ("comment", "increment", "after"),
    ("empty", "increment", "after"),
    ("choice", "increment", "after"),
    ("choice", "decrement", "after"),
    ("choice", "randomize", "after"),
    ("linked-plain", "increment", "after"),
    ("linked-choice", "increment", "after"),
    ("choice", "increment", "before"),
    ("choice", "decrement", "before"),
    ("choice", "randomize", "before"),
    ("linked-choice", "increment", "before"),
], ids=lambda value: value)
def test_fixed_mode_suppresses_automatic_seed_changes(ui, node_type, text_case, control, timing):
    control_timing(ui, timing)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.set_number(node_type, "seed", 5)
    ui.choose(node_type, "control_after_generate", control)
    source = f"unused {uuid4().hex}"
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(source)
    dynamic = "choice" in text_case
    text = "{red|green|blue}" if dynamic else "ordinary prompt"
    if text_case == "comment":
        text = "# {ignored} __ignored__\n" + text
    elif text_case == "empty":
        text = "# {ignored} __ignored__"
    expected = "blue" if dynamic else "" if text_case == "empty" else "ordinary prompt"
    linked = text_case.startswith("linked-")
    if linked:
        source_id = ui.add_node("Concatenate Text", "StringConcatenate", 100, 650)
        show_node(ui, source_id)
        first_part, last_part = ("{red|", "green|blue}") if dynamic else ("ordinary ", "prompt")
        ui.page.get_by_placeholder("string_a", exact=True).fill(first_part)
        ui.page.get_by_placeholder("string_b", exact=True).fill(last_part)
        show_node(ui, node_id)
        ui.connect("StringConcatenate", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    else:
        ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill(text)

    first = run(ui, preview_id, expected, "first")
    second = run(ui, preview_id, expected, "second")
    assert {node_id, preview_id}.isdisjoint(cached(first))
    assert {node_id, preview_id} <= cached(second)
    for entry in (first, second):
        inputs = entry["prompt"][2][node_id]["inputs"]
        assert inputs["populated_text"] == ([source_id, 0] if linked else text)
        assert inputs["seed"] == (0 if text_case == "plain" else 5)
        saved = next(n for n in entry["prompt"][3]["extra_pnginfo"]["workflow"]["nodes"]
                     if str(n["id"]) == node_id)
        assert saved["widgets_values"][3 if node_type == "ImpactWildcardProcessor" else 5] == 5
    values = {w["name"]: w["value"] for w in widget_state(ui, node_id)["widgets"]}
    assert (values["seed"], values["control_after_generate"]) == (5, control)

    if linked:
        show_node(ui, source_id)
        ui.page.get_by_placeholder("string_b", exact=True).fill(last_part + " changed")
        show_node(ui, node_id)
    else:
        # Add text outside the comment in the comment-only case.
        changed_text = text + ("\n" if text_case == "empty" else "") + " changed"
        ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill(changed_text)
    changed = run(ui, preview_id, expected + " changed", "changed")
    assert {node_id, preview_id}.isdisjoint(cached(changed))
    if dynamic:
        ui.set_number(node_type, "seed", 2)
        changed_seed = run(ui, preview_id, "red changed", "manual-seed")
        assert {node_id, preview_id}.isdisjoint(cached(changed_seed))
    if (text_case, control) == ("choice", "increment"):
        ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill("{red|green|blue}")
        ui.choose(node_type, "mode", "populate")
        ui.set_number(node_type, "seed", 5)
        _, resumed = run_submitted(ui, preview_id, "blue" if timing == "after" else "green", "resume-populate")
        assert resumed[node_id]["inputs"]["seed"] == (5 if timing == "after" else 6)
        ui.page.wait_for_function("""({app, id}) =>
            app.graph.getNodeById(id).widgets.find(w => w.name === 'seed').value === 6
        """, arg={"app": ui.app, "id": node_id})
        _, next_populate = run_submitted(ui, preview_id, "green", "next-populate")
        assert next_populate[node_id]["inputs"]["seed"] == (6 if timing == "after" else 7)
        assert next(w["value"] for w in widget_state(ui, node_id)["widgets"] if w["name"] == "seed") == 7


def convert_to_subgraph(ui, node_id):
    show_node(ui, node_id)
    title, body = ui.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        return [app.canvasPosToClientPos([node.pos[0] + 40, node.pos[1] - 15]),
            app.canvasPosToClientPos([node.pos[0] + 2, node.pos[1] + node.size[1] - 5])];
    }""", node_id)
    ui.page.mouse.click(*body)
    ui.page.wait_for_function("({app, id}) => app.canvas.selected_nodes[id] === app.graph.getNodeById(id)", arg={"app": ui.app, "id": node_id})
    ui.page.mouse.click(*title, button="right")
    ui.page.get_by_role("menuitem", name="Convert to Subgraph", exact=True).click()
    ui.page.wait_for_function("({app, id}) => !app.graph.getNodeById(id)", arg={"app": ui.app, "id": node_id})
    host = str(ui.page.wait_for_function("app => app.graph._nodes.find(n => n.isSubgraphNode())?.id", arg=ui.app).json_value())
    rendered(ui)
    ui.page.keyboard.press("Escape")  # Close the new subgraph's inline title editor.
    return host


def enter_subgraphs(ui, count):
    for _ in range(count):
        enter = ui.app.evaluate("""app => {
            const node = app.canvas.graph._nodes.find(n => n.isSubgraphNode());
            return app.canvasPosToClientPos([node.pos[0] + node.size[0] - 15, node.pos[1] - 15]);
        }""")
        previous = ui.app.evaluate_handle("app => app.canvas.graph")
        ui.page.mouse.click(*enter)
        ui.page.wait_for_function("({app, previous}) => app.canvas.graph !== previous", arg={"app": ui.app, "previous": previous})
        previous.dispose()
        ui.page.get_by_role("button", name="Fit View (.)", exact=True).click()
        rendered(ui)


def interior_mode_point(ui, node_type):
    return ui.app.evaluate("""(app, type) => {
        const node = app.canvas.graph._nodes.find(n => n.type === type);
        const widget = node.widgets.find(w => w.name === 'mode');
        return app.canvasPosToClientPos([node.pos[0] + node.size[0] / 2, node.pos[1] + (widget.last_y ?? widget.y) + 10]);
    }""", node_type)


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("restore_by_name, control, nesting, promote_mode", [
    (False, "increment", 1, False),
    (True, "increment", 1, False),
    (False, "decrement", 1, False),
    (False, "randomize", 1, False),
    (False, "increment", 2, False),
    (False, "increment", 1, True),
], ids=["positional", "named", "decrement", "randomize", "nested", "promoted-mode"])
def test_promoted_fixed_seed_preserves_cache_and_widgets(ui, node_type, timing, restore_by_name, control, nesting, promote_mode):
    control_timing(ui, timing)
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.set_number(node_type, "seed", 5)
    ui.choose(node_type, "control_after_generate", control)
    source = f"# promoted {uuid4().hex}\n{{red|green|blue}}"
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(source)
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("{red|green|blue}")
    host = convert_to_subgraph(ui, node_id)
    if nesting == 2:
        host = convert_to_subgraph(ui, host)
    if promote_mode:
        show_node(ui, host)
        enter_subgraphs(ui, nesting)
        ui.page.mouse.click(*interior_mode_point(ui, node_type), button="right")
        ui.page.get_by_role("menuitem", name="Promote Widget: mode", exact=True).click()
        ui.page.get_by_test_id("subgraph-breadcrumb-back").click()
        ui.page.wait_for_function("app => app.canvas.graph === app.rootGraph", arg=ui.app)
    names = ["seed", "mode"] if promote_mode else ["seed"]
    before = native_roundtrip(ui, host, expected_widget_names=names)
    workflow = json.loads((ui.artifacts / "saved.json").read_text())
    interior = next(n for sg in workflow["definitions"]["subgraphs"] for n in sg["nodes"] if n["type"] == node_type)
    controls = interior["widgets_values"]
    seed_index = 3 if node_type == "ImpactWildcardProcessor" else 5
    assert controls[seed_index:seed_index + 2] == [5, control]
    for iteration in range(10):
        show_node(ui, preview_id)
        entry = run(ui, preview_id, "blue", f"fixed-{iteration + 1}")
        target = next(k for k, v in entry["prompt"][2].items() if v["class_type"] == node_type)
        assert entry["prompt"][2][target]["inputs"]["seed"] == 5
        if iteration:
            assert {target, preview_id} <= cached(entry)
        else:
            assert {target, preview_id}.isdisjoint(cached(entry))
        show_node(ui, host)
        assert promoted_widget_state(ui, host) == before
        submitted = entry["prompt"][3]["extra_pnginfo"]["workflow"]
        submitted_inner = next(n for sg in submitted["definitions"]["subgraphs"] for n in sg["nodes"] if n["type"] == node_type)
        assert submitted_inner["widgets_values"][seed_index + 1] == control
        assert submitted_inner["widgets_values_named"]["control_after_generate"] == control
    after = native_roundtrip(ui, host, expected_widget_names=names)
    assert after == before
    restored = json.loads((ui.artifacts / "saved.json").read_text())
    saved_interior = next(n for sg in restored["definitions"]["subgraphs"] for n in sg["nodes"] if n["type"] == node_type)
    assert saved_interior["widgets_values"] == controls

    # A manual host seed edit is still authoritative in fixed mode.
    ui.set_number(before["type"], "seed", 2)
    show_node(ui, preview_id)
    manual = run(ui, preview_id, "red", "manual-host-seed")
    target = next(k for k, v in manual["prompt"][2].items() if v["class_type"] == node_type)
    assert manual["prompt"][2][target]["inputs"]["seed"] == 2
    assert {target, preview_id}.isdisjoint(cached(manual))


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("text", ["{red|\ngreen|blue}", "{red|{green}|blue}"], ids=["multiline", "nested"])
def test_fixed_syntax_preserves_seed(ui, node_type, text):
    _, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill(text)
    for seed, expected in ((2, "red"), (5, "blue")):
        ui.set_number(node_type, "seed", seed)
        run(ui, preview_id, expected, f"fixed-{seed}")


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("timing", ["before", "after"])
def test_fixed_policy_edit_survives_reload_and_resumes(ui, node_type, timing):
    control_timing(ui, timing)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.set_number(node_type, "seed", 5)
    ui.choose(node_type, "control_after_generate", "increment")
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(f"resumed {uuid4().hex}")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("{red|green|blue}")
    first = run(ui, preview_id, "blue", "initial-fixed")
    assert {node_id, preview_id}.isdisjoint(cached(first))

    # Editing the automatic policy must neither invalidate fixed output nor be
    # overwritten by the policy that was selected when the node was created.
    ui.choose(node_type, "control_after_generate", "decrement")
    before = native_roundtrip(ui, node_id)
    repeat = run(ui, preview_id, "blue", "restored-fixed")
    assert {node_id, preview_id} <= cached(repeat)
    assert widget_state(ui, node_id) == before

    source = next(w["value"] for w in before["widgets"] if w["name"] == "wildcard_text")
    ui.choose(node_type, "mode", "populate")
    run(ui, preview_id, source, "resumed-decrement")
    values = {w["name"]: w["value"] for w in widget_state(ui, node_id)["widgets"]}
    assert (values["seed"], values["control_after_generate"]) == (4, "decrement")


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("timing", ["before", "after"])
def test_linked_seed_remains_authoritative_in_fixed_mode(ui, node_type, timing):
    control_timing(ui, timing)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.choose(node_type, "control_after_generate", "randomize")
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(f"unused {uuid4().hex}")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("{red|green|blue}")
    seed_id = ui.add_node("ImpactInt", "ImpactInt", 850, 180)
    show_node(ui, seed_id)
    ui.set_number("ImpactInt", "value", 5)
    show_node(ui, node_id)
    ui.connect("ImpactInt", node_type, target_slot=input_slot(ui, node_type, "seed"))

    for label, seed, expected, reuse in (
        ("first", 5, "blue", False), ("repeat", 5, "blue", True), ("upstream-edit", 2, "red", False),
    ):
        show_node(ui, seed_id)
        ui.set_number("ImpactInt", "value", seed)
        show_node(ui, node_id)
        entry = run(ui, preview_id, expected, label)
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == [seed_id, 0]
        assert entry["prompt"][2][seed_id]["inputs"]["value"] == seed
        if reuse:
            assert {node_id, preview_id} <= cached(entry)
        else:
            assert {node_id, preview_id}.isdisjoint(cached(entry))


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_fixed_native_workflow_preserves_widgets(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill("unused {red|blue}")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("saved plain α")
    ui.set_number(node_type, "seed", 731)
    ui.choose(node_type, "control_after_generate", "increment")
    rendered(ui)
    before = native_roundtrip(ui, node_id)
    run(ui, preview_id, "saved plain α", "restored")
    second = run(ui, preview_id, "saved plain α", "restored-again")
    assert {node_id, preview_id} <= cached(second)
    assert widget_state(ui, node_id) == before


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("timing", ["before", "after"])
def test_fixed_seed_is_preserved_within_one_batched_queue(ui, node_type, timing):
    control_timing(ui, timing)
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.set_number(node_type, "seed", 5)
    ui.choose(node_type, "control_after_generate", "increment")
    source = f"batch {uuid4().hex}"
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill(source)
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill("{red|green|blue}")
    batch_count = ui.page.get_by_role("textbox", name="Batch Count", exact=True)
    batch_count.fill("3")
    batch_count.press("Enter")
    ui.run()  # One click submits all three prompts through the real queue loop.
    entries = ui.page.wait_for_function("""async ({url, id, source}) => {
        const response = await fetch(url + '/history');
        if (!response.ok) throw new Error('History request failed: ' + response.status);
        const history = await response.json();
        const entries = Object.values(history).filter(entry =>
            entry.prompt[2][id]?.inputs.wildcard_text === source);
        return entries.length >= 3 ? entries.sort((a, b) => a.prompt[0] - b.prompt[0]) : false;
    }""", arg={"url": ui.base_url, "id": node_id, "source": source},
        timeout=ui.timeout * 1000).json_value()
    (ui.artifacts / "batch-history.json").write_text(json.dumps(entries, indent=2) + "\n")
    assert len(entries) == len({entry["prompt"][1] for entry in entries}) == 3
    for index, entry in enumerate(entries):
        assert entry["status"]["status_str"] == "success"
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == 5
        assert entry["outputs"][preview_id]["text"] == ["blue"]
        if index:
            assert {node_id, preview_id} <= cached(entry)
        else:
            assert {node_id, preview_id}.isdisjoint(cached(entry))
    values = {w["name"]: w["value"] for w in widget_state(ui, node_id)["widgets"]}
    assert (values["seed"], values["control_after_generate"]) == (5, "increment")
    expect(ui.page.get_by_role("textbox", name="preview_text", exact=True)).to_have_value("blue")
