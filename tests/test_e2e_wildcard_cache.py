"""Real queue/cache behavior and native wildcard widget restoration in Chrome.

Use an isolated ComfyUI server. Encode cases use an installed checkpoint,
defaulting to v1-5-pruned-emaonly-fp16.safetensors (no downloads).
"""
from argparse import Namespace
import json
import os
from pathlib import Path
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


def native_roundtrip(ui, node_id):
    """Compare rendered widget names/order/values and the native serialization."""
    show_node(ui, node_id)
    before = widget_state(ui, node_id)
    trusted = json.loads((ROOT / "tests/workflows/all-impact-nodes.widgets.json").read_text())
    reference = next(node for node in trusted["nodes"] if node["type"] == before["type"])
    assert [w["name"] for w in before["widgets"]] == [w["name"] for w in reference["widgets"]]
    workflow = ui.artifacts / "saved.json"
    export_workflow(ui, workflow)
    saved = json.loads(workflow.read_text())
    ui.new_workflow()
    ui.load_workflow(workflow, len(saved["nodes"]))
    full_zoom(ui)
    show_node(ui, node_id)
    assert widget_state(ui, node_id) == before
    roundtrip = ui.artifacts / "roundtrip.json"
    export_workflow(ui, roundtrip)
    restored = json.loads(roundtrip.read_text())
    assert {str(n["id"]): n.get("widgets_values", []) for n in restored["nodes"]} == {
        str(n["id"]): n.get("widgets_values", []) for n in saved["nodes"]}
    # Native loading can reorder link records; their IDs and endpoints must
    # remain identical. Widget arrays above deliberately retain their order.
    assert sorted(restored["links"]) == sorted(saved["links"])
    assert restored.get("definitions") == saved.get("definitions")
    (ui.artifacts / "widget-state.json").write_text(json.dumps(before, indent=2) + "\n")
    return before


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


@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_plain_text_reuses_downstream_cache(ui, node_type):
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.choose(node_type, "control_after_generate", "increment")
    ui.set_number(node_type, "seed", 731)
    populated = ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True)
    text = f"ordinary prompt {uuid4().hex}"
    populated.fill(text)
    first = run(ui, preview_id, text, "first")
    second = run(ui, preview_id, text, "second")
    assert {node_id, preview_id}.isdisjoint(cached(first))
    assert {node_id, preview_id} <= cached(second)
    assert second["prompt"][2][node_id]["inputs"]["seed"] == 0
    ui.page.wait_for_function("""({app, id}) =>
        app.graph.getNodeById(id).widgets.find(w => w.name === 'seed').value === 733
    """, arg={"app": ui.app, "id": node_id})
    populated.fill(text + " changed")
    changed = run(ui, preview_id, text + " changed", "changed")
    assert {node_id, preview_id}.isdisjoint(cached(changed))


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("text", ["{red|\ngreen|blue}", "{red|{green}|blue}"], ids=["multiline", "nested"])
def test_fixed_syntax_preserves_seed(ui, node_type, text):
    _, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    ui.page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True).fill(text)
    for seed, expected in ((2, "red"), (5, "blue")):
        ui.set_number(node_type, "seed", seed)
        run(ui, preview_id, expected, f"fixed-{seed}")


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
    assert widget_state(ui, node_id)["widgets"] == [
        dict(widget, value=732) if widget["name"] == "seed" else widget for widget in before["widgets"]
    ]
