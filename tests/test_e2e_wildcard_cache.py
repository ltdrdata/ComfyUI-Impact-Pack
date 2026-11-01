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


def connect(ui, source, target, source_slot=0, target_slot=0):
    page = ui.page
    page.mouse.move(*ui.point(source, slot=source_slot))
    page.mouse.down()
    page.mouse.move(*ui.point(target, slot=target_slot, is_input=True), steps=12)
    page.mouse.up()
    page.wait_for_function("""({app, type, slot}) =>
        app.graph._nodes.find(n => n.type === type).inputs[slot].link != null
    """, arg={"app": ui.app, "type": target, "slot": target_slot})


def graph(ui, node_type):
    node_id = ui.add_node(node_type, node_type, 550, 170)
    preview_id = ui.add_node("Preview as Text", "PreviewAny", 1150, 450)
    connect(ui, node_type, "PreviewAny", 3 if node_type == "ImpactWildcardEncode" else 0)
    if node_type == "ImpactWildcardEncode":
        ui.add_node("Load Checkpoint", "CheckpointLoaderSimple", 100, 500)
        checkpoint = os.environ.get("IMPACT_E2E_CHECKPOINT", "v1-5-pruned-emaonly-fp16.safetensors")
        ui.choose("CheckpointLoaderSimple", "ckpt_name", checkpoint)
        connect(ui, "CheckpointLoaderSimple", node_type, 0, 0)
        connect(ui, "CheckpointLoaderSimple", node_type, 1, 1)
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


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_seeded_output_and_native_widgets(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id = graph(ui, node_type)
    page = ui.page
    wildcard = page.get_by_placeholder("Wildcard Prompt (User input)", exact=True)
    populated = page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True)
    wildcard.fill("{red|green|blue}")
    # Trusted pre-change selections: seed 2 -> red, seed 5 -> blue.
    for seed, expected in ((2, "red"), (5, "blue")):
        ui.set_number(node_type, "seed", seed)
        run(ui, preview_id, expected, f"populate-{seed}")
        expect(populated).to_have_value(expected)

    ui.choose(node_type, "mode", "fixed")
    for label, text in (("multiline", "{red|\ngreen|blue}"), ("nested", "{red|{green}|blue}")):
        populated.fill(text)
        for seed, expected in ((2, "red"), (5, "blue")):
            ui.set_number(node_type, "seed", seed)
            run(ui, preview_id, expected, f"{label}-{seed}")

    populated.fill("saved plain α")
    ui.set_number(node_type, "seed", 731)
    ui.choose(node_type, "control_after_generate", "increment")
    rendered(ui)
    before = widget_state(ui, node_id)
    trusted = json.loads((ROOT / "tests/workflows/all-impact-nodes.widgets.json").read_text())
    reference = next(node for node in trusted["nodes"] if node["type"] == node_type)
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
    saved_widgets = {str(node["id"]): node.get("widgets_values", []) for node in saved["nodes"]}
    assert {str(node["id"]): node.get("widgets_values", []) for node in restored["nodes"]} == saved_widgets
    (ui.artifacts / "widget-state.json").write_text(json.dumps(before, indent=2) + "\n")
    run(ui, preview_id, "saved plain α", "restored")
    assert widget_state(ui, node_id)["widgets"] == [
        dict(widget, value=732) if widget["name"] == "seed" else widget for widget in before["widgets"]
    ]
