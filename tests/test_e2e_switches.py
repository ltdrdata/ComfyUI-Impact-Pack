"""Chrome E2E for Switch/Combine workflow loading and interactive links.

Requires a running ComfyUI server with Impact Pack. No models are needed.
"""
from argparse import Namespace
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import expect

from e2e_ui import ComfyUI, browser_session


def slots(ui):
    # The frontend restores schema-declared ANY types on load. Compare slot
    # identities and typed links, rather than transient labels/type narrowing.
    return ui.app.evaluate("""app => ({
        nodes: Object.fromEntries(app.graph._nodes.map(node => [node.id, {
            type: node.type,
            inputs: node.inputs.map(slot => ({name: slot.name, link: slot.link ?? null})),
            outputs: node.outputs.map(slot => slot.links ?? []),
            select: node.widgets?.find(widget => widget.name === 'select')?.value,
        }])),
        links: app.graph.serialize().links,
    })""")


def delete_node(ui, node_id):
    x, y = ui.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        return app.canvasPosToClientPos([node.pos[0] + node.size[0] / 2, node.pos[1] - 10]);
    }""", node_id)
    ui.page.mouse.click(x, y)
    ui.page.keyboard.press("Delete")
    ui.page.wait_for_function("({app, id}) => !app.graph.getNodeById(id)", arg={"app": ui.app, "id": node_id})
    return x, y + 60


@pytest.fixture
def ui(tmp_path, request):
    artifacts = Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30, artifacts=artifacts)
    with browser_session(args) as browser:
        # Fresh ComfyUI profiles open the template browser on startup.
        browser.page.keyboard.press("Escape")
        browser.page.set_default_timeout(10000)
        yield browser, args


CASES = [
    ("Switch (Any)", "ImpactSwitch"),
    ("Make List (Any)", "ImpactMakeAnyList"),
    ("Inversed Switch (Any)", "ImpactInversedSwitch"),
]

WORKFLOWS = {
    "ImpactSwitch": ("switch-inputs.json", {"20": ["second"]}),
    "ImpactMakeAnyList": ("list-inputs.json", {"20": ["first", "second", "third"]}),
    "ImpactInversedSwitch": ("inversed-switch-outputs.json", {"21": ["routed"]}),
}


def fixture_slots(workflow):
    nodes = {}
    for node in workflow["nodes"]:
        state = {
            "type": node["type"],
            "inputs": [{"name": slot["name"], "link": slot.get("link")} for slot in node["inputs"]],
            "outputs": [slot.get("links") or [] for slot in node["outputs"]],
            "select": node.get("widgets_values_named", {}).get("select"),
        }
        nodes[str(node["id"])] = state
    return {"nodes": nodes, "links": workflow["links"]}


def load_example(browser, node_type):
    filename, expected_text = WORKFLOWS[node_type]
    path = Path(__file__).parent / "workflows" / filename
    workflow = json.loads(path.read_text())
    browser.new_workflow()
    browser.load_workflow(path, len(workflow["nodes"]))
    assert slots(browser) == fixture_slots(workflow)
    return expected_text


def text_outputs(browser):
    return {node: output["text"] for node, output in browser.run()["outputs"].items() if "text" in output}


def create_workflow(ui, title, node_type):
    ui.new_workflow()
    source = ui.add_node("ImpactWildcardProcessor", "ImpactWildcardProcessor", 220, 230)
    ui.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill("switch-e2e")
    target = ui.add_node(title, node_type, 750, 230)
    preview = ui.add_node("Preview as Text", "PreviewAny", 1200, 520)
    ui.connect("ImpactWildcardProcessor", node_type)
    ui.connect(node_type, "PreviewAny")
    assert ui.run()["outputs"][preview]["text"] == ["switch-e2e"]
    workflow = ui.app.evaluate("app => app.graph.serialize()")
    return source, target, preview, {"name": f"{node_type}.json", "mimeType": "application/json",
                                     "buffer": json.dumps(workflow).encode("utf-8")}


@pytest.mark.parametrize("node_type", WORKFLOWS)
def test_workflow_import_and_browser_reload(ui, node_type):
    browser, args = ui
    expected = load_example(browser, node_type)
    before = slots(browser)
    assert text_outputs(browser) == expected

    browser.page.reload(wait_until="domcontentloaded")
    expect(browser.page.get_by_test_id("queue-button")).to_be_visible(timeout=60000)
    browser = ComfyUI(browser.page, args)
    browser.page.wait_for_function("""({app, count}) => app.graph._nodes.length === count && !app.configuringGraph""",
                                   arg={"app": browser.app, "count": len(before["nodes"])})
    assert slots(browser) == before
    assert text_outputs(browser) == expected


@pytest.mark.parametrize("title,node_type", CASES, ids=[case[1] for case in CASES])
def test_disconnect_and_reconnect_after_loading(ui, title, node_type):
    browser, _ = ui
    source, target, preview, payload = create_workflow(browser, title, node_type)
    before = slots(browser)
    browser.new_workflow()
    browser.load_workflow(payload, 3)
    assert slots(browser) == before
    before = before["nodes"][target]
    if node_type == "ImpactInversedSwitch":
        position = delete_node(browser, preview)
        assert len(slots(browser)["nodes"][target]["outputs"]) == len(before["outputs"]) - 1
        preview = browser.add_node("Preview as Text", "PreviewAny", *position)
        browser.connect(node_type, "PreviewAny")
    else:
        position = delete_node(browser, source)
        assert len(slots(browser)["nodes"][target]["inputs"]) == len(before["inputs"]) - 1
        browser.add_node("ImpactWildcardProcessor", "ImpactWildcardProcessor", *position)
        target_slot = browser.app.evaluate("""(app, type) => app.graph._nodes.find(node => node.type === type)
            .inputs.findIndex(slot => slot.name === (type === 'ImpactSwitch' ? 'input1' : 'value1'))""", node_type)
        browser.connect("ImpactWildcardProcessor", node_type, target_slot)
    browser.page.get_by_placeholder("Wildcard Prompt (User input)", exact=True).fill("reconnected-e2e")
    assert browser.run()["outputs"][preview]["text"] == ["reconnected-e2e"]


@pytest.mark.parametrize("node_type,removed,side", [
    ("ImpactSwitch", 3, "inputs"),
    ("ImpactInversedSwitch", 22, "outputs"),
])
def test_keep_selection_when_removing_another_connection(ui, node_type, removed, side):
    browser, _ = ui
    expected = load_example(browser, node_type)
    assert text_outputs(browser) == expected
    before = slots(browser)["nodes"]["10"]
    delete_node(browser, removed)
    after = slots(browser)["nodes"]["10"]
    assert len(after[side]) == len(before[side]) - 1
    assert after["select"] == 2
    assert text_outputs(browser) == expected


def test_inversed_switch_keeps_other_outputs_when_reroute_disconnects_during_load(ui):
    browser, _ = ui
    path = Path(__file__).parent / "workflows" / "inversed-switch-outputs.json"
    workflow = json.loads(path.read_text())
    reroute = next(node for node in workflow["nodes"] if node["id"] == 20)
    reroute.update(type="Reroute", size=[100, 26], widgets_values=[],
                   inputs=[{"name": "", "type": "*", "link": 2}],
                   outputs=[{"name": "", "type": "STRING", "links": None}])
    expected = fixture_slots(workflow)
    # Impact Pack rejects Reroute links, including links saved by older frontends.
    # Disconnecting output1 during configure must leave output2/output3 in place.
    expected["links"] = [link for link in expected["links"] if link[0] != 2]
    expected["nodes"]["10"]["outputs"][0] = []
    expected["nodes"]["20"]["inputs"][0]["link"] = None
    browser.new_workflow()
    payload = {"name": "inversed-switch-reroute.json", "mimeType": "application/json",
               "buffer": json.dumps(workflow).encode("utf-8")}
    browser.load_workflow(payload, len(workflow["nodes"]))
    assert slots(browser) == expected
    assert text_outputs(browser) == {"21": ["routed"]}


@pytest.mark.parametrize("node_type", ["ImpactSwitch", "ImpactInversedSwitch", "LatentSwitch", "SEGSSwitch"])
@pytest.mark.parametrize("select_on_prompt", [False, True], ids=["on_execution", "on_prompt"])
def test_switch_selection_modes_and_legacy_names(ui, node_type, select_on_prompt):
    browser, _ = ui
    original_type = "ImpactInversedSwitch" if node_type == "ImpactInversedSwitch" else "ImpactSwitch"
    filename, expected = WORKFLOWS[original_type]
    workflow = json.loads((Path(__file__).parent / "workflows" / filename).read_text())
    target = next(node for node in workflow["nodes"] if node["id"] == 10)
    target["type"] = node_type
    target["widgets_values"][1] = select_on_prompt
    target["widgets_values_named"]["sel_mode"] = select_on_prompt
    browser.new_workflow()
    browser.load_workflow({"name": f"{node_type}-{select_on_prompt}.json", "mimeType": "application/json",
                           "buffer": json.dumps(workflow).encode("utf-8")}, len(workflow["nodes"]))
    assert slots(browser) == fixture_slots(workflow)
    assert text_outputs(browser) == expected


@pytest.mark.parametrize("node_type,second_size", [
    ("ImpactMakeImageList", 32), ("ImpactMakeImageBatch", 32), ("ImpactMakeMaskList", 32),
    ("ImpactMakeMaskBatch", 16), ("ImpactMakeMaskBatch", 32),
])
def test_image_and_mask_combine_after_loading(ui, node_type, second_size):
    browser, _ = ui
    is_mask = "Mask" in node_type
    prefix = "mask" if is_mask else "image"
    prompt = {
        "first": {"class_type": "SolidMask" if is_mask else "EmptyImage",
                  "inputs": {"width": 16, "height": 16,
                             **({"value": 0.0} if is_mask else {"batch_size": 1, "color": 0xFF0000})}},
        "second": {"class_type": "SolidMask" if is_mask else "EmptyImage",
                   "inputs": {"width": second_size, "height": second_size,
                              **({"value": 1.0} if is_mask else {"batch_size": 1, "color": 0x0000FF})}},
        "combine": {"class_type": node_type, "inputs": {f"{prefix}1": ["first", 0], f"{prefix}2": ["second", 0]}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["to_image" if is_mask else "combine", 0]}},
    }
    if is_mask:
        prompt["to_image"] = {"class_type": "MaskToImage", "inputs": {"mask": ["combine", 0]}}
    browser.new_workflow()
    browser.load_workflow({"name": f"{node_type}-api.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode("utf-8")}, len(prompt))
    before = slots(browser)
    assert [slot["name"] for slot in before["nodes"]["combine"]["inputs"]] == [f"{prefix}{i}" for i in (1, 2, 3)]
    saved = browser.app.evaluate("app => app.graph.serialize()")
    colors = [(0, 0, 0), (255, 255, 255)] if is_mask else [(255, 0, 0), (0, 0, 255)]
    sizes = [(16, 16), (16, 16) if node_type.endswith("Batch") else (second_size, second_size)]
    for reload_workflow in (False, True):
        if reload_workflow:
            browser.new_workflow()
            browser.load_workflow({"name": f"{node_type}.json", "mimeType": "application/json",
                                   "buffer": json.dumps(saved).encode("utf-8")}, len(prompt))
            assert slots(browser) == before
        images = browser.run()["outputs"]["preview"]["images"]
        assert len(images) == 2
        for meta, color, size in zip(images, colors, sizes):
            response = browser.page.request.get(f"{browser.base_url}/view", params=meta)
            assert response.ok
            with Image.open(io.BytesIO(response.body())) as image:
                assert image.size == size
                assert image.convert("RGB").getextrema() == tuple((value, value) for value in color)


def test_segs_concat_after_loading(ui):
    browser, _ = ui
    prompt = {
        "mask": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 32, "height": 32}},
        "segs": {"class_type": "MaskToSEGS", "inputs": {"mask": ["mask", 0], "combined": False,
                 "crop_factor": 1.0, "bbox_fill": False, "drop_size": 1, "contour_fill": False}},
        "combine": {"class_type": "ImpactSEGSConcat", "inputs": {"segs1": ["segs", 0], "segs2": ["segs", 0]}},
        "count": {"class_type": "ImpactCount_Elts_in_SEGS", "inputs": {"segs": ["combine", 0]}},
        "preview": {"class_type": "PreviewAny", "inputs": {"source": ["count", 0]}},
    }
    browser.new_workflow()
    browser.load_workflow({"name": "segs-concat-api.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode("utf-8")}, len(prompt))
    before = slots(browser)
    assert [slot["name"] for slot in before["nodes"]["combine"]["inputs"]] == ["segs1", "segs2", "segs3"]
    assert text_outputs(browser) == {"preview": ["2"]}
    saved = browser.app.evaluate("app => app.graph.serialize()")
    browser.new_workflow()
    browser.load_workflow({"name": "segs-concat.json", "mimeType": "application/json",
                           "buffer": json.dumps(saved).encode("utf-8")}, len(prompt))
    assert slots(browser) == before
    assert text_outputs(browser) == {"preview": ["2"]}
