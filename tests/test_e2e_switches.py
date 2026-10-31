"""Chrome E2E for Switch/Combine workflow loading and interactive links.

Requires a running ComfyUI server with Impact Pack. No models are needed.
"""
from argparse import Namespace
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, expect

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


@pytest.mark.parametrize("title,node_type", CASES, ids=[case[1] for case in CASES])
@pytest.mark.parametrize("output_first", [False, True], ids=["input_first", "output_first"])
def test_wildcard_output_connections_in_native_workflow(ui, title, node_type, output_first):
    browser, _ = ui
    browser.new_workflow()
    source = browser.add_node("ImpactValueReceiver", "ImpactValueReceiver", 220, 230)
    target = browser.add_node(title, node_type, 750, 230)
    preview = browser.add_node("Preview as Text", "PreviewAny", 1200, 520)
    browser.choose("ImpactValueReceiver", "typ", "BOOLEAN")
    target_slot = browser.app.evaluate("""(app, id) => app.graph.getNodeById(id).inputs.findIndex(
        slot => ['input', 'input1', 'value1'].includes(slot.name))""", target)
    if output_first:
        browser.connect(node_type, "PreviewAny")
    browser.connect("ImpactValueReceiver", node_type, target_slot)
    if not output_first:
        browser.connect(node_type, "PreviewAny")
    assert text_outputs(browser) == {preview: ["False"]}
    before = slots(browser)
    widget_names = {source: ["typ", "value", "link_id"], target: [] if node_type == "ImpactMakeAnyList" else ["select", "sel_mode"]}
    widgets = browser.app.evaluate("""(app, ids) => Object.fromEntries(ids.map(id =>
        [id, (app.graph.getNodeById(id).widgets ?? []).map(widget => widget.name)]))""", [source, target])
    assert widgets == widget_names
    saved = browser.app.evaluate("app => app.graph.serialize()")
    browser.new_workflow()
    browser.load_workflow({"name": "wildcard-output-native.json", "mimeType": "application/json",
                           "buffer": json.dumps(saved).encode()}, 3)
    assert slots(browser) == before
    after = browser.app.evaluate("app => app.graph.serialize()")
    for node_id in (source, target):
        original = next(node for node in saved["nodes"] if str(node["id"]) == node_id)
        restored = next(node for node in after["nodes"] if str(node["id"]) == node_id)
        assert restored.get("widgets_values", []) == original.get("widgets_values", [])
    assert browser.app.evaluate("""(app, ids) => Object.fromEntries(ids.map(id =>
        [id, (app.graph.getNodeById(id).widgets ?? []).map(widget => widget.name)]))""", [source, target]) == widget_names
    assert text_outputs(browser) == {preview: ["False"]}
    browser.choose("ImpactValueReceiver", "typ", "STRING")
    assert text_outputs(browser) == {preview: [""]}


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


@pytest.mark.parametrize("node_type", ["ImpactSwitch", "LatentSwitch", "SEGSSwitch"])
@pytest.mark.parametrize("select_on_prompt", [False, True], ids=["on_execution", "on_prompt"])
def test_inversed_switch_branches_rejoin(ui, node_type, select_on_prompt):
    browser, _ = ui
    prompt = {
        "source": {"class_type": "EmptyImage", "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0xFF0000}},
        "select": {"class_type": "ImpactInt", "inputs": {"value": 2}},
        "inverse": {"class_type": "ImpactInversedSwitch", "inputs": {
            "input": ["source", 0], "select": ["select", 0], "sel_mode": select_on_prompt}},
        "branch1": {"class_type": "ImageInvert", "inputs": {"image": ["inverse", 0]}},
        "branch2": {"class_type": "ImageToMask", "inputs": {"image": ["inverse", 1], "channel": "red"}},
        "branch3": {"class_type": "ImageToMask", "inputs": {"image": ["inverse", 2], "channel": "green"}},
        "white": {"class_type": "MaskToImage", "inputs": {"mask": ["branch2", 0]}},
        "black": {"class_type": "MaskToImage", "inputs": {"mask": ["branch3", 0]}},
        "switch": {"class_type": node_type, "inputs": {
            "input1": ["branch1", 0], "input2": ["white", 0], "input3": ["black", 0],
            "select": ["select", 0], "sel_mode": select_on_prompt}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["switch", 0]}},
    }
    if select_on_prompt:
        for index in (1, 2, 3):
            prompt[f"side{index}"] = {"class_type": "PreviewImage", "inputs": {"images": ["inverse", index - 1]}}
    else:
        # Execution mode accepts an INT computed by an upstream node, not just
        # the static ImpactInt/Primitive selectors supported by prompt mode.
        prompt["choice"] = prompt.pop("select")
        prompt["selector_image"] = {"class_type": "EmptyImage", "inputs": {
            "width": 16, "height": 16, "batch_size": ["choice", 0], "color": 0}}
        prompt["select"] = {"class_type": "GetImageSize", "inputs": {"image": ["selector_image", 0]}}
        for node in ("inverse", "switch"):
            prompt[node]["inputs"]["select"] = ["select", 2]
    browser.new_workflow()
    browser.load_workflow({"name": "switch-branches-api.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode()}, len(prompt))
    before = slots(browser)
    for value in (2, 1, 3, 2, None):
        if value is None:
            saved = browser.app.evaluate("app => app.graph.serialize()")
            browser.new_workflow()
            browser.load_workflow({"name": "switch-branches.json", "mimeType": "application/json",
                                   "buffer": json.dumps(saved).encode()}, len(prompt))
            assert slots(browser) == before
            value = 2
        else:
            browser.set_number("ImpactInt", "value", value)
        entry = browser.run()
        outputs = browser.images(entry)
        expected = {"preview": {1: (0, 255, 255), 2: (255, 255, 255), 3: (0, 0, 0)}[value]}
        if select_on_prompt:
            expected[f"side{value}"] = (255, 0, 0)
            executed_prompt = entry["prompt"][2]
            for index in (1, 2, 3):
                # Every consumer must retain only the selected output, regardless of node order.
                link = ["inverse", index - 1] if index == value else None
                assert executed_prompt[f"branch{index}"]["inputs"].get("image") == link
                assert executed_prompt[f"side{index}"]["inputs"].get("images") == link
            assert {name for name in executed_prompt["switch"]["inputs"] if name.startswith("input")} == {f"input{value}"}
        assert outputs.keys() == expected.keys()
        for node, color in expected.items():
            assert len(outputs[node]) == 1
            image = outputs[node][0]
            assert image.size == (16, 16)
            assert image.getextrema() == tuple((v, v) for v in color)


@pytest.mark.parametrize("select_on_prompt", [False, True], ids=["on_execution", "on_prompt"])
def test_switch_does_not_execute_unselected_input(ui, select_on_prompt):
    browser, _ = ui
    prompt = {
        "select": {"class_type": "ImpactInt", "inputs": {"value": 1}},
        "safe": {"class_type": "PrimitiveInt", "inputs": {"value": 7}},
        "poison": {"class_type": "ImpactValueReceiver", "inputs": {"typ": "INT", "value": "not-an-integer", "link_id": 0}},
        "poison_int": {"class_type": "ImpactConvertDataType", "inputs": {"value": ["poison", 0]}},
        "switch": {"class_type": "ImpactSwitch", "inputs": {
            "select": ["select", 0], "sel_mode": select_on_prompt, "input1": ["safe", 0], "input2": ["poison_int", 2]}},
        "preview": {"class_type": "PreviewAny", "inputs": {"source": ["switch", 0]}},
    }
    browser.new_workflow()
    browser.load_workflow({"name": "unselected-input.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode()}, len(prompt))
    inputs = slots(browser)["nodes"]["switch"]["inputs"]
    assert {slot["name"] for slot in inputs if slot["link"] is not None} == {"select", "input1", "input2"}
    assert text_outputs(browser) == {"preview": ["7"]}
    saved = browser.app.evaluate("app => app.graph.serialize()")
    browser.new_workflow()
    browser.load_workflow({"name": "unselected-input-saved.json", "mimeType": "application/json",
                           "buffer": json.dumps(saved).encode()}, len(prompt))
    assert text_outputs(browser) == {"preview": ["7"]}
    # Prove the unselected branch really would fail if it were evaluated.
    browser.set_number("ImpactInt", "value", 2)
    with pytest.raises(AssertionError, match="invalid literal for int"):
        browser.run()
    browser.page.keyboard.press("Escape")
    browser.set_number("ImpactInt", "value", 1)
    assert text_outputs(browser) == {"preview": ["7"]}


def test_inversed_switch_disconnected_chain_on_prompt(ui):
    browser, _ = ui
    prompt = {
        "source": {"class_type": "EmptyImage", "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0xFF0000}},
        "select": {"class_type": "ImpactInt", "inputs": {"value": 1}},
        "first": {"class_type": "ImpactInversedSwitch", "inputs": {
            "input": ["source", 0], "select": ["select", 0], "sel_mode": True}},
        "second": {"class_type": "ImpactInversedSwitch", "inputs": {
            "input": ["first", 0], "select": ["select", 0], "sel_mode": True}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["second", 0]}},
        "control": {"class_type": "PreviewImage", "inputs": {"images": ["source", 0]}},
    }
    browser.new_workflow()
    browser.load_workflow({"name": "inversed-switch-chain.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode()}, len(prompt))
    for value in (1, 2, 1):
        browser.set_number("ImpactInt", "value", value)
        entry = browser.run()
        outputs = set(browser.images(entry))
        assert outputs == ({"control", "preview"} if value == 1 else {"control"})
        executed_prompt = entry["prompt"][2]
        assert ("input" in executed_prompt["first"]["inputs"]) == (value == 1)
        assert ("input" in executed_prompt["second"]["inputs"]) == (value == 1)


@pytest.mark.parametrize("failure", [None, "blocked", "stale"], ids=["updated", "blocked", "stale"])
def test_image_previews_follow_current_execution(ui, failure):
    browser, _ = ui
    prompt = {
        "color": {"class_type": "ImpactInt", "inputs": {"value": 0xFF0000}},
        "source": {"class_type": "EmptyImage", "inputs": {
            "width": 16, "height": 16, "batch_size": 1, "color": ["color", 0]}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["source", 0]}},
    }
    browser.new_workflow()
    browser.load_workflow({"name": "preview-update.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode()}, len(prompt))
    entry = browser.run()
    assert browser.images(entry)["preview"][0].getextrema() == ((255, 255), (0, 0), (0, 0))
    if failure == "stale":
        meta = entry["outputs"]["preview"]["images"][0]
        old_image = browser.page.request.get(f"{browser.base_url}/view", params={
            key: meta[key] for key in ("filename", "subfolder", "type")
        }).body()
    intercepted = []

    def intercept(route):
        if route.request.resource_type != "image":
            route.continue_()
        else:
            intercepted.append(route.request.url)
            if failure == "blocked":
                route.abort()
            else:
                route.fulfill(content_type="image/png", body=old_image)

    if failure:
        browser.page.route("**/view?**", intercept)
    browser.set_number("ImpactInt", "value", 0x0000FF)
    entry = browser.run()
    if failure == "blocked":
        browser.page.set_default_timeout(1000)
        with pytest.raises(PlaywrightTimeoutError, match="Page.wait_for_function"):
            browser.images(entry)
    elif failure == "stale":
        with pytest.raises(AssertionError, match=r"Preview preview\[0\] pixels differ"):
            browser.images(entry)
    else:
        assert browser.images(entry)["preview"][0].getextrema() == ((0, 0), (0, 0), (255, 255))
    if failure:
        assert intercepted


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
        images = browser.images(browser.run())["preview"]
        assert len(images) == 2
        for image, color, size in zip(images, colors, sizes):
            assert image.size == size
            assert image.getextrema() == tuple((value, value) for value in color)


@pytest.mark.parametrize("sizes,crops", [
    ([(12, 8)], [None]),
    ([(12, 8), (12, 8)], [None, None]),
    ([(12, 8), (48, 16), (12, 8)], [None, (12, 0, 36, 16), None]),
    ([(24, 16), (6, 8)], [None, (0, 2, 6, 6)]),
    ([(8, 1), (16, 4)], [None, (0, 1, 16, 3)]),
    ([(1, 8), (4, 16)], [None, (1, 0, 3, 16)]),
], ids=["single", "same_size", "downscale_crop", "upscale_crop", "height_one", "width_one"])
def test_mask_batch_preserves_patterns_and_order(ui, sizes, crops):
    browser, _ = ui
    prompt = {}
    inputs = {}
    expected = []
    for index, ((width, height), crop) in enumerate(zip(sizes, crops), 1):
        prefix = f"source{index}"
        value = index / 4
        x, y = width // 4, 0
        patch_width, patch_height = max(1, width // 2), max(1, height // 2)
        prompt.update({
            f"{prefix}_base": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": width, "height": height}},
            f"{prefix}_patch": {"class_type": "SolidMask", "inputs": {
                "value": value, "width": patch_width, "height": patch_height}},
            f"{prefix}_pattern": {"class_type": "MaskComposite", "inputs": {
                "destination": [f"{prefix}_base", 0], "source": [f"{prefix}_patch", 0],
                "x": x, "y": y, "operation": "add"}},
            f"{prefix}_inverse": {"class_type": "InvertMask", "inputs": {"mask": [f"{prefix}_pattern", 0]}},
            f"{prefix}_image": {"class_type": "MaskToImage", "inputs": {"mask": [f"{prefix}_pattern", 0]}},
            f"{prefix}_inverse_image": {"class_type": "MaskToImage", "inputs": {"mask": [f"{prefix}_inverse", 0]}},
            f"{prefix}_batch": {"class_type": "ImageBatch", "inputs": {
                "image1": [f"{prefix}_image", 0], "image2": [f"{prefix}_inverse_image", 0]}},
            prefix: {"class_type": "ImageToMask", "inputs": {"image": [f"{prefix}_batch", 0], "channel": "red"}},
        })
        inputs[f"mask{index}"] = [prefix, 0]
        # Explicit crop boxes are independent of MakeMaskBatch's resizing code.
        for background, foreground in ((0, value), (255, 1 - value)):
            image = Image.new("L", (width, height), background)
            image.paste(int(foreground * 255), (x, y, x + patch_width, y + patch_height))
            if crop is not None:
                image = image.crop(crop).resize(sizes[0], Image.Resampling.LANCZOS)
            expected.append(image.convert("RGB"))
    prompt.update({
        "combine": {"class_type": "ImpactMakeMaskBatch", "inputs": inputs},
        "to_image": {"class_type": "MaskToImage", "inputs": {"mask": ["combine", 0]}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["to_image", 0]}},
    })
    browser.new_workflow()
    browser.load_workflow({"name": "mask-batch-patterns-api.json", "mimeType": "application/json",
                           "buffer": json.dumps(prompt).encode("utf-8")}, len(prompt))
    before = slots(browser)
    assert [slot["name"] for slot in before["nodes"]["combine"]["inputs"]] == [
        f"mask{i}" for i in range(1, len(sizes) + 2)
    ]
    saved = browser.app.evaluate("app => app.graph.serialize()")
    for reload_workflow in (False, True):
        if reload_workflow:
            browser.new_workflow()
            browser.load_workflow({"name": "mask-batch-patterns.json", "mimeType": "application/json",
                                   "buffer": json.dumps(saved).encode("utf-8")}, len(prompt))
            assert slots(browser) == before
        images = browser.images(browser.run())["preview"]
        assert len(images) == len(expected)
        for index, (image, expected_image) in enumerate(zip(images, expected)):
            assert image.size == sizes[0]
            assert image.tobytes() == expected_image.tobytes(), f"Mask {index} pixels differ"


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
