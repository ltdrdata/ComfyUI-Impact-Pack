"""Chrome E2E for the Preview Bridge image/latent tutorial workflows.

Requires an isolated ComfyUI server with Impact Pack, Playwright and Chrome.
Image compositing and a second latent bridge observe the edited image, MASK
and latent noise_mask independently. Only the legacy SAM cases require model
weights (SAM ViT-B and the SD1.5 VAE).
"""
from argparse import Namespace
import base64
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageOps
from playwright.sync_api import expect
from safetensors.numpy import load as load_safetensors

from e2e_ui import browser_session
from test_e2e_legacy_sam import upload_image
from test_e2e_widget_catalog import export_workflow, setting


BRIDGES = ["PreviewBridge", "PreviewBridgeLatent"]
WIDGET_NAMES = {
    "PreviewBridge": ["image", "block", "restore_mask", "alpha_as_mask", "$$canvas-image-preview"],
    "PreviewBridgeLatent": ["image", "preview_method", "block", "restore_mask", "$$canvas-image-preview"],
}


@pytest.fixture
def ui(tmp_path, request):
    artifacts = Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30, artifacts=artifacts)
    with browser_session(args) as browser:
        browser.page.keyboard.press("Escape")
        browser.page.set_viewport_size({"width": 2200, "height": 1300})
        browser.page.set_default_timeout(10000)
        browser.new_workflow()
        yield browser, args


def add_node(ui, node_type, x, y):
    previous = ui.app.evaluate("app => app.graph._nodes.map(node => node.id)")
    response = ui.page.request.get(f"{ui.base_url}/object_info/{node_type}")
    assert response.ok
    title = response.json()[node_type]["display_name"] or node_type
    ui.add_node(title, node_type, x, y)
    return str(ui.app.evaluate("""(app, previous) =>
        app.graph._nodes.find(node => !previous.includes(node.id)).id
    """, previous))


def point(ui, node_id, *, widget=None, slot=0, is_input=False, menu=False):
    return ui.app.evaluate("""(app, {id, widget, slot, isInput, menu}) => {
        const node = app.graph.getNodeById(id);
        const pos = menu ? [node.pos[0] + node.size[0] / 2, node.pos[1] + 15]
            : widget ? [node.pos[0] + node.size[0] / 2,
                        node.pos[1] + node.widgets.find(w => w.name === widget).last_y + 10]
            : node.getConnectionPos(isInput, slot);
        return app.canvasPosToClientPos(pos);
    }""", {"id": node_id, "widget": widget, "slot": slot, "isInput": is_input, "menu": menu})


def connect(ui, source, target, source_slot=0, target_slot=0):
    if isinstance(target_slot, str):
        target_slot = ui.app.evaluate("""(app, {id, name}) =>
            app.graph.getNodeById(id).inputs.findIndex(input => input.name === name)
        """, {"id": target, "name": target_slot})
        assert target_slot >= 0, "Target input was not found"
    ui.page.mouse.move(*point(ui, source, slot=source_slot))
    ui.page.mouse.down()
    ui.page.mouse.move(*point(ui, target, slot=target_slot, is_input=True), steps=12)
    ui.page.mouse.up()
    ui.page.wait_for_function("""({app, source, target, sourceSlot, targetSlot}) => {
        const input = app.graph.getNodeById(target).inputs[targetSlot];
        const link = app.graph.links.get(input.link);
        return link && String(link.origin_id) === source && link.origin_slot === sourceSlot;
    }""", arg={"app": ui.app, "source": source, "target": target,
               "sourceSlot": source_slot, "targetSlot": target_slot})


def number(ui, node_id, name, value):
    ui.page.mouse.dblclick(*point(ui, node_id, widget=name), delay=100)
    editor = ui.page.locator("input.value:visible")
    editor.fill(str(value))
    editor.press("Enter")
    ui.page.wait_for_function("""({app, id, name, value}) =>
        app.graph.getNodeById(id).widgets.find(w => w.name === name).value === value
    """, arg={"app": ui.app, "id": node_id, "name": name, "value": value})


def choose(ui, node_id, name, value):
    ui.page.mouse.click(*point(ui, node_id, widget=name))
    ui.page.get_by_role("menuitem", name=value, exact=True).click()


def color(ui, node_id, value):
    ui.page.mouse.click(*point(ui, node_id, widget="color"))
    editor = ui.page.locator('input[type="color"], input.value:visible')
    editor.wait_for(state="attached")
    if editor.get_attribute("type") == "color":
        editor.fill(f"#{value:06x}")
    else:
        editor.fill(str(value))
        editor.press("Enter")
    ui.page.keyboard.press("Escape")
    ui.page.wait_for_function("""({app, id, value}) =>
        app.graph.getNodeById(id).widgets.find(w => w.name === 'color').value === value
    """, arg={"app": ui.app, "id": node_id, "value": value})


def context_menu(ui, node_id, name):
    ui.page.mouse.click(*point(ui, node_id, menu=True), button="right")
    ui.page.get_by_role("menuitem", name=name).click()


def file_bytes(ui, meta):
    response = ui.page.request.get(f"{ui.base_url}/view", params={
        key: meta[key] for key in ("filename", "subfolder", "type") if key in meta})
    assert response.ok, f"Cannot read output: {meta} ({response.status})"
    return response.body()


def pixels_equal(actual, expected, tolerance=0):
    assert actual.size == expected.size, f"Image size differs: {actual.size} != {expected.size}"
    difference = np.abs(np.asarray(actual, dtype=np.int16) - np.asarray(expected, dtype=np.int16))
    assert difference.max() <= tolerance, f"Pixels differ: max error {difference.max()} > {tolerance}"


def preview(ui, entry, node_id):
    meta, = entry["outputs"][node_id]["images"]
    with Image.open(io.BytesIO(file_bytes(ui, meta))) as image:
        expected = image.convert("RGBA")
    ui.page.wait_for_function("""({app, id, meta}) => {
        const image = app.graph.getNodeById(id)?.imgs?.[0];
        if (!image?.complete || !image.naturalWidth) return false;
        const params = new URL(image.src).searchParams;
        return ['filename', 'subfolder', 'type'].every(key =>
            (params.get(key) ?? '') === (meta[key] ?? ''));
    }""", arg={"app": ui.app, "id": node_id, "meta": meta})
    encoded = ui.app.evaluate("""(app, id) => {
        const image = app.graph.getNodeById(id).imgs[0];
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.getContext('2d').drawImage(image, 0, 0);
        return canvas.toDataURL('image/png').split(',')[1];
    }""", node_id)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        displayed = image.convert("RGBA")
    pixels_equal(displayed.getchannel("A"), expected.getchannel("A"))
    # Canvas discards RGB beneath zero alpha. Compare visible colors on black.
    background = Image.new("RGBA", expected.size, (0, 0, 0, 255))
    pixels_equal(Image.alpha_composite(background, displayed),
                 Image.alpha_composite(background, expected), tolerance=1)
    return expected


def mask_output(ui, entry, node_id, expected):
    actual = preview(ui, entry, node_id).convert("L")
    # MASK float32 -> PNG truncation can lose one 8-bit level.
    pixels_equal(actual, expected, tolerance=1)
    return actual


def editor_mask(ui):
    encoded = ui.page.evaluate("""() => document.querySelectorAll(
        '#maskEditorCanvasContainer canvas')[2].toDataURL('image/png').split(',')[1]
    """)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return image.getchannel("A")


def editor_image(ui):
    layers = ui.page.evaluate("""() => [...document.querySelectorAll(
        '#maskEditorCanvasContainer canvas')].slice(0, 2)
        .map(canvas => canvas.toDataURL('image/png').split(',')[1])
    """)
    images = []
    for encoded in layers:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            images.append(image.convert("RGBA"))
    return Image.alpha_composite(*images).convert("RGB")


def edit_mask(ui, node_id, artifacts, name, *, right=False, existing=None):
    context_menu(ui, node_id, re.compile(r"^Open in MaskEditor"))
    expect(ui.page.get_by_test_id("mask-editor-root")).to_be_visible()
    zone = ui.page.get_by_test_id("pointer-zone")
    expect(zone).to_be_visible()
    if existing is not None:
        ui.page.wait_for_function("""() => {
            const canvas = document.querySelectorAll('#maskEditorCanvasContainer canvas')[2];
            const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
            return data.some((value, index) => index % 4 === 3 && value > 0);
        }""")
        pixels_equal(editor_mask(ui), existing, tolerance=1)
    ui.page.locator('[data-tool="pen"]').click()
    ui.page.get_by_role("button", name=re.compile(r"Clear$")).click()
    box = zone.bounding_box()
    start, end = (0.60, 0.75) if right else (0.25, 0.40)
    ui.page.mouse.move(box["x"] + box["width"] * start, box["y"] + box["height"] * 0.45)
    ui.page.mouse.down()
    ui.page.mouse.move(box["x"] + box["width"] * end, box["y"] + box["height"] * 0.55, steps=12)
    ui.page.mouse.up()
    expected = editor_mask(ui)
    count = np.count_nonzero(np.asarray(expected))
    assert 0 < count < expected.width * expected.height / 2, "Draw a nonempty, localized mask"
    expected.save(artifacts / f"{name}-expected-mask.png")
    ui.page.screenshot(path=str(artifacts / f"{name}-editor.png"))
    ui.page.get_by_role("button", name=re.compile(r"Save$")).click()
    expect(zone).not_to_be_visible()
    return expected


def build_bridge(ui, node_type):
    latent = node_type == "PreviewBridgeLatent"
    source = add_node(ui, "EmptyLatentImage" if latent else "EmptyImage", 220, 200)
    number(ui, source, "width", 256)
    number(ui, source, "height", 256)
    if not latent:
        color(ui, source, 0x4080C0)
    bridge = add_node(ui, node_type, 720, 200)
    if latent:
        choose(ui, bridge, "preview_method", "Latent2RGB-SD15")
    connect(ui, source, bridge)
    output = add_node(ui, "SaveLatent" if latent else "PreviewImage", 1220, 200)
    connect(ui, bridge, output)
    convert = add_node(ui, "MaskToImage", 220, 760)
    mask = add_node(ui, "PreviewImage", 720, 760)
    connect(ui, bridge, convert, source_slot=1)
    connect(ui, convert, mask)
    return source, bridge, output, mask


def latent_samples(ui, entry, node_id):
    meta, = entry["outputs"][node_id]["latents"]
    return load_safetensors(file_bytes(ui, meta))["latent_tensor"]


def widget_state(ui, ids):
    return ui.app.evaluate("""(app, ids) => Object.fromEntries(ids.map(id => {
        const node = app.graph.getNodeById(id);
        return [id, {type: node.type, widgets: node.widgets.map(w => ({name: w.name, value: w.value}))}];
    }))""", ids)


def reload_workflow(ui, artifacts, ids, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    before = widget_state(ui, ids)
    for state in before.values():
        assert [w["name"] for w in state["widgets"]] == WIDGET_NAMES[state["type"]]
    path = artifacts / "edited-workflow.json"
    export_workflow(ui, path)
    saved = json.loads(path.read_text())
    ui.new_workflow()
    ui.load_workflow(path, len(saved["nodes"]))
    assert widget_state(ui, ids) == before, "Widget names/order/values changed during native import"
    restored_path = artifacts / "reloaded-workflow.json"
    export_workflow(ui, restored_path)
    restored = json.loads(restored_path.read_text())
    saved_values = {str(n["id"]): n.get("widgets_values", []) for n in saved["nodes"]}
    restored_values = {str(n["id"]): n.get("widgets_values", []) for n in restored["nodes"]}
    assert restored_values == saved_values, "Serialized widget positions changed during native import"


@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_edit_mask_repeat_and_native_reload(ui, node_type, restore_by_name):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    choose(browser, bridge, "restore_mask", "if_same_size")
    initial = browser.run()
    original = preview(browser, initial, bridge).convert("RGB")
    latent = node_type == "PreviewBridgeLatent"
    samples = latent_samples(browser, initial, output) if latent else None
    previous = None
    for index in range(2):
        expected = edit_mask(browser, bridge, args.artifacts, f"edit-{index}", right=bool(index), existing=previous)
        if previous is not None:
            assert ImageChops.difference(previous, expected).getbbox(), "Second edit must replace the first mask"
        entry = browser.run()
        mask_output(browser, entry, mask, expected)
        edited_preview = preview(browser, entry, bridge)
        pixels_equal(ImageOps.invert(edited_preview.getchannel("A")), expected)
        if latent:
            np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)
        else:
            pixels_equal(preview(browser, entry, output).convert("RGB"), original)
        previous = expected
    reload_workflow(browser, args.artifacts, [bridge], restore_by_name)
    entry = browser.run()
    mask_output(browser, entry, mask, expected)
    if latent:
        np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)
    else:
        pixels_equal(preview(browser, entry, output).convert("RGB"), original)


@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
def test_clear_mask_then_refresh_keeps_it_empty(ui, node_type, restore):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    choose(browser, bridge, "restore_mask", restore)
    browser.run()
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    mask_output(browser, browser.run(), mask, painted)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    expect(browser.page.get_by_test_id("pointer-zone")).to_be_visible()
    browser.page.get_by_role("button", name=re.compile(r"Clear$")).click()
    empty = editor_mask(browser)
    assert not np.any(np.asarray(empty))
    browser.page.get_by_role("button", name=re.compile(r"Save$")).click()
    expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
    saved = browser.app.evaluate("""(app, id) => Object.fromEntries(
        new URL(app.graph.getNodeById(id).imgs[0].src).searchParams
    )""", bridge)
    with Image.open(io.BytesIO(file_bytes(browser, saved))) as image:
        pixels_equal(ImageOps.invert(image.getchannel("A")), empty)
    if node_type == "PreviewBridge":
        color(browser, source, 0xC08040)
    else:
        # Refresh the latent without changing its image dimensions.
        number(browser, source, "batch_size", 2)
    for _ in range(3):
        mask_output(browser, browser.run(), mask, empty)


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"), reason="Set IMPACT_E2E_SAM_IMAGE and install SAM ViT-B and the SD1.5 VAE")
@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("restore", ["never", "always", "if_same_size"])
def test_legacy_sam_temp_mask_reaches_bridge_and_survives_reload(ui, node_type, restore):
    browser, args = ui
    browser.timeout = 180
    browser.page.set_viewport_size({"width": 2200, "height": 1600})
    source = add_node(browser, "LoadImage", 220, 200)
    upload_image(browser, source, os.environ["IMPACT_E2E_SAM_IMAGE"])
    scale = add_node(browser, "ImageScale", 220, 650)
    number(browser, scale, "width", 0)
    number(browser, scale, "height", 512)
    choose(browser, scale, "upscale_method", "area")
    connect(browser, source, scale)
    bridge = add_node(browser, node_type, 720, 200)
    choose(browser, bridge, "restore_mask", restore)
    latent = node_type == "PreviewBridgeLatent"
    if latent:
        vae = add_node(browser, "VAELoader", 220, 1050)
        choose(browser, vae, "vae_name", "vae-ft-mse-840000-ema-pruned.safetensors")
        encode = add_node(browser, "VAEEncode", 720, 650)
        connect(browser, scale, encode)
        connect(browser, vae, encode, target_slot="vae")
        connect(browser, encode, bridge)
        connect(browser, vae, bridge, target_slot="vae_opt")
    else:
        connect(browser, scale, bridge)
    output = add_node(browser, "SaveLatent" if latent else "PreviewImage", 1220, 200)
    connect(browser, bridge, output)
    convert = add_node(browser, "MaskToImage", 720, 1050)
    mask = add_node(browser, "PreviewImage", 1220, 1050)
    connect(browser, bridge, convert, source_slot=1)
    connect(browser, convert, mask)
    bridge_ids = [bridge]
    if latent:
        second = add_node(browser, "PreviewBridgeLatent", 1700, 200)
        bridge_ids.append(second)
        connect(browser, bridge, second)
        connect(browser, vae, second, target_slot="vae_opt")
        connect(browser, second, output)
        second_convert = add_node(browser, "MaskToImage", 1700, 650)
        second_mask = add_node(browser, "PreviewImage", 1700, 1050)
        connect(browser, second, second_convert, source_slot=1)
        connect(browser, second_convert, second_mask)
    initial = browser.run()
    original = preview(browser, initial, bridge).convert("RGB")
    samples = latent_samples(browser, initial, output) if latent else None
    responses = []

    def capture(route):
        response = route.fetch(timeout=180000)
        responses.append(response.body())
        route.fulfill(response=response)

    browser.page.route("**/sam/detect", capture)
    for index in range(2):
        context_menu(browser, bridge, "Open in SAM Detector")
        canvas = browser.page.locator("#pointsCanvas")
        expect(canvas).to_be_visible()
        save = browser.page.get_by_role("button", name="Save to node", exact=True)
        expect(save).to_be_enabled()
        box = canvas.bounding_box()
        browser.page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * (0.5 if index == 0 else 0.4))
        browser.page.mouse.click(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.1, button="right")
        with browser.page.expect_response(lambda response: response.url.endswith("/sam/detect"), timeout=180000) as detected:
            browser.page.get_by_role("button", name="Detect", exact=True).click()
        assert detected.value.ok
        with Image.open(io.BytesIO(responses[-1])) as image:
            expected = image.convert("L")
        assert expected.getextrema() == (0, 255)
        browser.page.wait_for_function("""() => {
            const canvas = document.getElementById('samEditorMaskCanvas');
            return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
                .some((value, index) => index % 4 === 0 && value > 0);
        }""")
        browser.page.screenshot(path=str(args.artifacts / f"legacy-sam-{index}.png"))
        with browser.page.expect_response(lambda response: response.url.endswith("/upload/mask")) as uploaded:
            save.click()
        assert uploaded.value.ok
        assert uploaded.value.json()["type"] == "temp"
        expect(canvas).not_to_be_visible()
        entry = browser.run()
        mask_output(browser, entry, mask, expected)
        pixels_equal(ImageOps.invert(preview(browser, entry, bridge).getchannel("A")), expected)
        if latent:
            mask_output(browser, entry, second_mask, expected)
            np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)
        else:
            pixels_equal(preview(browser, entry, output).convert("RGB"), original)
    reload_workflow(browser, args.artifacts, bridge_ids, latent)
    entry = browser.run()
    mask_output(browser, entry, mask, expected)
    if latent:
        mask_output(browser, entry, second_mask, expected)


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_a_workflow_with_the_image_object_of_the_first_bridge_versions_runs(ui, restore_by_name):
    """The first versions of the bridge saved its image as an object. Later ones read it as a bridge without an image."""
    browser, args = ui
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    current = args.artifacts / "current.json"
    export_workflow(browser, current)
    saved = json.loads(current.read_text())
    for node in saved["nodes"]:
        node.pop("widgets_values_named", None)  # a workflow of that time has the values by position only
    node, = [n for n in saved["nodes"] if n["type"] == "PreviewBridge"]
    values = node["widgets_values"]
    values[0] = {"filename": "clipspace-mask-470361.200000003.png", "subfolder": "clipspace", "type": "input", "image_hash": -9057483542318295000,
                 "forward_filename": "ComfyUI_temp_oimqn_00002_.png", "forward_subfolder": "", "forward_type": "temp"}
    legacy = args.artifacts / "image-object.json"
    legacy.write_text(json.dumps(saved))
    browser.new_workflow()
    browser.load_workflow(legacy, len(saved["nodes"]))

    entry = browser.run()

    preview(browser, entry, bridge)
    no_mask(browser, entry, mask)
    state = widget_state(browser, [bridge])[bridge]
    assert [w["name"] for w in state["widgets"]] == WIDGET_NAMES["PreviewBridge"]
    assert [w["value"] for w in state["widgets"][1:len(values)]] == values[1:], "The other saved values of the bridge stay as saved"
    assert isinstance(state["widgets"][0]["value"], str), "The run gives the bridge its image"


def test_a_bridge_that_was_not_executed_explains_itself_instead_of_opening_an_editor(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    browser.page.route("**/sam/prepare", lambda route: route.fulfill(status=200))
    errors = []
    browser.page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    notice = browser.page.get_by_test_id("toast").get_by_text(re.compile(r"the node itself (is|has been) executed"))
    editor, detector = browser.page.get_by_test_id("mask-editor-root"), browser.page.locator("#pointsCanvas")

    def toolbar(name):
        browser.page.mouse.click(*point(browser, bridge, menu=True))
        browser.page.get_by_role("button", name=name, exact=True).click()

    ways = {"menu": lambda: context_menu(browser, bridge, re.compile(r"^Open in MaskEditor")),
            "toolbar": lambda: toolbar("Open Mask Editor for Selected Node"),
            "sam": lambda: context_menu(browser, bridge, "Open in SAM Detector")}

    def explained(state):
        for name, open_it in ways.items():
            open_it()
            expect(notice).to_be_visible()
            expect(editor).not_to_be_visible()
            expect(detector).not_to_be_visible()
            browser.page.screenshot(path=str(args.artifacts / f"{state}-{name}.png"))
            expect(notice).not_to_be_visible(timeout=15000)

    explained("not-run")
    # A run of the workflow is not an execution of the bridge: bypassed, the run passes it by and it stays empty.
    toolbar("Bypass/Unbypass Selected Nodes")
    assert bridge not in browser.run()["outputs"]
    explained("run-bypassed")
    assert not [error for error in errors if "MaskEditor" in error]
    toolbar("Bypass/Unbypass Selected Nodes")
    assert bridge in browser.run()["outputs"]
    for name, open_it in ways.items():
        open_it()
        opened = detector if name == "sam" else editor
        expect(opened).to_be_visible()
        expect(notice).not_to_be_visible()
        browser.page.screenshot(path=str(args.artifacts / f"executed-{name}.png"))
        browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()
        expect(opened).not_to_be_visible()


def test_image_clipspace_feedback_for_repeated_edits(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    invert = add_node(browser, "ImageInvert", 1700, 200)
    final = add_node(browser, "PreviewImage", 1700, 760)
    connect(browser, bridge, invert)
    connect(browser, invert, final)
    entry = browser.run()
    expected = preview(browser, entry, bridge).convert("RGB")
    for index in range(2):
        copied = preview(browser, entry, final).convert("RGB")
        pixels_equal(copied, ImageOps.invert(expected), tolerance=1)
        context_menu(browser, final, "Copy (Clipspace)")
        context_menu(browser, bridge, "Paste (Clipspace)")
        entry = browser.run()
        pixels_equal(preview(browser, entry, bridge).convert("RGB"), copied)
        pixels_equal(preview(browser, entry, output).convert("RGB"), copied)
        expected_mask = edit_mask(browser, bridge, args.artifacts, f"pasted-{index}", right=bool(index))
        entry = browser.run()
        mask_output(browser, entry, mask, expected_mask)
        pixels_equal(preview(browser, entry, output).convert("RGB"), copied)
        expected = copied


def test_vue_paint_and_mask_survive_native_reload(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    original = preview(browser, browser.run(), bridge).convert("RGB")
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    expect(browser.page.get_by_test_id("mask-editor-root")).to_be_visible()
    zone = browser.page.get_by_test_id("pointer-zone")
    box = zone.bounding_box()
    for tool, x in (("pen", 0.25), ("rgbPaint", 0.6)):
        browser.page.locator(f'[data-tool="{tool}"]').click()
        browser.page.mouse.move(box["x"] + box["width"] * x, box["y"] + box["height"] * 0.45)
        browser.page.mouse.down()
        browser.page.mouse.move(box["x"] + box["width"] * (x + 0.15), box["y"] + box["height"] * 0.55, steps=12)
        browser.page.mouse.up()
    expected_image, expected_mask = editor_image(browser), editor_mask(browser)
    assert ImageChops.difference(original, expected_image).getbbox(), "Paint stroke did not change the image"
    assert np.count_nonzero(np.asarray(expected_mask)), "Mask stroke is empty"
    expected_image.save(args.artifacts / "painted-expected-image.png")
    expected_mask.save(args.artifacts / "painted-expected-mask.png")
    browser.page.screenshot(path=str(args.artifacts / "painted-editor.png"))
    browser.page.get_by_role("button", name=re.compile(r"Save$")).click()
    expect(zone).not_to_be_visible()
    entry = browser.run()
    pixels_equal(preview(browser, entry, output).convert("RGB"), expected_image, tolerance=1)
    mask_output(browser, entry, mask, expected_mask)
    reload_workflow(browser, args.artifacts, [bridge], False)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    expect(zone).to_be_visible()
    pixels_equal(editor_image(browser), expected_image, tolerance=1)
    pixels_equal(editor_mask(browser), expected_mask, tolerance=1)
    browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()


def test_image_mask_controls_downstream_composite(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    fill = add_node(browser, "EmptyImage", 1700, 200)
    number(browser, fill, "width", 256)
    number(browser, fill, "height", 256)
    color(browser, fill, 0xE02060)
    composite = add_node(browser, "ImageCompositeMasked", 1220, 760)
    final = add_node(browser, "PreviewImage", 1700, 760)
    connect(browser, bridge, composite)
    connect(browser, fill, composite, target_slot=1)
    connect(browser, bridge, composite, source_slot=1, target_slot="mask")
    connect(browser, composite, final)
    entry = browser.run()
    original = preview(browser, entry, output).convert("RGB")
    pixels_equal(preview(browser, entry, final).convert("RGB"), original)
    preview(browser, entry, bridge)
    for index in range(2):
        expected_mask = edit_mask(browser, bridge, args.artifacts, f"composite-{index}", right=bool(index))
        entry = browser.run()
        mask_output(browser, entry, mask, expected_mask)
        expected = Image.composite(Image.new("RGB", original.size, (224, 32, 96)), original, expected_mask)
        actual = preview(browser, entry, final).convert("RGB")
        pixels_equal(actual, expected, tolerance=1)
        assert ImageChops.difference(actual, original).getbbox(), "Masked consumer did not change the image"
        outside = np.asarray(expected_mask) == 0
        np.testing.assert_array_equal(np.asarray(actual)[outside], np.asarray(original)[outside])


def test_latent_mask_is_visible_and_editable_in_second_bridge(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridgeLatent")
    second = add_node(browser, "PreviewBridgeLatent", 1700, 200)
    choose(browser, second, "preview_method", "Latent2RGB-SD15")
    connect(browser, bridge, second)
    connect(browser, second, output)
    convert = add_node(browser, "MaskToImage", 1220, 760)
    second_mask = add_node(browser, "PreviewImage", 1700, 760)
    connect(browser, second, convert, source_slot=1)
    connect(browser, convert, second_mask)
    initial = browser.run()
    samples = latent_samples(browser, initial, output)
    preview(browser, initial, bridge)
    first = edit_mask(browser, bridge, args.artifacts, "first-latent")
    entry = browser.run()
    mask_output(browser, entry, mask, first)
    mask_output(browser, entry, second_mask, first)
    pixels_equal(ImageOps.invert(preview(browser, entry, second).getchannel("A")), first, tolerance=1)
    edited = edit_mask(browser, second, args.artifacts, "second-latent", right=True, existing=first)
    assert ImageChops.difference(first, edited).getbbox()
    entry = browser.run()
    mask_output(browser, entry, mask, first)
    mask_output(browser, entry, second_mask, edited)
    np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)


@pytest.mark.parametrize("node_type", BRIDGES)
def test_block_waits_for_mask_and_then_allows_downstream(ui, node_type):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    browser.page.mouse.click(*point(browser, bridge, widget="block"))
    entry = browser.run()
    preview(browser, entry, bridge)
    assert output not in entry["outputs"] and mask not in entry["outputs"], "Unmasked bridge did not block its consumers"
    expected = edit_mask(browser, bridge, args.artifacts, "unblock")
    entry = browser.run()
    assert output in entry["outputs"], "Edited bridge did not resume its consumers"
    mask_output(browser, entry, mask, expected)


@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("selection", ["clear", "full"])
def test_saved_mask_selection_controls_blocking(ui, node_type, selection):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    choose(browser, bridge, "restore_mask", "always")
    browser.page.mouse.click(*point(browser, bridge, widget="block"))
    preview(browser, browser.run(), bridge)
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    mask_output(browser, browser.run(), mask, painted)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    wait_editor_canvas(browser, painted.size)
    browser.page.get_by_role("button", name=re.compile(r"Clear$")).click()
    if selection == "full":
        browser.page.get_by_role("button", name=re.compile(r"Invert$")).click()
    expected = editor_mask(browser)
    assert expected.getextrema() == ((0, 0) if selection == "clear" else (255, 255))
    expected.save(args.artifacts / "saved-selection.png")
    browser.page.get_by_role("button", name=re.compile(r"Save$")).click()
    expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
    observed = []
    for stage in ["same-input", "changed-input", "native-reload"]:
        if stage == "changed-input":
            if node_type == "PreviewBridge":
                color(browser, source, 0xC08040)
            else:
                number(browser, source, "batch_size", 2)
        elif stage == "native-reload":
            reload_workflow(browser, args.artifacts, [bridge], selection == "full")
        entry = browser.run()
        preview(browser, entry, bridge)
        record = {"stage": stage, "prompt_id": entry["prompt"][1],
                  "submitted_image": entry["prompt"][2][bridge]["inputs"]["image"],
                  "consumer_ran": output in entry["outputs"], "mask_ran": mask in entry["outputs"]}
        observed.append(record)
        browser.page.screenshot(path=str(args.artifacts / f"{stage}.png"))
        (args.artifacts / "queues.json").write_text(json.dumps(observed, indent=2))
        assert record["consumer_ran"] == record["mask_ran"] == (selection == "full"), record
        if selection == "full":
            mask_output(browser, entry, mask, expected)
    assert len({row["prompt_id"] for row in observed}) == 3


@pytest.mark.parametrize("restore", ["never", "always", "if_same_size"])
@pytest.mark.parametrize("resize", [False, True], ids=["same_size", "new_size"])
def test_new_image_refreshes_preview_and_obeys_restore_mask(ui, restore, resize):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    choose(browser, bridge, "restore_mask", restore)
    preview(browser, browser.run(), bridge)
    expected = edit_mask(browser, bridge, args.artifacts, "before-refresh")
    mask_output(browser, browser.run(), mask, expected)
    color(browser, source, 0xC08040)
    if resize:
        number(browser, source, "width", 320)
    entry = browser.run()
    size = (320 if resize else 256, 256)
    pixels_equal(preview(browser, entry, output).convert("RGB"), Image.new("RGB", size, (192, 128, 64)))
    refreshed = preview(browser, entry, bridge)
    restore_expected = restore == "always" or (restore == "if_same_size" and not resize)
    if restore_expected:
        mask_output(browser, entry, mask, expected)
        assert np.count_nonzero(np.asarray(ImageOps.invert(refreshed.getchannel("A")))) > 0
    else:
        assert preview(browser, entry, mask).convert("L").getextrema() == (0, 0)
        assert refreshed.getchannel("A").getextrema() == (255, 255)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    wait_editor_canvas(browser, size)
    pixels_equal(editor_image(browser), Image.new("RGB", size, (192, 128, 64)))
    browser.page.screenshot(path=str(args.artifacts / "reopened-after-upstream-change.png"))
    browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()


@pytest.mark.parametrize("restore", ["always", "if_same_size"])
def test_restored_mask_stays_out_of_the_rgba_image_other_nodes_read(ui, restore):
    browser, args = ui
    source = add_node(browser, "EmptyImage", 150, 150)
    number(browser, source, "width", 256)
    number(browser, source, "height", 256)
    alpha = add_node(browser, "SolidMask", 150, 700)
    number(browser, alpha, "value", 0.25)
    number(browser, alpha, "width", 256)
    number(browser, alpha, "height", 256)
    join = add_node(browser, "JoinImageWithAlpha", 560, 150)
    connect(browser, source, join)
    connect(browser, alpha, join, target_slot="alpha")
    bridge = add_node(browser, "PreviewBridge", 970, 150)
    choose(browser, bridge, "restore_mask", restore)
    connect(browser, join, bridge)
    convert = add_node(browser, "MaskToImage", 1380, 150)
    bridge_mask = add_node(browser, "PreviewImage", 1790, 150)
    connect(browser, bridge, convert, source_slot=1)
    connect(browser, convert, bridge_mask)

    def split_mask(x, y):
        split = add_node(browser, "SplitImageWithAlpha", x, y)
        convert = add_node(browser, "MaskToImage", x + 410, y)
        output = add_node(browser, "PreviewImage", x + 820, y)
        connect(browser, join, split)
        connect(browser, split, convert, source_slot=1)
        connect(browser, convert, output)
        return output

    first = split_mask(560, 700)
    preview(browser, browser.run(), bridge)
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    mask_output(browser, browser.run(), bridge_mask, painted)
    number(browser, alpha, "value", 0.75)
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, painted)
    joined = Image.new("L", (256, 256), 191)
    mask_output(browser, entry, first, joined)
    # A consumer added afterwards reads the Join output from the execution cache.
    late = split_mask(150, 900)
    mask_output(browser, browser.run(), late, joined)


def build_rgba_bridge(ui, restore):
    """EmptyImage + SolidMask -> Join -> PreviewBridge, with previews of the bridge's mask and of its image's alpha."""
    source = add_node(ui, "EmptyImage", 150, 150)
    number(ui, source, "width", 256)
    number(ui, source, "height", 256)
    alpha = add_node(ui, "SolidMask", 150, 700)
    number(ui, alpha, "value", 0.25)
    number(ui, alpha, "width", 256)
    number(ui, alpha, "height", 256)
    join = add_node(ui, "JoinImageWithAlpha", 560, 150)
    connect(ui, source, join)
    connect(ui, alpha, join, target_slot="alpha")
    bridge = add_node(ui, "PreviewBridge", 970, 150)
    choose(ui, bridge, "restore_mask", restore)
    connect(ui, join, bridge)
    convert = add_node(ui, "MaskToImage", 1380, 150)
    bridge_mask = add_node(ui, "PreviewImage", 1790, 150)
    connect(ui, bridge, convert, source_slot=1)
    connect(ui, convert, bridge_mask)
    split = add_node(ui, "SplitImageWithAlpha", 560, 700)
    convert = add_node(ui, "MaskToImage", 970, 700)
    image_alpha = add_node(ui, "PreviewImage", 1380, 700)
    connect(ui, bridge, split)
    connect(ui, split, convert, source_slot=1)
    connect(ui, convert, image_alpha)
    return alpha, bridge, bridge_mask, image_alpha


def switch(ui, node_id, name, value):
    ui.page.mouse.click(*point(ui, node_id, widget=name))
    ui.page.wait_for_function("""({app, id, name, value}) =>
        app.graph.getNodeById(id).widgets.find(w => w.name === name).value === value
    """, arg={"app": ui.app, "id": node_id, "name": name, "value": value})


def level(value):
    return Image.new("L", (256, 256), value)


def over_checkerboard(ui, entry, node_id, opacity):
    """The bridge's file shows a black picture of this opacity over the two colours of the checkerboard."""
    shown = preview(ui, entry, node_id).convert("L")
    for position, board in (((0, 0), 255), ((16, 0), 204), ((16, 16), 255)):
        assert abs(shown.getpixel(position) - board * (1 - opacity)) <= 1


def no_mask(ui, entry, node_id):
    # An empty mask is returned as 64 x 64 whatever the image size.
    assert preview(ui, entry, node_id).convert("L").getextrema() == (0, 0)


@pytest.mark.parametrize("restore", ["never", "always"])
def test_rgba_alpha_stays_in_the_image_while_alpha_as_mask_is_disabled(ui, restore):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, "never")
    image = add_node(browser, "PreviewImage", 1790, 700)
    connect(browser, bridge, image)
    black = ((0, 0),) * 3
    # The server keeps the last mask under the node's id, across workflows, and "always" would restore one that a
    # case before this one left. A run with "never" drops it.
    browser.run()
    choose(browser, bridge, "restore_mask", restore)
    # The bridge runs again on the image it already has: nothing was painted, so there is still no mask.
    for _ in range(2):
        entry = browser.run()
        assert preview(browser, entry, bridge).getchannel("A").getextrema() == (255, 255)
        # The transparency is drawn into the bridge's file and is not in the picture it passes on.
        over_checkerboard(browser, entry, bridge, 0.75)
        assert preview(browser, entry, image).convert("RGB").getextrema() == black
        no_mask(browser, entry, bridge_mask)
        mask_output(browser, entry, image_alpha, level(64))
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    wait_editor_canvas(browser, (256, 256))
    assert editor_mask(browser).getextrema() == (0, 0)
    browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, painted)
    mask_output(browser, entry, image_alpha, level(64))
    assert preview(browser, entry, image).convert("RGB").getextrema() == black
    number(browser, alpha, "value", 0.75)
    entry = browser.run()
    if restore == "always":
        mask_output(browser, entry, bridge_mask, painted)
    else:
        no_mask(browser, entry, bridge_mask)
        over_checkerboard(browser, entry, bridge, 0.25)
    mask_output(browser, entry, image_alpha, level(191))
    assert preview(browser, entry, image).convert("RGB").getextrema() == black


@pytest.mark.parametrize("restore,restore_by_name", [("never", False), ("always", True)], ids=["never-positional", "always-named"])
def test_rgba_alpha_is_the_mask_while_alpha_as_mask_is_enabled(ui, restore, restore_by_name):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, restore)
    switch(browser, bridge, "alpha_as_mask", True)
    for _ in range(2):
        entry = browser.run()
        pixels_equal(preview(browser, entry, bridge).getchannel("A"), level(191), tolerance=1)
        mask_output(browser, entry, bridge_mask, level(64))
        mask_output(browser, entry, image_alpha, level(64))
    # The editor opens with the alpha as its mask; the edit replaces it in the mask and in the image's alpha.
    painted = edit_mask(browser, bridge, args.artifacts, "painted", existing=level(64))
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, painted)
    mask_output(browser, entry, image_alpha, painted)
    reload_workflow(browser, args.artifacts, [bridge], restore_by_name)
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, painted)
    mask_output(browser, entry, image_alpha, painted)
    # A new image brings its own alpha, and nothing is restored over it.
    number(browser, alpha, "value", 0.75)
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, level(191))
    mask_output(browser, entry, image_alpha, level(191))


def test_switching_alpha_as_mask_rereads_the_alpha_and_keeps_an_edit(ui):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, "always")
    switch(browser, bridge, "alpha_as_mask", True)
    mask_output(browser, browser.run(), bridge_mask, level(64))
    # Off: a mask that is still the alpha is dropped, also on the run after that one.
    switch(browser, bridge, "alpha_as_mask", False)
    warning = browser.page.get_by_test_id("toast").get_by_text("alpha_as_mask was switched off")
    for _ in range(2):
        entry = browser.run()
        no_mask(browser, entry, bridge_mask)
        mask_output(browser, entry, image_alpha, level(64))
    expect(warning).to_have_count(0)
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    mask_output(browser, browser.run(), bridge_mask, painted)
    # On: the alpha of the image is read at once.
    switch(browser, bridge, "alpha_as_mask", True)
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, level(64))
    mask_output(browser, entry, image_alpha, level(64))
    # Off after an edit: the edited mask is kept and the image gets its own alpha back.
    painted = edit_mask(browser, bridge, args.artifacts, "edited", existing=level(64))
    mask_output(browser, browser.run(), bridge_mask, painted)
    switch(browser, bridge, "alpha_as_mask", False)
    entry = browser.run()
    mask_output(browser, entry, bridge_mask, painted)
    mask_output(browser, entry, image_alpha, level(64))
    # The bridge's file is written again in the form the editor gets while the option is off.
    over_checkerboard(browser, entry, bridge, 0.75)
    # The kept mask lies on transparent pixels of the image: the user is told that it was kept.
    expect(warning).to_be_visible()
    browser.page.screenshot(path=str(args.artifacts / "warning.png"))


def shown_on_node(ui, node_id, path):
    """The picture the node displays once it comes from this path, as the server returns it."""
    ui.page.wait_for_function("""({app, id, path}) => {
        const image = app.graph.getNodeById(id).imgs?.[0];
        return image?.complete && image.naturalWidth && new URL(image.src).pathname.endsWith(path);
    }""", arg={"app": ui.app, "id": node_id, "path": path})
    source = ui.app.evaluate("(app, id) => app.graph.getNodeById(id).imgs[0].src", node_id)
    response = ui.page.request.get(source)
    assert response.ok
    return source, Image.open(io.BytesIO(response.body())).convert("RGBA")


def test_switching_alpha_as_mask_shows_the_picture_of_the_other_setting_at_once(ui):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, "never")
    entry = browser.run()
    over_checkerboard(browser, entry, bridge, 0.75)
    ran = entry["outputs"][bridge]["images"][0]["filename"]
    runs = len(browser.page.request.get(f"{browser.base_url}/history").json())
    # On: the picture with its alpha, where the run showed it over the checkerboard.
    switch(browser, bridge, "alpha_as_mask", True)
    _, shown = shown_on_node(browser, bridge, "/impact/view/pb_alpha_preview")
    pixels_equal(shown.getchannel("A"), level(191), tolerance=1)
    browser.page.screenshot(path=str(args.artifacts / "switched-on.png"))
    # Back: the file of the run itself.
    switch(browser, bridge, "alpha_as_mask", False)
    source, _ = shown_on_node(browser, bridge, "/view")
    assert ran in source
    # Nothing ran for it, and the run that follows shows the same picture.
    switch(browser, bridge, "alpha_as_mask", True)
    _, shown = shown_on_node(browser, bridge, "/impact/view/pb_alpha_preview")
    assert len(browser.page.request.get(f"{browser.base_url}/history").json()) == runs
    entry = browser.run()
    assert preview(browser, entry, bridge).convert("RGBA").tobytes() == shown.tobytes()
    # Off: over the checkerboard again, as the run that follows shows it.
    switch(browser, bridge, "alpha_as_mask", False)
    _, shown = shown_on_node(browser, bridge, "/impact/view/pb_alpha_preview")
    assert shown.getchannel("A").getextrema() == (255, 255)
    for position, board in (((0, 0), 255), ((16, 0), 204), ((16, 16), 255)):
        assert abs(shown.convert("L").getpixel(position) - board * 0.25) <= 1
    browser.page.screenshot(path=str(args.artifacts / "switched-off.png"))
    entry = browser.run()
    assert preview(browser, entry, bridge).convert("RGBA").tobytes() == shown.tobytes()


def test_the_sam_detector_opens_the_file_of_the_run_after_alpha_as_mask_is_switched(ui):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, "never")
    ran = browser.run()["outputs"][bridge]["images"][0]
    prepared, viewed = [], []
    browser.page.route("**/sam/prepare", lambda route: (prepared.append(route.request.post_data_json), route.fulfill(status=200)))
    browser.page.on("request", lambda request: viewed.append(parse_qs(urlsplit(request.url).query)) if urlsplit(request.url).path.endswith("/view") else None)
    switch(browser, bridge, "alpha_as_mask", True)
    shown_on_node(browser, bridge, "/impact/view/pb_alpha_preview")
    viewed.clear()
    context_menu(browser, bridge, "Open in SAM Detector")
    expect(browser.page.locator("#pointsCanvas")).to_be_visible()
    expect(browser.page.get_by_role("button", name="Save to node", exact=True)).to_be_enabled()
    browser.page.screenshot(path=str(args.artifacts / "sam-after-switch.png"))
    # The detector loads and shows the file the bridge holds, not the picture the node shows meanwhile.
    file = {key: ran[key] for key in ("filename", "type", "subfolder")}
    assert prepared == [{"sam_model_name": "auto", **file}]
    assert [{key: query[key][0] for key in file} | {"channel": query["channel"]} for query in viewed] == [file | {"channel": ["rgb"]}]
    browser.page.get_by_role("button", name="Cancel", exact=True).click()
    expect(browser.page.locator("#pointsCanvas")).not_to_be_visible()


def test_the_sam_detector_opens_the_picture_a_receiver_shows_after_a_mask_was_saved_on_it(ui):
    """ImageReceiver shows what the sender sent last, while its record of files stays at a mask saved on it earlier."""
    browser, args = ui
    load = add_node(browser, "LoadImage", 150, 150)
    sender = add_node(browser, "ImageSender", 700, 150)
    connect(browser, load, sender)
    receiver = add_node(browser, "ImageReceiver", 150, 750)
    output = add_node(browser, "PreviewImage", 800, 750)
    connect(browser, receiver, output)
    prepared = []
    browser.page.route("**/sam/prepare", lambda route: (prepared.append(route.request.post_data_json), route.fulfill(status=200)))
    # The detector's answer is a mask picture; no model is run here.
    found = io.BytesIO()
    ImageOps.expand(Image.new("L", (128, 128), 255), border=64).save(found, "PNG")
    browser.page.route("**/sam/detect", lambda route: route.fulfill(status=200, content_type="image/png", body=found.getvalue()))

    def shown(after):
        """The file the receiver displays, once it is another one than `after`."""
        return browser.page.wait_for_function("""({app, id, after}) => {
            const image = app.graph.getNodeById(id).imgs?.[0];
            const name = image?.naturalWidth ? new URL(image.src).searchParams.get('filename') : null;
            return name && name !== after ? name : false;
        }""", arg={"app": browser.app, "id": receiver, "after": after}).json_value()

    def send(colour, after):
        path = args.artifacts / f"picture-{len(prepared)}-{colour[0]}.png"
        Image.new("RGB", (256, 256), colour).save(path)
        upload_image(browser, load, path)
        browser.run()
        return shown(after)

    first = send((200, 60, 40), None)
    context_menu(browser, receiver, "Open in SAM Detector")
    save = browser.page.get_by_role("button", name="Save to node", exact=True)
    expect(save).to_be_enabled()
    assert prepared[-1]["filename"] == first
    box = browser.page.locator("#pointsCanvas").bounding_box()
    browser.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    with browser.page.expect_response(lambda response: response.url.endswith("/sam/detect")):
        browser.page.get_by_role("button", name="Detect", exact=True).click()
    # Until the answer is drawn there is no mask to save.
    browser.page.wait_for_function("""() => {
        const canvas = document.getElementById('samEditorMaskCanvas');
        return canvas.width > 0 && canvas.getContext('2d').getImageData(canvas.width / 2, canvas.height / 2, 1, 1).data[3] > 0;
    }""")
    with browser.page.expect_response(lambda response: response.url.endswith("/upload/mask")):
        save.click()
    expect(browser.page.locator("#pointsCanvas")).not_to_be_visible()
    saved = shown(first)
    second = send((40, 90, 200), saved)
    context_menu(browser, receiver, "Open in SAM Detector")
    expect(save).to_be_enabled()
    browser.page.screenshot(path=str(args.artifacts / "sam-on-receiver.png"))
    assert prepared[-1]["filename"] == second
    browser.page.get_by_role("button", name="Cancel", exact=True).click()
    expect(browser.page.locator("#pointsCanvas")).not_to_be_visible()


def test_rgba_paint_of_the_editor_is_laid_over_the_input_while_alpha_as_mask_is_disabled(ui):
    browser, args = ui
    alpha, bridge, bridge_mask, image_alpha = build_rgba_bridge(browser, "never")
    image = add_node(browser, "PreviewImage", 1790, 700)
    connect(browser, bridge, image)
    given = preview(browser, browser.run(), image)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    zone = browser.page.get_by_test_id("pointer-zone")
    expect(zone).to_be_visible()
    box = zone.bounding_box()
    browser.page.locator('[data-tool="rgbPaint"]').click()
    browser.page.mouse.move(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.45)
    browser.page.mouse.down()
    browser.page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.55, steps=12)
    browser.page.mouse.up()
    encoded = browser.page.evaluate("""() => document.querySelectorAll(
        '#maskEditorCanvasContainer canvas')[1].toDataURL('image/png').split(',')[1]
    """)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as layer:
        paint = layer.convert("RGBA")
    levels = paint.getchannel("A").getextrema()
    assert levels[0] == 0 and levels[1] > 0, "Paint stroke is empty"
    browser.page.screenshot(path=str(args.artifacts / "painted-editor.png"))
    browser.page.get_by_role("button", name=re.compile(r"Save$")).click()
    expect(zone).not_to_be_visible()
    for _ in range(2):
        entry = browser.run()
        # The paint lies over the half-transparent picture; nothing of the checkerboard the editor showed is in it.
        pixels_equal(preview(browser, entry, image), Image.alpha_composite(given, paint), tolerance=1)
        no_mask(browser, entry, bridge_mask)


def test_the_selected_image_of_an_rgba_batch_goes_to_the_editor(ui):
    browser, args = ui
    solid = add_node(browser, "SolidMask", 150, 750)
    number(browser, solid, "value", 0.25)
    number(browser, solid, "width", 256)
    number(browser, solid, "height", 256)
    joins = []
    for row, value in enumerate((0x4080C0, 0xC08040)):
        source = add_node(browser, "EmptyImage", 150, 100 + 320 * row)
        number(browser, source, "width", 256)
        number(browser, source, "height", 256)
        color(browser, source, value)
        join = add_node(browser, "JoinImageWithAlpha", 560, 100 + 320 * row)
        connect(browser, source, join)
        connect(browser, solid, join, target_slot="alpha")
        joins.append(join)
    batch = add_node(browser, "ImpactMakeImageBatch", 900, 150)
    connect(browser, joins[0], batch)
    connect(browser, joins[1], batch, target_slot=1)
    bridge = add_node(browser, "PreviewBridge", 1250, 150)
    connect(browser, batch, bridge)
    image = add_node(browser, "PreviewImage", 1700, 150)
    connect(browser, bridge, image)
    convert = add_node(browser, "MaskToImage", 1250, 800)
    bridge_mask = add_node(browser, "PreviewImage", 1700, 800)
    connect(browser, bridge, convert, source_slot=1)
    connect(browser, convert, bridge_mask)
    entry = browser.run()
    browser.images(entry)
    files = [Image.open(io.BytesIO(file_bytes(browser, meta))).convert("RGBA") for meta in entry["outputs"][bridge]["images"]]
    passed = [Image.open(io.BytesIO(file_bytes(browser, meta))).convert("RGBA") for meta in entry["outputs"][image]["images"]]
    assert len(files) == len(passed) == 2
    # Both images are shown over the checkerboard, and both are passed on as they came.
    for shown, given, value in zip(files, passed, ((0x40, 0x80, 0xC0), (0xC0, 0x80, 0x40))):
        assert given.getpixel((0, 0)) == value + (191,)
        assert all(abs(a - round(b * 0.75 + 255 * 0.25)) <= 1 for a, b in zip(shown.getpixel((0, 0)), value))
        assert all(abs(a - round(b * 0.75 + 204 * 0.25)) <= 1 for a, b in zip(shown.getpixel((16, 0)), value))
    # Select the second image on the node, then open the editor on it.
    x, y = browser.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        const [x, y, width, height] = node.imageRects[1];
        return app.canvasPosToClientPos([node.pos[0] + x + width / 2, node.pos[1] + y + height / 2]);
    }""", bridge)
    # The thumbnail under the pointer is the one a click selects.
    browser.page.mouse.move(x - 5, y - 5)
    browser.page.mouse.move(x, y, steps=5)
    browser.page.mouse.down()
    # The node takes the press when it is drawn with the button held.
    browser.page.wait_for_function("({app, id}) => app.graph.getNodeById(id).pointerDown?.index === 1", arg={"app": browser.app, "id": bridge})
    browser.page.mouse.up()
    browser.page.wait_for_function("({app, id}) => app.graph.getNodeById(id).imageIndex === 1", arg={"app": browser.app, "id": bridge})
    second = entry["outputs"][bridge]["images"][1]
    assert widget_state(browser, [bridge])[bridge]["widgets"][0]["value"].startswith(f'{second["subfolder"]}/{second["filename"]}')
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    wait_editor_canvas(browser, (256, 256))
    pixels_equal(editor_image(browser), files[1].convert("RGB"), tolerance=1)
    browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()
    painted = edit_mask(browser, bridge, args.artifacts, "painted")
    entry = browser.run()
    pixels_equal(preview(browser, entry, image), passed[1])
    mask_output(browser, entry, bridge_mask, painted)


@pytest.mark.parametrize("restore,restored", [("always", True), ("if_same_size", False)],
                         ids=["restored_after_resize", "discarded_after_resize"])
def test_restored_latent_mask_reaches_the_next_bridge(ui, restore, restored):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridgeLatent")
    second = add_node(browser, "PreviewBridgeLatent", 1700, 200)
    choose(browser, second, "preview_method", "Latent2RGB-SD15")
    connect(browser, bridge, second)
    connect(browser, second, output)
    convert = add_node(browser, "MaskToImage", 1220, 1050)
    second_mask = add_node(browser, "PreviewImage", 1700, 760)
    connect(browser, second, convert, source_slot=1)
    connect(browser, convert, second_mask)
    choose(browser, bridge, "restore_mask", restore)
    preview(browser, browser.run(), bridge)
    expected = edit_mask(browser, bridge, args.artifacts, "before-latent-refresh")
    entry = browser.run()
    mask_output(browser, entry, second_mask, expected)
    number(browser, source, "width", 320)
    entry = browser.run()
    samples = latent_samples(browser, entry, output)
    assert samples.shape == (1, 4, 32, 40)
    assert not np.any(samples), "Bridge changed latent samples while restoring a mask"
    if restored:
        mask_output(browser, entry, mask, expected)
        mask_output(browser, entry, second_mask, expected)
    else:
        for node_id in (mask, second_mask):
            assert preview(browser, entry, node_id).convert("L").getextrema() == (255, 255)
        assert preview(browser, entry, second).getchannel("A").getextrema() == (255, 255)


def test_registration_failure_blocks_queue_and_recovers(ui):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridge")
    preview(browser, browser.run(), bridge)
    requests = []
    prompts = []
    browser.page.on("request", lambda request: prompts.append(request.url)
                    if request.method == "POST" and request.url.endswith("/prompt") else None)

    def reject_registration(route):
        requests.append(route.request.url)
        route.fulfill(status=400, body="")

    for index, rejected in enumerate((True, False, True)):
        if rejected:
            browser.page.route("**/impact/set/pb_id_image?*", reject_registration)
        else:
            browser.page.unroute("**/impact/set/pb_id_image?*", reject_registration)
        before = len(requests)
        submitted = len(prompts)
        expected = edit_mask(browser, bridge, args.artifacts, f"registration-{index}", right=bool(index % 2))
        if rejected:
            browser.page.get_by_test_id("queue-button").click()
            expect(browser.page.get_by_text("Preview Bridge: image registration failed (400)", exact=False).first).to_be_visible()
            assert len(requests) > before, "Fault injection did not intercept registration"
            assert len(prompts) == submitted, "An unregistered edit must not queue stale output"
            browser.page.keyboard.press("Escape")
            browser.page.get_by_test_id("toast").filter(has_text="image registration failed").get_by_role("button").click()
        else:
            entry = browser.run()
            mask_output(browser, entry, mask, expected)


@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("shared_file", [False, True], ids=["new_edit", "shared_file"])
def test_queued_bridge_results_preserve_the_latest_saved_edit(ui, node_type, shared_file):
    browser, args = ui
    pending = []
    target = None
    holding = False

    def is_target_result(message):
        event = json.loads(message) if isinstance(message, str) else {}
        data = event.get("data", {})
        return event.get("type") == "executed" and str(data.get("display_node", data.get("node"))) == target

    def delay_results(socket):
        server = socket.connect_to_server()

        def receive(message):
            if holding and (pending or is_target_result(message)):
                pending.append((socket, message))
            else:
                socket.send(message)

        server.on_message(receive)

    browser.page.route_web_socket(re.compile(r"/ws(?:\?|$)"), delay_results)
    browser.app.dispose()
    browser.page.reload()
    expect(browser.page.get_by_test_id("queue-button")).to_be_visible(timeout=60000)
    browser.app = browser.page.evaluate_handle("async () => (await import('/scripts/app.js')).app")
    browser.new_workflow()
    source, target, output, mask = build_bridge(browser, node_type)
    preview(browser, browser.run(), target)
    expected = edit_mask(browser, target, args.artifacts, "first")
    if shared_file:
        choose(browser, target, "restore_mask", "always")
        number(browser, source, "width", 512)
        number(browser, source, "height", 512)
    holding = True
    first = browser.run()
    if shared_file:
        choose(browser, target, "restore_mask", "if_same_size")
        number(browser, source, "width", 256)
        number(browser, source, "height", 256)
    else:
        expected = edit_mask(browser, target, args.artifacts, "second", right=True)
    saved = widget_state(browser, [target])
    second = browser.run()
    first_reference = first["prompt"][2][target]["inputs"]["image"]
    second_reference = second["prompt"][2][target]["inputs"]["image"]
    assert (first_reference == second_reference) == shared_file
    results = [index for index, (_, message) in enumerate(pending) if is_target_result(message)]
    assert len(results) == 2, "Both actual execution results must be delayed"
    first_messages, second_messages = pending[:results[1]], pending[results[1]:]
    holding = False
    for socket, message in first_messages:
        socket.send(message)
    preview(browser, first, target)
    if not shared_file:
        assert widget_state(browser, [target]) == saved, "An earlier queued result must preserve the newer saved edit"
    for socket, message in second_messages:
        socket.send(message)
    preview(browser, second, target)
    mask_output(browser, second, mask, expected)
    if not shared_file:
        assert widget_state(browser, [target]) == saved
    reload_workflow(browser, args.artifacts, [target], False)
    mask_output(browser, browser.run(), mask, expected)
    context_menu(browser, target, re.compile(r"^Open in MaskEditor"))
    wait_editor_canvas(browser, expected.size)
    pixels_equal(editor_mask(browser), expected, tolerance=1)
    browser.page.get_by_role("button", name=re.compile(r"Cancel$")).click()


def wait_editor_canvas(browser, size):
    canvas = browser.page.locator("#maskEditorCanvasContainer canvas").nth(2)
    expect(canvas).to_be_visible()
    expect(canvas).to_have_js_property("width", size[0])
    expect(canvas).to_have_js_property("height", size[1])


@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("block", [False, True], ids=["output", "blocked"])
def test_unedited_requeue_preserves_no_mask_and_block(ui, node_type, block):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    if block:
        browser.page.mouse.click(*point(browser, bridge, widget="block"))
    observed = []
    for index in range(3):
        entry = browser.run()
        preview(browser, entry, bridge)
        record = {"prompt_id": entry["prompt"][1], "submitted_image": entry["prompt"][2][bridge]["inputs"]["image"],
                  "consumer_ran": output in entry["outputs"], "mask_ran": mask in entry["outputs"]}
        if not block:
            actual = preview(browser, entry, mask).convert("L")
            actual.save(args.artifacts / f"queue-{index + 1}-mask.png")
            record.update(size=actual.size, extrema=actual.getextrema())
        browser.page.screenshot(path=str(args.artifacts / f"queue-{index + 1}.png"))
        observed.append(record)
    (args.artifacts / "queues.json").write_text(json.dumps(observed, indent=2))
    assert len({record["prompt_id"] for record in observed}) == 3
    if block:
        assert all(not row["consumer_ran"] and not row["mask_ran"] for row in observed), observed
    else:
        size, extrema = ((32, 32), (255, 255)) if node_type == "PreviewBridgeLatent" else ((64, 64), (0, 0))
        assert all(row["size"] == size and row["extrema"] == extrema for row in observed), observed


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_RESTART_COMMAND"),
                    reason="Provide a restart command for the isolated test server")
@pytest.mark.parametrize("node_type", BRIDGES)
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
def test_saved_mask_survives_server_restart_without_reloading_tab(ui, node_type, restore):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, node_type)
    choose(browser, bridge, "restore_mask", restore)
    browser.run()
    expected = edit_mask(browser, bridge, args.artifacts, "before-restart")
    before = browser.run()
    mask_output(browser, before, mask, expected)
    submitted = before["prompt"][2][bridge]["inputs"]["image"]
    saved, = before["outputs"][bridge]["images"]
    assert saved["type"] == "input", "Restart must preserve a durable MaskEditor save"
    saved_bytes = file_bytes(browser, saved)
    state = widget_state(browser, [bridge])
    browser.page.evaluate("""async () => {
        const {api} = await import('/scripts/api.js');
        window.bridgeReconnects = 0;
        api.addEventListener('reconnected', () => window.bridgeReconnects++, {once: true});
    }""")
    subprocess.run(shlex.split(os.environ["IMPACT_E2E_RESTART_COMMAND"]), check=True, timeout=90)
    browser.page.wait_for_function("window.bridgeReconnects === 1", timeout=90000)
    assert widget_state(browser, [bridge]) == state
    assert file_bytes(browser, saved) == saved_bytes
    for _ in range(3):
        entry = browser.run()
        mask_output(browser, entry, mask, expected)
        assert entry["prompt"][2][bridge]["inputs"]["image"] != submitted, "Server restart requires a new registration"
