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
    "PreviewBridge": ["image", "block", "restore_mask", "$$canvas-image-preview"],
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
