"""Chrome E2E for the Preview Bridge image/latent tutorial workflows.

Requires an isolated ComfyUI server with Impact Pack, Playwright and Chrome.
Default cases need no weights: image compositing and a second latent bridge
observe the edited image, MASK and latent noise_mask independently. The optional
SAM case requires IMPACT_E2E_SAM_IMAGE and the server's configured SAM model.
"""
from argparse import Namespace
import base64
from email.parser import BytesParser
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageOps
from playwright.sync_api import expect
from safetensors.numpy import load as load_safetensors

from e2e_ui import browser_session


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


def export_workflow(ui, path):
    ui.page.get_by_role("button", name="Graph mode, workflow actions", exact=True).click()
    ui.page.get_by_text("Export", exact=True).filter(visible=True).click()
    dialog = ui.page.get_by_role("dialog")
    dialog.get_by_role("textbox").fill(path.stem)
    with ui.page.expect_download() as download:
        dialog.get_by_role("button", name="Confirm", exact=True).click()
    download.value.save_as(path)
    return json.loads(path.read_text())


def reload_workflow(ui, artifacts, ids, restore_by_name):
    ui.page.get_by_role("button", name="Settings (Ctrl + ,)", exact=True).click()
    ui.page.get_by_placeholder("Search Settings...", exact=True).fill("Restore widget values")
    switch = ui.page.locator('[id="Comfy.Workflow.NamedValuesRestore"]')
    expect(switch).to_be_visible()
    if switch.get_attribute("aria-checked") != str(restore_by_name).lower():
        switch.click()
    ui.page.keyboard.press("Escape")
    before = widget_state(ui, ids)
    for state in before.values():
        assert [w["name"] for w in state["widgets"]] == WIDGET_NAMES[state["type"]]
    path = artifacts / "edited-workflow.json"
    saved = export_workflow(ui, path)
    ui.new_workflow()
    ui.load_workflow(path, len(saved["nodes"]))
    assert widget_state(ui, ids) == before, "Widget names/order/values changed during native import"
    restored = export_workflow(ui, artifacts / "reloaded-workflow.json")
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


@pytest.mark.parametrize("restore,restored", [("always", True), ("if_same_size", False)],
                         ids=["restored_after_resize", "discarded_after_resize"])
def test_restored_latent_mask_reaches_the_next_bridge(ui, restore, restored):
    browser, args = ui
    source, bridge, output, mask = build_bridge(browser, "PreviewBridgeLatent")
    multiply = add_node(browser, "LatentMultiply", 1220, 760)
    connect(browser, source, multiply)
    connect(browser, multiply, bridge)
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
    # Changing this parameter creates a fresh latent with the same sample values.
    number(browser, multiply, "multiplier", 2)
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


def test_queued_bridge_results_preserve_the_latest_saved_edit(ui):
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
    browser.app = browser.page.evaluate_handle("async () => (await import('/scripts/app.js')).app")
    browser.new_workflow()
    source, target, output, mask = build_bridge(browser, "PreviewBridge")
    preview(browser, browser.run(), target)
    edit_mask(browser, target, args.artifacts, "first")
    holding = True
    first = browser.run()
    expected = edit_mask(browser, target, args.artifacts, "second", right=True)
    saved = widget_state(browser, [target])
    second = browser.run()
    results = [index for index, (_, message) in enumerate(pending) if is_target_result(message)]
    assert len(results) == 2, "Both actual execution results must be delayed"
    first_messages, second_messages = pending[:results[1]], pending[results[1]:]
    holding = False
    for socket, message in first_messages:
        socket.send(message)
    preview(browser, first, target)
    assert widget_state(browser, [target]) == saved, "An earlier queued result must preserve the newer saved edit"
    for socket, message in second_messages:
        socket.send(message)
    preview(browser, second, target)
    mask_output(browser, second, mask, expected)
    assert widget_state(browser, [target]) == saved
    reload_workflow(browser, args.artifacts, [target], False)
    mask_output(browser, browser.run(), mask, expected)
    context_menu(browser, target, re.compile(r"^Open in MaskEditor"))
    wait_sam_canvas(browser, expected.size)
    pixels_equal(editor_mask(browser), expected, tolerance=1)
    browser.page.get_by_role("button", name="Cancel", exact=True).click()


def sam_image_source(browser):
    source = add_node(browser, "LoadImage", 220, 200)
    upload = browser.app.evaluate("""(app, id) =>
        app.graph.getNodeById(id).widgets.find(w => w.type === 'button').name
    """, source)
    with browser.page.expect_file_chooser() as chooser:
        browser.page.mouse.click(*point(browser, source, widget=upload))
    chooser.value.set_files(os.environ["IMPACT_E2E_SAM_IMAGE"])
    browser.page.wait_for_function("""({app, id}) => {
        const image = app.graph.getNodeById(id).imgs?.[0];
        return image?.complete && image.naturalWidth > 0;
    }""", arg={"app": browser.app, "id": source})
    return source


def capture_sam(browser, *, invalid_image=False, hold=False):
    captured = []

    def capture(route):
        request = route.request
        multipart = BytesParser().parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.post_data_buffer)
        fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                  for part in multipart.get_payload()}
        started = time.monotonic()
        body = request.post_data_buffer
        if invalid_image and not captured:
            body = body.replace(fields["image"], b"invalid PNG", 1)
        response = route.fetch(post_data=body, timeout=180000)
        captured.append({"status": response.status, "body": response.body(), "image": fields["image"],
                         "prompts": json.loads(fields["prompts"]), "seconds": time.monotonic() - started})
        if hold:
            captured[-1]["release"] = lambda: route.fulfill(response=response)
        else:
            route.fulfill(response=response)

    browser.page.route("**/impact/sam/detect", capture)
    return captured


def select_sam_points(browser, y=0.5):
    box = browser.page.get_by_test_id("mask-editor-mask-canvas").bounding_box()
    browser.page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * y)
    browser.page.mouse.click(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.1, button="right")
    expect(browser.page.get_by_label("SAM status")).to_have_text("2 points.")
    expect(browser.page.get_by_test_id("mask-editor-tool-overlay").locator("circle")).to_have_count(2)


def detect_sam(browser, artifacts, name, captured, positive_y=0.5):
    size = editor_image(browser).size
    base = browser.page.get_by_test_id("mask-editor-image-canvas").evaluate(
        "canvas => canvas.toDataURL('image/png').split(',')[1]")
    with browser.page.expect_response(lambda r: r.url.endswith("/impact/sam/detect"), timeout=180000):
        browser.page.get_by_role("button", name="Detect", exact=True).click()
    assert captured[-1]["status"] == 200, captured[-1]["body"]
    with Image.open(io.BytesIO(captured[-1]["image"])) as uploaded, Image.open(io.BytesIO(base64.b64decode(base))) as original:
        pixels_equal(uploaded.convert("RGB"), original.convert("RGB"))
    prompts = captured[-1]["prompts"]
    np.testing.assert_allclose(prompts["positive_points"], [[size[0] * 0.5, size[1] * positive_y]], atol=1)
    np.testing.assert_allclose(prompts["negative_points"], [[size[0] * 0.1, size[1] * 0.1]], atol=1)
    with Image.open(io.BytesIO(captured[-1]["body"])) as image:
        expected = image.convert("L")
    assert expected.size == size
    assert expected.getextrema() == (0, 255), "SAM must select a region and exclude the background"
    expected.save(artifacts / f"{name}-response-mask.png")
    (artifacts / f"{name}-request.json").write_text(json.dumps({
        "prompts": prompts, "image_size": list(size), "seconds": captured[-1]["seconds"],
        "selected_pixels": int(np.count_nonzero(np.asarray(expected)))
    }, indent=2))
    expect(browser.page.get_by_label("SAM status")).to_have_text("Mask applied.")
    pixels_equal(editor_mask(browser), expected)
    browser.page.screenshot(path=str(artifacts / f"{name}-editor.png"))
    return expected


def sam_stroke(browser, tool, x, y):
    browser.page.locator(f'[data-tool="{tool}"]').click()
    box = browser.page.get_by_test_id("mask-editor-mask-canvas").bounding_box()
    browser.page.mouse.move(box["x"] + box["width"] * x, box["y"] + box["height"] * y)
    browser.page.mouse.down()
    browser.page.mouse.move(box["x"] + box["width"] * (x + 0.1), box["y"] + box["height"] * (y + 0.05), steps=12)
    browser.page.mouse.up()


def open_sam_source(browser):
    source = sam_image_source(browser)
    context_menu(browser, source, re.compile(r"^Open in MaskEditor"))
    browser.page.get_by_role("button", name="SAM Detector", exact=True).click()
    select_sam_points(browser)
    return source


def wait_sam_canvas(browser, size):
    canvas = browser.page.get_by_test_id("mask-editor-mask-canvas")
    expect(canvas).to_be_visible()
    expect(canvas).to_have_js_property("width", size[0])
    expect(canvas).to_have_js_property("height", size[1])


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"), reason="Set IMPACT_E2E_SAM_IMAGE and install the configured SAM model")
def test_interactive_sam_http_error_preserves_points_and_retries(ui):
    browser, args = ui
    source = open_sam_source(browser)
    before = widget_state(browser, [source])
    captured = capture_sam(browser, invalid_image=True)
    with browser.page.expect_response(lambda response: response.url.endswith("/impact/sam/detect")):
        browser.page.get_by_role("button", name="Detect", exact=True).click()
    assert captured[-1]["status"] == 400
    message = json.loads(captured[-1]["body"])["error"]
    expect(browser.page.get_by_label("SAM status")).to_have_text(f"SAM detection failed: {message}")
    expect(browser.page.get_by_role("button", name="Detect", exact=True)).to_be_enabled()
    expect(browser.page.get_by_test_id("mask-editor-tool-overlay").locator("circle")).to_have_count(2)
    assert not np.count_nonzero(np.asarray(editor_mask(browser)))
    assert widget_state(browser, [source]) == before
    expected = detect_sam(browser, args.artifacts, "sam-retry", captured)
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
    context_menu(browser, source, re.compile(r"^Open in MaskEditor"))
    wait_sam_canvas(browser, expected.size)
    pixels_equal(editor_mask(browser), expected, tolerance=1)
    browser.page.get_by_role("button", name="Cancel", exact=True).click()


@pytest.mark.parametrize("action", ["edit", "switch", "reopen"])
@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"), reason="Set IMPACT_E2E_SAM_IMAGE and install the configured SAM model")
def test_interactive_sam_delayed_response_cannot_replace_later_edits(ui, action):
    browser, args = ui
    source = open_sam_source(browser)
    before = widget_state(browser, [source])
    original = editor_image(browser)
    captured = capture_sam(browser, hold=True)
    browser.page.get_by_role("button", name="Detect", exact=True).click()
    deadline = time.monotonic() + 180
    while not captured:
        assert time.monotonic() < deadline, "Actual SAM response was not captured"
        browser.page.evaluate("() => new Promise(requestAnimationFrame)")
    assert captured[-1]["status"] == 200
    with Image.open(io.BytesIO(captured[-1]["body"])) as response:
        assert response.convert("L").getextrema() == (0, 255)
    expect(browser.page.get_by_label("SAM status")).to_have_text("Detecting…")
    with browser.page.expect_event("requestfailed", predicate=lambda request: request.url.endswith("/impact/sam/detect")):
        if action == "edit":
            browser.page.get_by_role("button", name="Invert", exact=True).click()
        elif action == "switch":
            browser.page.locator('[data-tool="pen"]').click()
        else:
            browser.page.get_by_role("button", name="Cancel", exact=True).click()
            expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
            context_menu(browser, source, re.compile(r"^Open in MaskEditor"))
            wait_sam_canvas(browser, original.size)
    if action != "edit":
        sam_stroke(browser, "pen", 0.8, 0.1)
    expected = editor_mask(browser)
    assert np.count_nonzero(np.asarray(expected))
    captured[-1]["release"]()
    browser.page.evaluate("() => new Promise(requestAnimationFrame)")
    pixels_equal(editor_mask(browser), expected)
    pixels_equal(editor_image(browser), original)
    assert widget_state(browser, [source]) == before
    browser.page.screenshot(path=str(args.artifacts / "cancelled-response-editor.png"))
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
    context_menu(browser, source, re.compile(r"^Open in MaskEditor"))
    wait_sam_canvas(browser, expected.size)
    pixels_equal(editor_mask(browser), expected, tolerance=1)
    pixels_equal(editor_image(browser), original, tolerance=1)
    browser.page.get_by_role("button", name="Cancel", exact=True).click()


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"), reason="Set IMPACT_E2E_SAM_IMAGE and install the configured SAM model")
def test_interactive_sam_legacy_backend_matches_snapshot_and_reprepares(ui):
    browser, args = ui
    source = sam_image_source(browser)
    filename = browser.app.evaluate("(app, id) => app.graph.getNodeById(id).widgets.find(widget => widget.name === 'image').value", source)
    with Image.open(os.environ["IMPACT_E2E_SAM_IMAGE"]) as image:
        width, height = image.size
        prompts = {"positive_points": [[width * 0.5, height * 0.5]],
                   "negative_points": [[width * 0.1, height * 0.1]], "threshold": 0.7}
        snapshot = io.BytesIO()
        image.convert("RGB").save(snapshot, format="PNG")
    response = browser.page.request.post(f"{browser.base_url}/impact/sam/detect", multipart={
        "image": {"name": "snapshot.png", "mimeType": "image/png", "buffer": snapshot.getvalue()},
        "prompts": json.dumps(prompts)}, timeout=180000)
    assert response.ok, response.text()
    with Image.open(io.BytesIO(response.body())) as image:
        expected = image.convert("L")
    for _ in range(2):
        prepare = browser.page.request.post(f"{browser.base_url}/sam/prepare", data={
            "filename": filename, "type": "input", "subfolder": "", "sam_model_name": "auto"}, timeout=180000)
        assert prepare.ok, prepare.text()
        detected = browser.page.request.post(f"{browser.base_url}/sam/detect", data=prompts, timeout=180000)
        assert detected.ok, detected.text()
        with Image.open(io.BytesIO(detected.body())) as image:
            pixels_equal(image.convert("L"), expected)
        assert browser.page.request.post(f"{browser.base_url}/sam/release").ok


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"),
                    reason="Set IMPACT_E2E_SAM_IMAGE and install the server's configured SAM model")
def test_interactive_sam_saves_detected_mask_to_bridge(ui):
    browser, args = ui
    source = sam_image_source(browser)
    bridge = add_node(browser, "PreviewBridge", 720, 200)
    output = add_node(browser, "PreviewImage", 1220, 200)
    convert = add_node(browser, "MaskToImage", 220, 760)
    mask = add_node(browser, "PreviewImage", 720, 760)
    connect(browser, source, bridge)
    connect(browser, bridge, output)
    connect(browser, bridge, convert, source_slot=1)
    connect(browser, convert, mask)
    entry = browser.run()
    original = preview(browser, entry, bridge).convert("RGB")
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    canvas = browser.page.get_by_test_id("mask-editor-mask-canvas")
    expect(canvas).to_be_visible()
    browser.page.get_by_role("button", name="SAM Detector", exact=True).click()
    status = browser.page.get_by_label("SAM status")
    expect(status).to_have_text("Add points, then Detect.")
    assert not browser.page.evaluate("() => performance.getEntriesByType('resource').some(item => item.name.includes('legacy-sam-editor'))")
    captured = capture_sam(browser)
    select_sam_points(browser)
    expected = detect_sam(browser, args.artifacts, "sam-first", captured)
    browser.page.get_by_role("button", name="Undo point", exact=True).click()
    expect(status).to_have_text("1 point.")
    pixels_equal(editor_mask(browser), expected)
    browser.page.get_by_title("Undo", exact=True).click()
    assert not np.count_nonzero(np.asarray(editor_mask(browser)))
    browser.page.get_by_title("Redo", exact=True).click()
    pixels_equal(editor_mask(browser), expected)
    sam_stroke(browser, "rgbPaint", 0.7, 0.7)
    painted = editor_image(browser)
    assert ImageChops.difference(painted, original).getbbox(), "Paint stroke must change the image"
    pixels_equal(editor_mask(browser), expected)
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(canvas).not_to_be_visible()
    entry = browser.run()
    mask_output(browser, entry, mask, expected)
    pixels_equal(preview(browser, entry, output).convert("RGB"), painted, tolerance=1)
    first_widgets = widget_state(browser, [bridge])

    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    expect(browser.page.get_by_test_id("pointer-zone")).to_be_visible()
    pixels_equal(editor_mask(browser), expected, tolerance=1)
    pixels_equal(editor_image(browser), painted, tolerance=1)
    sam_stroke(browser, "pen", 0.8, 0.1)
    manual = editor_mask(browser)
    assert ImageChops.difference(manual, expected).getbbox()
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(canvas).not_to_be_visible()
    entry = browser.run()
    mask_output(browser, entry, mask, manual)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    expect(browser.page.get_by_test_id("pointer-zone")).to_be_visible()
    pixels_equal(editor_mask(browser), manual, tolerance=1)
    pixels_equal(editor_image(browser), painted, tolerance=1)
    browser.page.get_by_role("button", name="SAM Detector", exact=True).click()
    expect(status).to_have_text("Add points, then Detect.")
    select_sam_points(browser, 0.25)
    expected = detect_sam(browser, args.artifacts, "sam-reopened", captured, positive_y=0.25)
    assert ImageChops.difference(expected, manual).getbbox(), "SAM must replace the manually drawn mask"
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(canvas).not_to_be_visible()
    assert widget_state(browser, [bridge]) != first_widgets, "Saving must create a new image reference"
    entry = browser.run()
    mask_output(browser, entry, mask, expected)
    pixels_equal(preview(browser, entry, output).convert("RGB"), painted, tolerance=1)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    browser.page.get_by_role("button", name="SAM Detector", exact=True).click()
    select_sam_points(browser)
    browser.page.get_by_title("Rotate Right", exact=True).click()
    expect(status).to_have_text("Image or mask changed. Add points to detect again.")
    expect(browser.page.get_by_test_id("mask-editor-tool-overlay")).not_to_be_visible()
    painted = painted.transpose(Image.Transpose.ROTATE_270)
    pixels_equal(editor_image(browser), painted, tolerance=1)
    select_sam_points(browser)
    expected = detect_sam(browser, args.artifacts, "sam-rotated", captured)
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(canvas).not_to_be_visible()
    for restore_by_name in (False, True):
        reload_workflow(browser, args.artifacts, [bridge], restore_by_name)
        entry = browser.run()
        mask_output(browser, entry, mask, expected)
        pixels_equal(preview(browser, entry, output).convert("RGB"), painted, tolerance=1)


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"),
                    reason="Set IMPACT_E2E_SAM_IMAGE and install the server's configured SAM model and SD1.5 VAE")
def test_interactive_sam_preserves_latent_samples_and_downstream_noise_mask(ui):
    browser, args = ui
    source = sam_image_source(browser)
    vae = add_node(browser, "VAELoader", 220, 760)
    choose(browser, vae, "vae_name", "vae-ft-mse-840000-ema-pruned.safetensors")
    encode = add_node(browser, "VAEEncode", 720, 760)
    bridge = add_node(browser, "PreviewBridgeLatent", 720, 200)
    choose(browser, bridge, "preview_method", "Latent2RGB-SD15")
    output = add_node(browser, "SaveLatent", 1220, 200)
    second = add_node(browser, "PreviewBridgeLatent", 1700, 200)
    choose(browser, second, "preview_method", "Latent2RGB-SD15")
    convert = add_node(browser, "MaskToImage", 1220, 760)
    mask = add_node(browser, "PreviewImage", 1700, 760)
    connect(browser, source, encode)
    connect(browser, vae, encode, target_slot=1)
    connect(browser, encode, bridge)
    connect(browser, bridge, second)
    connect(browser, second, output)
    connect(browser, second, convert, source_slot=1)
    connect(browser, convert, mask)
    initial = browser.run()
    samples = latent_samples(browser, initial, output)
    preview(browser, initial, bridge)
    context_menu(browser, bridge, re.compile(r"^Open in MaskEditor"))
    browser.page.get_by_role("button", name="SAM Detector", exact=True).click()
    captured = capture_sam(browser)
    select_sam_points(browser)
    expected = detect_sam(browser, args.artifacts, "sam-latent", captured)
    browser.page.get_by_role("button", name="Save", exact=True).click()
    expect(browser.page.get_by_test_id("pointer-zone")).not_to_be_visible()
    entry = browser.run()
    mask_output(browser, entry, mask, expected)
    pixels_equal(ImageOps.invert(preview(browser, entry, bridge).getchannel("A")), expected)
    pixels_equal(ImageOps.invert(preview(browser, entry, second).getchannel("A")), expected)
    np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)
    for restore_by_name in (False, True):
        reload_workflow(browser, args.artifacts, [bridge, second], restore_by_name)
        entry = browser.run()
        mask_output(browser, entry, mask, expected)
        np.testing.assert_array_equal(latent_samples(browser, entry, output), samples)
