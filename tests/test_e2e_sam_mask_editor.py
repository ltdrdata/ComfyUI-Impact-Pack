"""Real SAM tool integration against the frontend MaskEditor extension API.

Preview Bridge editing and workflow compatibility are tested by the prerequisite
Preview Bridge suite. These cases require IMPACT_E2E_SAM_IMAGE and the configured
SAM model; latent coverage also requires the SD1.5 VAE.
"""
import base64
from email.parser import BytesParser
import io
import json
import os
import re
import time

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageOps
from playwright.sync_api import expect

from test_e2e_preview_bridge import (
    add_node,
    choose,
    connect,
    context_menu,
    editor_image,
    editor_mask,
    latent_samples,
    mask_output,
    pixels_equal,
    point,
    preview,
    reload_workflow,
    ui as ui,
    wait_editor_canvas,
    widget_state,
)


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
    wait_editor_canvas(browser, expected.size)
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
            wait_editor_canvas(browser, original.size)
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
    wait_editor_canvas(browser, expected.size)
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
