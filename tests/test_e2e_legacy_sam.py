"""Legacy SAM compatibility on a frontend without the ClipspaceDialog export."""
from argparse import Namespace
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import expect

from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, setting


@pytest.fixture
def ui(tmp_path, request):
    artifacts = Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=120, artifacts=artifacts)
    with browser_session(args) as browser:
        browser.page.keyboard.press("Escape")
        browser.page.set_viewport_size({"width": 2200, "height": 1300})
        browser.page.set_default_timeout(10000)
        browser.new_workflow()
        yield browser, artifacts


def assert_sam_loaded(browser):
    names = browser.app.evaluate("app => app.extensions.map(extension => extension.name)")
    assert names.count("Comfy.Impact.SAMEditor") == 1, "Legacy SAM failed to register"
    requests = browser.page.evaluate("() => performance.getEntriesByType('resource').map(item => new URL(item.name).pathname)")
    assert "/extensions/core/clipspace.js" not in requests, "Removed Clipspace module was requested"
    assert "/scripts/ui.js" not in requests, "Deprecated UI module was requested"


def node_menu(browser, node_id, name):
    pos = browser.app.evaluate("""(app, id) => {
        const node = app.graph.getNodeById(id);
        return app.canvasPosToClientPos([node.pos[0] + node.size[0] / 2, node.pos[1] + 15]);
    }""", node_id)
    browser.page.mouse.click(*pos, button="right")
    browser.page.get_by_role("menuitem", name=name, exact=True).click()


def source_image(browser, path):
    source = browser.add_node("Load Image", "LoadImage", 220, 200)
    upload_image(browser, source, path)
    return source


def upload_image(browser, source, path):
    button = browser.app.evaluate("""(app, id) =>
        app.graph.getNodeById(id).widgets.find(widget => widget.type === 'button').name
    """, source)
    with browser.page.expect_file_chooser() as chooser:
        browser.page.mouse.click(*browser.point("LoadImage", widget=button))
    with browser.page.expect_response(lambda response: response.url.endswith("/upload/image")) as uploaded:
        chooser.value.set_files(path)
    assert uploaded.value.ok
    filename = uploaded.value.json()["name"]
    browser.page.wait_for_function("""({app, filename}) => {
        const image = app.graph._nodes.find(node => node.type === 'LoadImage').imgs?.[0];
        return image?.complete && image.naturalWidth > 0 && new URL(image.src).searchParams.get('filename') === filename;
    }""", arg={"app": browser.app, "filename": filename})


def clipspace_menu(browser):
    browser.page.locator('[data-testid="comfy-menu-button"], .comfy-menu-button-wrapper').click()
    browser.page.get_by_role("menuitem", name="Edit", exact=True).hover()
    browser.page.get_by_role("menuitem", name="Clipspace", exact=True).click()


def open_clipspace(browser):
    clipspace_menu(browser)
    preview = browser.page.locator("#clipspace_preview")
    expect(preview).to_be_visible()
    content = preview.locator("xpath=ancestor::div[contains(@class, 'comfy-modal-content')]")
    expect(content.get_by_role("button", name="Impact SAM Detector", exact=True)).to_have_count(1)
    return content


def clipspace_state(browser):
    return browser.page.evaluate("""() => {
        const state = window.comfyAPI.app.ComfyApp.clipspace;
        return {selectedIndex: state.selectedIndex, images: state.images,
                urls: state.imgs.map(image => image.src)};
    }""")


def widgets(browser):
    # SAM updates LoadImage. Output preview widgets are recreated only on execution.
    return browser.app.evaluate("""app => app.graph._nodes.filter(node => node.type === 'LoadImage').map(node => ({
        id: node.id, type: node.type,
        widgets: (node.widgets ?? []).map(widget => ({name: widget.name, value: widget.value ?? null}))
    }))""")


def supports_named_restore(browser):
    return browser.app.evaluate("app => 'Comfy.Workflow.NamedValuesRestore' in app.ui.settings.settingsLookup")


def reload_native(browser, artifacts, named):
    if supports_named_restore(browser):
        setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", named)
    else:
        assert not named, "This frontend only provides positional widget restoration"
    before = widgets(browser)
    path = artifacts / "edited-workflow.json"
    export_workflow(browser, path)
    saved = json.loads(path.read_text())
    browser.new_workflow()
    browser.load_workflow(path, len(saved["nodes"]))
    assert widgets(browser) == before, "Widget names, order or values changed on native load"
    restored_path = artifacts / "reloaded-workflow.json"
    export_workflow(browser, restored_path)
    restored = json.loads(restored_path.read_text())
    assert {node["id"]: node.get("widgets_values", []) for node in restored["nodes"]} == {
        node["id"]: node.get("widgets_values", []) for node in saved["nodes"]}


def test_legacy_sam_context_menu_and_clipspace_reopen(ui):
    browser, artifacts = ui
    assert_sam_loaded(browser)
    clipspace_menu(browser)
    empty_dialog = browser.page.locator(".comfy-modal").filter(has_text="Clipspace is Empty!")
    expect(empty_dialog).to_be_visible()
    expect(browser.page.get_by_role("button", name="Impact SAM Detector", exact=True)).not_to_be_visible()
    empty_dialog.get_by_role("button", name="Close", exact=True).click()
    expect(empty_dialog).not_to_be_visible()
    browser.page.mouse.click(600, 400)
    expect(browser.page.get_by_role("menuitem", name="Edit", exact=True)).not_to_be_visible()
    path = artifacts / "source.png"
    Image.new("RGB", (256, 256), (64, 128, 192)).save(path)
    source = source_image(browser, path)
    for _ in range(2):
        node_menu(browser, source, "Open in SAM Detector")
        expect(browser.page.locator("#pointsCanvas")).to_be_visible()
        expect(browser.page.get_by_role("button", name="Save to node", exact=True)).to_be_enabled()
        browser.page.get_by_role("button", name="Cancel", exact=True).click()
        expect(browser.page.locator("#pointsCanvas")).not_to_be_visible()
    node_menu(browser, source, "Copy (Clipspace)")
    for _ in range(2):
        content = open_clipspace(browser)
        content.get_by_role("button", name="Impact SAM Detector", exact=True).click()
        expect(browser.page.locator("#pointsCanvas")).to_be_visible()
        browser.page.get_by_role("button", name="Cancel", exact=True).click()
        content.get_by_role("button", name="Close", exact=True).click()


def test_legacy_sam_keyboard_cancel_resize_and_change_image(ui):
    browser, artifacts = ui
    assert_sam_loaded(browser)
    first = artifacts / "landscape.png"
    second = artifacts / "portrait.png"
    Image.new("RGB", (1400, 800), (64, 128, 192)).save(first)
    Image.new("RGB", (700, 1300), (192, 64, 128)).save(second)
    source = source_image(browser, first)
    for path, size, color in [(first, (1400, 800), [64, 128, 192]),
                              (second, (700, 1300), [192, 64, 128])]:
        if path == second:
            upload_image(browser, source, path)
        before = widgets(browser)
        for _ in range(2):
            node_menu(browser, source, "Open in SAM Detector")
            canvas = browser.page.locator("#pointsCanvas")
            expect(canvas).to_be_visible()
            expect(browser.page.get_by_role("button", name="Save to node", exact=True)).to_be_enabled()
            assert canvas.evaluate("canvas => [...canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data].every(value => value === 0)"), "Reopening retained prompt points"
            brush = browser.page.locator("#sam-brush")
            browser.page.keyboard.press("-")
            expect(brush).to_have_css("background-color", "rgb(255, 0, 0)")
            browser.page.keyboard.press("=")
            expect(brush).to_have_css("background-color", "rgb(0, 0, 255)")
            for width, height in [(1100, 900), (2200, 1300)]:
                browser.page.set_viewport_size({"width": width, "height": height})
                browser.page.wait_for_function("""({width, height}) => {
                    const image = document.getElementById('imageCanvas');
                    return image.width === width - 250 && image.height === height - 200;
                }""", arg={"width": width, "height": height})
                box = canvas.bounding_box()
                assert box["width"] / box["height"] == pytest.approx(size[0] / size[1], abs=0.005)
                pixel = browser.page.locator("#imageCanvas").evaluate("canvas => [...canvas.getContext('2d').getImageData(10, 10, 1, 1).data].slice(0, 3)")
                assert pixel == color, "Resize drew a previously edited image"
            canvas.click(position={"x": 60, "y": 60})
            assert canvas.evaluate("canvas => canvas.getContext('2d').getImageData(60, 60, 1, 1).data[3]") > 0
            browser.page.get_by_role("button", name="Cancel", exact=True).click()
            expect(canvas).not_to_be_visible()
            assert widgets(browser) == before, "Cancel changed the source image"
            browser.page.keyboard.press("-")
            expect(brush).to_have_css("background-color", "rgb(0, 0, 255)")


@pytest.mark.skipif(not os.environ.get("IMPACT_E2E_SAM_IMAGE"),
                    reason="Set IMPACT_E2E_SAM_IMAGE and install the configured SAM model")
@pytest.mark.parametrize("named", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("origin", ["input", "output_subfolder"])
def test_legacy_sam_detect_save_and_native_reload(ui, named, origin):
    browser, artifacts = ui
    assert_sam_loaded(browser)
    if named and not supports_named_restore(browser):
        pytest.skip("This frontend does not provide named widget restoration")
    source = source_image(browser, os.environ["IMPACT_E2E_SAM_IMAGE"])
    output_type, mask_type = ("PreviewImage", "SaveImage") if origin == "input" else ("SaveImage", "PreviewImage")
    titles = {"PreviewImage": "Preview Image", "SaveImage": "Save Image"}
    output = browser.add_node(titles[output_type], output_type,
                              1350 if origin == "output_subfolder" else 750, 200)
    convert = browser.add_node("Convert Mask to Image", "MaskToImage", 220, 800)
    mask_output = browser.add_node(titles[mask_type], mask_type, 750, 800)
    if origin == "output_subfolder":
        definition = browser.page.request.get(f"{browser.base_url}/object_info/RepeatImageBatch")
        assert definition.ok
        title = definition.json()["RepeatImageBatch"]["display_name"] or "RepeatImageBatch"
        browser.add_node(title, "RepeatImageBatch", 750, 200)
        browser.set_number("RepeatImageBatch", "amount", 2)
        browser.connect("LoadImage", "RepeatImageBatch")
        browser.connect("RepeatImageBatch", output_type)
    else:
        browser.connect("LoadImage", output_type)
    browser.page.mouse.move(*browser.point("LoadImage", slot=1))
    browser.page.mouse.down()
    browser.page.mouse.move(*browser.point("MaskToImage", is_input=True), steps=12)
    browser.page.mouse.up()
    browser.page.wait_for_function("""(app) => app.graph._nodes.find(node =>
        node.type === 'MaskToImage').inputs[0].link !== null""", arg=browser.app)
    browser.connect("MaskToImage", mask_type)
    if origin == "output_subfolder":
        browser.page.mouse.dblclick(*browser.point("SaveImage", widget="filename_prefix"), delay=100)
        editor = browser.page.locator("input.value:visible, textarea.value:visible")
        editor.fill("impact-legacy-sam/nested/source")
        editor.press("Enter")
    original = browser.images(browser.run())[output][0]
    responses = []

    def capture(route):
        response = route.fetch(timeout=180000)
        responses.append((route.request.post_data_json, response.body()))
        route.fulfill(response=response)

    browser.page.route("**/sam/detect", capture)
    previous = None
    for index in range(2):
        direct = index == 0 and origin == "input"
        with browser.page.expect_request(lambda request: request.url.endswith("/sam/prepare")) as prepared:
            if direct:
                node_menu(browser, source, "Open in SAM Detector")
            else:
                node_menu(browser, output if index == 0 else source, "Copy (Clipspace)")
                content = open_clipspace(browser)
                if index == 0:
                    content.locator("#clipspace_img_selector").select_option("1")
                before = clipspace_state(browser)
                if index == 0:
                    assert len(before["urls"]) == 2
                    assert before["selectedIndex"] == 1
                content.get_by_role("button", name="Impact SAM Detector", exact=True).click()
        if index == 0 and origin == "output_subfolder":
            assert prepared.value.post_data_json["type"] == "output"
            assert prepared.value.post_data_json["subfolder"] == "impact-legacy-sam/nested"
            assert prepared.value.post_data_json["filename"] == before["images"][1]["filename"]
        canvas = browser.page.locator("#pointsCanvas")
        expect(canvas).to_be_visible()
        save = browser.page.get_by_role("button", name="Save to node" if direct else "Save", exact=True)
        expect(save).to_be_enabled()
        box = canvas.bounding_box()
        browser.page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * (0.5 if index == 0 else 0.25))
        browser.page.mouse.click(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.1, button="right")
        with browser.page.expect_response(lambda r: r.url.endswith("/sam/detect"), timeout=180000) as detected:
            browser.page.get_by_role("button", name="Detect", exact=True).click()
        assert detected.value.ok, detected.value.text()
        prompts, body = responses[-1]
        assert len(prompts["positive_points"]) == len(prompts["negative_points"]) == 1
        with Image.open(io.BytesIO(body)) as image:
            expected = image.convert("RGB")
        assert expected.convert("L").getextrema() == (0, 255)
        expected.save(artifacts / f"detected-{index}.png")
        browser.page.wait_for_function("""() => {
            const canvas = document.getElementById('samEditorMaskCanvas');
            return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
                .some((value, index) => index % 4 === 0 && value > 0);
        }""")
        with browser.page.expect_response(lambda r: r.url.endswith("/upload/mask")) as uploaded:
            save.click()
        assert uploaded.value.ok
        expect(canvas).not_to_be_visible()
        if not direct:
            browser.page.wait_for_function("""() => {
                const image = document.getElementById('clipspace_preview');
                return image.complete && image.naturalWidth > 0 &&
                    image.src === window.comfyAPI.app.ComfyApp.clipspace.imgs[
                        window.comfyAPI.app.ComfyApp.clipspace.selectedIndex].src;
            }""")
            after = clipspace_state(browser)
            selected = before["selectedIndex"]
            assert after["selectedIndex"] == selected
            expect(content.locator("#clipspace_img_selector")).to_have_value(str(selected))
            assert len(after["urls"]) == len(before["urls"])
            assert after["urls"][selected] != before["urls"][selected]
            assert after["images"][selected]["type"] == "temp"
            for other in range(len(before["urls"])):
                if other != selected:
                    assert after["urls"][other] == before["urls"][other]
                    assert after["images"][other] == before["images"][other], "Save changed another image reference"
            expect(content.get_by_role("button", name="Impact SAM Detector", exact=True)).to_have_count(1)
            content.get_by_role("button", name="Close", exact=True).click()
            node_menu(browser, source, "Paste (Clipspace)")
        value = browser.app.evaluate("""(app, id) =>
            app.graph.getNodeById(id).widgets.find(widget => widget.name === 'image').value
        """, source)
        assert value != previous, "Each save must update the image reference"
        previous = value
        actual = browser.images(browser.run())
        assert actual[mask_output][0].tobytes() == expected.tobytes(), "Saved mask differs from real SAM response"
        assert actual[output][0].tobytes() == original.tobytes(), "SAM save changed the RGB image"
    reload_native(browser, artifacts, named)
    actual = browser.images(browser.run())
    assert actual[mask_output][0].tobytes() == expected.tobytes()
    assert actual[output][0].tobytes() == original.tobytes()
