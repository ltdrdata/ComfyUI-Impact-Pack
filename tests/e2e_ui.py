"""Shared Chrome UI interactions for the standalone E2E scripts.

Graph inspection is read-only. Nodes, widgets, links, workflow loading, and
execution are changed through browser input events, never app/graph API calls.
"""
import argparse
import base64
from contextlib import contextmanager
import io
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit

from PIL import Image
from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright
import pytest


def options(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-url", default="http://127.0.0.1:18188")
    parser.add_argument("--browser-channel", default="chrome")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


@pytest.fixture
def ui(tmp_path, request):
    artifacts = Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / request.node.name
    args = argparse.Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                              browser_channel="chrome", headed=False, timeout=180, artifacts=artifacts)
    with browser_session(args) as browser:
        browser.page.keyboard.press("Escape")
        yield browser, artifacts


class ComfyUI:
    def __init__(self, page, args):
        self.page = page
        self.base_url = args.base_url.rstrip("/")
        self.timeout = args.timeout
        self.app = page.evaluate_handle("async () => (await import('/scripts/app.js')).app")

    def new_workflow(self):
        self.page.get_by_role("button", name="Create a new blank workflow", exact=True).click()
        self.page.wait_for_function("""(app) => {
            return app.graph._nodes.length === 0 && app.canvas.graph === app.graph;
        }""", arg=self.app)

    def add_node(self, title, node_type, x, y):
        self.page.mouse.dblclick(x, y, delay=100)
        search = self.page.get_by_placeholder("Add a node...", exact=True)
        search.fill(node_type)
        label = self.page.locator("span.truncate, .option-container .font-semibold > span").filter(
            has_text=re.compile(rf"^\s*{re.escape(title)}\s*$"))
        self.page.get_by_role("option").filter(has=label).click()
        expect(search).not_to_be_visible()
        self.page.get_by_role("dialog").wait_for(state="hidden")
        self.page.mouse.move(x, y)
        self.page.mouse.click(x, y, delay=100)
        node_id = self.page.wait_for_function("""({app, type}) => {
            return app.canvas.state.ghostNodeId == null &&
                app.graph._nodes.find(n => n.type === type)?.id;
        }""", arg={"app": self.app, "type": node_type}).json_value()
        return str(node_id)

    def point(self, node_type, *, widget=None, slot=0, is_input=False):
        return self.page.evaluate("""({app, type, widget, slot, isInput}) => {
            const node = app.graph._nodes.find(n => n.type === type);
            const point = widget
                ? [node.pos[0] + node.size[0] / 2,
                   node.pos[1] + node.widgets.find(w => w.name === widget).last_y + 10]
                : node.getConnectionPos(isInput, slot);
            return app.canvasPosToClientPos(point);
        }""", {"app": self.app, "type": node_type, "widget": widget, "slot": slot, "isInput": is_input})

    def choose(self, node_type, widget, value):
        self.page.mouse.click(*self.point(node_type, widget=widget))
        self.page.get_by_role("menuitem", name=value, exact=True).click()

    def set_number(self, node_type, widget, value):
        self.page.mouse.dblclick(*self.point(node_type, widget=widget), delay=100)
        editor = self.page.locator("input.value:visible")
        editor.fill(str(value))
        editor.press("Enter")
        self.page.wait_for_function("""({app, type, widget, value}) =>
            app.graph._nodes.find(n => n.type === type).widgets.find(w => w.name === widget).value === value
        """, arg={"app": self.app, "type": node_type, "widget": widget, "value": value})

    def connect(self, source, target, target_slot=0):
        self.page.mouse.move(*self.point(source))
        self.page.mouse.down()
        self.page.mouse.move(*self.point(target, slot=target_slot, is_input=True), steps=12)
        self.page.mouse.up()
        self.page.wait_for_function("""({app, type, slot}) => {
            return app.graph._nodes.find(n => n.type === type).inputs[slot].link != null;
        }""", arg={"app": self.app, "type": target, "slot": target_slot})

    def load_workflow(self, workflow, node_count, *, check_node_errors=True):
        previous_nodes = self.app.evaluate_handle("app => app.graph._nodes.slice()")
        self.page.locator("#comfy-file-input").set_input_files(workflow)
        self.page.wait_for_function("""({app, count, previous, checkErrors}) => {
            return !app.configuringGraph && app.graph._nodes.length === count &&
                app.graph._nodes.every(n => (!checkErrors || !n.has_errors) && !previous.includes(n));
        }""", arg={"app": self.app, "count": node_count, "previous": previous_nodes, "checkErrors": check_node_errors})
        previous_nodes.dispose()
        self.page.get_by_role("button", name="Fit View (.)", exact=True).click()
        self.page.evaluate("""async app => {
            let previous;
            for (let frame = 0; frame < 120; frame++) {
                await new Promise(requestAnimationFrame);
                const current = [app.canvas.ds.scale, ...app.canvas.ds.offset].join();
                if (current === previous) return;
                previous = current;
            }
            throw new Error('Canvas did not settle after Fit View');
        }""", self.app)

    def run(self):
        with self.page.expect_response(
            lambda r: urlsplit(r.url).path in ("/prompt", "/api/prompt") and r.request.method == "POST"
        ) as submitted:
            self.page.get_by_test_id("queue-button").click()
        response = submitted.value
        assert response.ok, f"UI submission failed: {response.status} {response.text()}"
        prompt_id = response.json()["prompt_id"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            history = self.page.request.get(f"{self.base_url}/history/{prompt_id}")
            assert history.ok, f"History request failed: {history.status}"
            entry = history.json().get(prompt_id)
            if entry is not None:
                assert entry["status"]["status_str"] == "success", entry["status"]
                print(f"PASS: UI Run completed ({prompt_id})")
                return entry
            time.sleep(0.25)
        raise TimeoutError(f"UI execution did not finish within {self.timeout}s: {prompt_id}")

    def images(self, entry):
        outputs = {node: output["images"] for node, output in entry["outputs"].items() if output.get("images")}
        self.page.wait_for_function("""({app, outputs}) => Object.entries(outputs).every(([id, expected]) => {
            const images = app.graph.getNodeById(id)?.imgs;
            return images?.length === expected.length && images.every((image, index) => {
                const params = new URL(image.src).searchParams;
                return image.complete && image.naturalWidth > 0 && image.naturalHeight > 0 &&
                    ['filename', 'subfolder', 'type'].every(key =>
                        (params.get(key) ?? '') === (expected[index][key] ?? ''));
            });
        })""", arg={"app": self.app, "outputs": outputs})
        result = {}
        for node, metadata in outputs.items():
            result[node] = []
            for index, meta in enumerate(metadata):
                response = self.page.request.get(f"{self.base_url}/view", params={
                    key: meta[key] for key in ("filename", "subfolder", "type") if key in meta
                })
                assert response.ok
                with Image.open(io.BytesIO(response.body())) as image:
                    expected = image.convert("RGB")
                displayed = self.app.evaluate("""(app, {node, index}) => {
                    const image = app.graph.getNodeById(node).imgs[index];
                    const canvas = document.createElement('canvas');
                    canvas.width = image.naturalWidth;
                    canvas.height = image.naturalHeight;
                    canvas.getContext('2d').drawImage(image, 0, 0);
                    return canvas.toDataURL('image/png').split(',')[1];
                }""", {"node": node, "index": index})
                with Image.open(io.BytesIO(base64.b64decode(displayed))) as image:
                    assert image.size == expected.size, f"Preview {node}[{index}] dimensions differ"
                    assert image.convert("RGB").tobytes() == expected.tobytes(), f"Preview {node}[{index}] pixels differ"
                result[node].append(expected)
        return result


@contextmanager
def browser_session(args):
    artifacts = args.artifacts or Path(tempfile.mkdtemp(prefix="impact-e2e-"))
    artifacts.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {artifacts.resolve()}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed, channel=args.browser_channel)
        context = browser.new_context(viewport={"width": 1600, "height": 1000}, locale="en-US")
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.set_default_timeout(30000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(args.base_url, wait_until="domcontentloaded")
            expect(page.get_by_test_id("queue-button")).to_be_visible(timeout=60000)
            yield ComfyUI(page, args)
        finally:
            already_failed = sys.exc_info()[0] is not None
            cleanup_error = None
            for name, action in (
                ("screenshot", lambda: page.screenshot(path=str(artifacts / "final.png"))),
                ("trace", lambda: context.tracing.stop(path=str(artifacts / "trace.zip"))),
                ("browser close", browser.close),
            ):
                try:
                    action()
                except (PlaywrightError, OSError) as error:
                    logging.warning("[Impact Pack E2E] %s failed: %s", name, error)
                    if cleanup_error is None:
                        cleanup_error = error
            if not already_failed:
                assert not errors, f"Uncaught browser JavaScript errors: {errors}"
                if cleanup_error is not None:
                    raise cleanup_error
