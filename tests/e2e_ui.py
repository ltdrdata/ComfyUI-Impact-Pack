"""Shared Chrome UI interactions for the standalone E2E scripts.

Graph inspection is read-only. Nodes, widgets, links, workflow loading, and
execution are changed through browser input events, never app/graph API calls.
"""
import argparse
from contextlib import contextmanager
import logging
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit

from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright


def options(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-url", default="http://127.0.0.1:18188")
    parser.add_argument("--browser-channel", default="chrome")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


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
        search.fill(title)
        self.page.get_by_role("option", name=re.compile(rf"^{re.escape(title)}(?:\s|$)")).click()
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

    def connect(self, source, target, target_slot=0):
        self.page.mouse.move(*self.point(source))
        self.page.mouse.down()
        self.page.mouse.move(*self.point(target, slot=target_slot, is_input=True), steps=12)
        self.page.mouse.up()
        self.page.wait_for_function("""({app, type, slot}) => {
            return app.graph._nodes.find(n => n.type === type).inputs[slot].link != null;
        }""", arg={"app": self.app, "type": target, "slot": target_slot})

    def load_workflow(self, workflow, node_count):
        previous_nodes = self.app.evaluate_handle("app => app.graph._nodes.slice()")
        self.page.locator("#comfy-file-input").set_input_files(workflow)
        self.page.wait_for_function("""({app, count, previous}) => {
            return !app.configuringGraph && app.graph._nodes.length === count &&
                app.graph._nodes.every(n => !n.has_errors && !previous.includes(n));
        }""", arg={"app": self.app, "count": node_count, "previous": previous_nodes})
        previous_nodes.dispose()
        self.page.get_by_role("button", name="Fit View (.)", exact=True).click()

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
            assert not errors, f"Uncaught browser JavaScript errors: {errors}"
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
            if cleanup_error is not None and not already_failed:
                raise cleanup_error
