"""Native restoration with real node registration and delayed category responses.

Use an isolated server configured with tests/wildcards/lora_samples as its
custom_wildcards directory. No models or graph API mutations are required.
"""
from argparse import Namespace
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from playwright.sync_api import expect

from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, full_zoom, rendered, setting, show_node, widget_state


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_category_response_cannot_overwrite_native_saved_values(tmp_path, restore_by_name):
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / str(restore_by_name))
    with browser_session(args) as ui:
        page = ui.page
        page.keyboard.press("Escape")
        definition = page.request.get(f"{ui.base_url}/object_info/LoraWildcardSource").json()["LoraWildcardSource"]
        assert definition["input"]["required"]["category"][0] == ["aaa", "ink"], (
            "Configure the isolated server's custom_wildcards as tests/wildcards/lora_samples")
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        ui.new_workflow()
        full_zoom(ui)
        node_id = ui.add_node("Lora Wildcard Source / Modes", "LoraWildcardSource", 450, 180)
        ui.choose("LoraWildcardSource", "category", "ink")
        page.wait_for_function("""app => app.graph._nodes.find(n => n.type === 'LoraWildcardSource')
            .widgets.find(w => w.name === 'lora_wildcard').options.values.includes('Ink second')""", arg=ui.app)
        ui.choose("LoraWildcardSource", "lora_wildcard", "Ink second")
        ui.set_number("LoraWildcardSource", "strength", 1.25)
        ui.set_number("LoraWildcardSource", "seed", 731)
        expect(page.locator("textarea")).to_have_count(1)
        page.locator("textarea").fill("saved prompt, __color__")
        rendered(ui)
        before = widget_state(ui, node_id)
        assert [widget["name"] for widget in before["widgets"]] == [
            "category", "lora_wildcard", "mode", "strength", "base_prompt", "seed", "control_after_generate",
        ]
        workflow = args.artifacts / "saved-workflow.json"
        export_workflow(ui, workflow)

        pending = []
        page.route("**/impact/get_items?*", lambda route: pending.append(route))
        ui.load_workflow(workflow, 1)
        full_zoom(ui)
        show_node(ui, node_id)
        assert widget_state(ui, node_id) == before
        categories = {parse_qs(urlsplit(route.request.url).query)["category"][0]: route for route in pending}
        assert categories.keys() == {"aaa", "ink"}
        # The restored category completes first; the stale creation request
        # completes last and must not change names, order, values or options.
        for category in ("ink", "aaa"):
            route = categories[category]
            with page.expect_response(lambda response: response.url == route.request.url):
                route.fulfill(response=route.fetch())
            rendered(ui)
        page.wait_for_timeout(200)
        assert widget_state(ui, node_id) == before
        assert ui.app.evaluate("""app => Array.from(app.graph._nodes.find(n => n.type === 'LoraWildcardSource')
            .widgets.find(w => w.name === 'lora_wildcard').options.values)""") == ["Ink first", "Ink second"]
