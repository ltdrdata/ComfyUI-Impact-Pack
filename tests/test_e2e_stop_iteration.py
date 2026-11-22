from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from playwright.sync_api import expect

from e2e_ui import ComfyUI
from test_e2e_switches import ui as ui


def queue_mode(browser):
    return browser.app.evaluate("app => app.extensionManager.queueSettings.mode")


def select_run_mode(browser, name):
    page = browser.page
    page.get_by_test_id("queue-mode-menu-trigger").click()
    page.get_by_role("menuitemradio", name=name, exact=True).click()
    page.keyboard.press("Escape")
    expect(page.get_by_role("menuitemradio")).to_have_count(0)


@pytest.mark.parametrize("comparison,stops", [("a = b", True), ("a <> b", False)], ids=["condition_true", "condition_false"])
@pytest.mark.parametrize("run_mode,running,stopped", [
    ("Run (Instant)", "instant-running", "instant-idle"),
    ("Run (On Change)", "change", "disabled"),
], ids=["instant", "on_change"])
def test_condition_ends_auto_queue(ui, run_mode, running, stopped, comparison, stops):
    browser, _ = ui
    page = browser.page
    posted = []
    page.on("request", lambda request: posted.append(request.url) if request.method == "POST"
            and urlsplit(request.url).path in ("/prompt", "/api/prompt") else None)
    browser.new_workflow()
    browser.add_node("Int", "PrimitiveInt", 250, 300)
    browser.set_number("PrimitiveInt", "value", uuid4().int % 10**9)
    browser.choose("PrimitiveInt", "control_after_generate", "increment")
    browser.add_node("ImpactCompare", "ImpactCompare", 700, 300)
    browser.choose("ImpactCompare", "cmp", comparison)
    browser.add_node("ImpactConditionalStopIteration", "ImpactConditionalStopIteration", 1150, 300)
    browser.connect("PrimitiveInt", "ImpactCompare", target_slot=0)
    browser.connect("PrimitiveInt", "ImpactCompare", target_slot=1)
    browser.connect("ImpactCompare", "ImpactConditionalStopIteration")

    select_run_mode(browser, run_mode)
    page.get_by_test_id("queue-button").click()
    if stops:
        page.wait_for_timeout(4000)
        assert (len(posted), queue_mode(browser)) == (1, stopped)
    else:
        for _ in range(100):
            if len(posted) >= 3:
                break
            page.wait_for_timeout(100)
        assert len(posted) >= 3 and queue_mode(browser) == running
        select_run_mode(browser, "Run")
        assert queue_mode(browser) == "disabled"


def test_condition_leaves_auto_queue_in_another_tab_running(ui):
    browser, args = ui
    page = browser.page.context.new_page()
    page.set_default_timeout(10000)
    page.goto(args.base_url, wait_until="domcontentloaded")
    expect(page.get_by_test_id("queue-button")).to_be_visible(timeout=60000)
    page.keyboard.press("Escape")
    other = ComfyUI(page, args)
    posted = []
    page.on("request", lambda request: posted.append(request.url) if request.method == "POST"
            and urlsplit(request.url).path in ("/prompt", "/api/prompt") else None)
    other.new_workflow()
    other.add_node("Int", "PrimitiveInt", 250, 300)
    other.choose("PrimitiveInt", "control_after_generate", "increment")
    other.add_node("Preview as Text", "PreviewAny", 700, 300)
    other.connect("PrimitiveInt", "PreviewAny")
    select_run_mode(other, "Run (Instant)")
    page.get_by_test_id("queue-button").click()

    browser.page.bring_to_front()
    browser.new_workflow()
    browser.add_node("Int", "PrimitiveInt", 250, 300)
    browser.set_number("PrimitiveInt", "value", uuid4().int % 10**9)
    browser.add_node("ImpactCompare", "ImpactCompare", 700, 300)
    browser.add_node("ImpactConditionalStopIteration", "ImpactConditionalStopIteration", 1150, 300)
    browser.connect("PrimitiveInt", "ImpactCompare", target_slot=0)
    browser.connect("PrimitiveInt", "ImpactCompare", target_slot=1)
    browser.connect("ImpactCompare", "ImpactConditionalStopIteration")
    browser.run()

    before = len(posted)
    page.wait_for_timeout(2000)
    assert (len(posted) > before, queue_mode(other)) == (True, "instant-running")
    page.bring_to_front()
    select_run_mode(other, "Run")

