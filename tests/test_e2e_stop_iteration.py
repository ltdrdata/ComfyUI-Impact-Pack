"""Conditional Stop Iteration ends Run (Instant) and Run (On Change) in Chrome.

The condition is computed from a number that increments after every queued
prompt. That keeps Run (On Change) repeating, and the node runs instead of
being served from the execution cache.
"""
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from playwright.sync_api import expect

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
        # Long enough for dozens of repeats when the condition is ignored.
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
