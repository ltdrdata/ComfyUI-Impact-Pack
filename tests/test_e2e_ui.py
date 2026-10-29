"""Browser cleanup regressions; requires Chrome, but no ComfyUI server."""
from argparse import Namespace
from zipfile import ZipFile

import pytest
from playwright.sync_api import BrowserContext, Error, Page

from e2e_ui import browser_session


@pytest.fixture
def browser_args(tmp_path, monkeypatch):
    new_page = BrowserContext.new_page

    def routed_page(context):
        page = new_page(context)

        def serve(route):
            if route.request.url.endswith("/scripts/app.js"):
                route.fulfill(content_type="application/javascript", body="export const app = {};")
            else:
                route.fulfill(content_type="text/html", body='<button data-testid="queue-button">Run</button>')

        page.route("**/*", serve)
        return page

    monkeypatch.setattr(BrowserContext, "new_page", routed_page)
    return Namespace(base_url="http://impact.test", browser_channel="chrome", headed=False,
                     timeout=5, artifacts=tmp_path)


def test_closed_page_preserves_original_failure_and_trace(browser_args, caplog):
    with pytest.raises(AssertionError, match="workflow failed"):
        with browser_session(browser_args) as ui:
            ui.page.close()
            raise AssertionError("workflow failed")

    assert "screenshot failed" in caplog.text
    with ZipFile(browser_args.artifacts / "trace.zip") as trace:
        assert any(name.endswith(".trace") for name in trace.namelist())


def test_artifact_failure_fails_an_otherwise_successful_run(browser_args, monkeypatch):
    def fail_screenshot(self, **kwargs):
        raise Error("screenshot unavailable")

    monkeypatch.setattr(Page, "screenshot", fail_screenshot)
    with pytest.raises(Error, match="screenshot unavailable"):
        with browser_session(browser_args):
            pass
    with ZipFile(browser_args.artifacts / "trace.zip") as trace:
        assert any(name.endswith(".trace") for name in trace.namelist())
