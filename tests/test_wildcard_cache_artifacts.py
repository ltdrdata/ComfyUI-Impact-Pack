"""Original prompt artifact regressions; no Chrome or ComfyUI server required."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

import test_e2e_wildcard_cache as cache


@pytest.fixture
def queued_run(tmp_path, monkeypatch):
    request = SimpleNamespace(method="POST", url="http://impact.test/prompt", post_data_json={
        "prompt": {"source": {"class_type": "ImpactWildcardProcessor", "inputs": {"seed": 5}}}})
    entry = {"outputs": {"preview": {"text": ["blue"]}},
             "prompt": [None, None, {"source": {"inputs": {"seed": 0}}}]}
    values = []
    preview_error = None

    class Page:
        @contextmanager
        def expect_request(self, predicate):
            assert predicate(request)
            yield SimpleNamespace(value=request)

        def get_by_role(self, role, *, name, exact):
            assert (role, name, exact) == ("textbox", "preview_text", True)
            return self

    def to_have_value(value):
        values.append(value)
        if preview_error is not None:
            raise preview_error

    monkeypatch.setattr(cache, "expect", lambda _: SimpleNamespace(to_have_value=to_have_value))
    ui = SimpleNamespace(page=Page(), artifacts=tmp_path, run=lambda: entry)

    def fail_preview(error):
        nonlocal preview_error
        preview_error = error

    return ui, request.post_data_json, entry, values, fail_preview


@pytest.mark.parametrize("failure", ["output", "preview"])
def test_original_post_survives_output_verification_failure(queued_run, failure):
    ui, request, entry, _, fail_preview = queued_run
    error = AssertionError("preview did not update")
    if failure == "preview":
        fail_preview(error)
    with pytest.raises(AssertionError) as caught:
        cache.run_submitted(ui, "preview", "green" if failure == "output" else "blue", "failed")
    if failure == "preview":
        assert caught.value is error
    assert json.loads((ui.artifacts / "failed-request.json").read_text()) == request
    assert json.loads((ui.artifacts / "failed.json").read_text()) == entry


def test_returns_the_original_seed_instead_of_normalized_history(queued_run):
    ui, request, entry, values, _ = queued_run
    actual, submitted = cache.run_submitted(ui, "preview", "blue", "success")
    assert actual is entry
    assert submitted == request["prompt"]
    assert submitted["source"]["inputs"]["seed"] == 5
    assert actual["prompt"][2]["source"]["inputs"]["seed"] == 0
    assert values == ["blue"]
    assert json.loads((ui.artifacts / "success-request.json").read_text()) == request
