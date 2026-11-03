"""Chrome queue/control interactions for constant-linked wildcard prompts.

Create and edit every graph through the UI; inspect real execution history and
websocket feedback. Encode cases use the checkpoint from the cache E2E suite.
"""
import json
import time
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from test_e2e_widget_catalog import export_workflow, full_zoom, setting, show_node, widget_state
from test_e2e_wildcard_cache import (
    NODE_TYPES, cached, graph, input_slot, native_roundtrip, set_text, run as run_preview,
)
from test_e2e_wildcard_cache import ui as ui  # shared Chrome fixture


def run(ui, preview_id, expected, label):
    show_node(ui, preview_id)
    return run_preview(ui, preview_id, expected, label)


def toggle(ui, node_type, widget, value):
    current = ui.app.evaluate("""(app, {type, widget}) =>
        app.graph._nodes.find(n => n.type === type).widgets.find(w => w.name === widget).value
    """, {"type": node_type, "widget": widget})
    if current != value:
        ui.page.mouse.click(*ui.point(node_type, widget=widget))
    ui.page.wait_for_function("""({app, type, widget, value}) =>
        app.graph._nodes.find(n => n.type === type).widgets.find(w => w.name === widget).value === value
    """, arg={"app": ui.app, "type": node_type, "widget": widget, "value": value})


def constant_graph(ui, node_type, text):
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "mode", "fixed")
    source_id = ui.add_node("Text", "PrimitiveString", 100, 650)
    set_text(ui, "PrimitiveString", "value", text)
    ui.connect("PrimitiveString", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    return node_id, preview_id, source_id


@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("field", ["populated_text", "wildcard_text"])
def test_cold_batch_reuses_constant_cache(ui, node_type, field):
    node_id, preview_id = graph(ui, node_type)
    ui.choose(node_type, "control_after_generate", "increment")
    ui.set_number(node_type, "seed", 731)
    text = f"cold batch {uuid4().hex}"
    ui.add_node("Text (Multiline)", "PrimitiveStringMultiline", 100, 650)
    ui.page.get_by_placeholder("value", exact=True).fill(text)
    ui.connect("PrimitiveStringMultiline", node_type, target_slot=input_slot(ui, node_type, field))
    native_roundtrip(ui, node_id)
    workflow = ui.artifacts / "saved.json"

    # Occupy the execution worker so both native batch requests arrive cold.
    ui.new_workflow()
    full_zoom(ui)
    ui.add_node("Text", "PrimitiveString", 100, 250)
    set_text(ui, "PrimitiveString", "value", uuid4().hex)
    ui.add_node("Sleep", "ImpactSleep", 650, 250)
    ui.set_number("ImpactSleep", "seconds", 20)
    ui.connect("PrimitiveString", "ImpactSleep")
    with ui.page.expect_response(lambda r: urlsplit(r.url).path in ("/prompt", "/api/prompt")
                                 and r.request.method == "POST") as submitted:
        ui.page.get_by_test_id("queue-button").click()
    assert submitted.value.ok
    blocker = submitted.value.json()["prompt_id"]
    ui.new_workflow()
    ui.load_workflow(workflow, len(json.loads(workflow.read_text())["nodes"]))
    full_zoom(ui)
    batch = ui.page.get_by_role("textbox", name="Batch Count", exact=True)
    batch.fill("2")
    batch.press("Enter")
    pending = []

    def observe_queue(response):
        if urlsplit(response.url).path in ("/prompt", "/api/prompt") and response.request.method == "POST":
            queue = ui.page.request.get(f"{ui.base_url}/queue").json()
            pending.append({"request": response.request.post_data_json, "queue": queue})
            assert queue["queue_running"][0][1] == blocker
            assert response.json()["prompt_id"] in [item[1] for item in queue["queue_pending"]]

    ui.page.on("response", observe_queue)
    try:
        entries = queue_runs(ui, 2, "cold-batch")
    finally:
        ui.page.remove_listener("response", observe_queue)
    (ui.artifacts / "cold-queue.json").write_text(json.dumps(pending, indent=2) + "\n")
    assert len(pending) == 2
    assert [item["request"]["prompt"][node_id]["inputs"]["seed"] for item in pending] == [731, 732]
    for entry in entries:
        assert entry["status"]["status_str"] == "success"
        assert entry["outputs"][preview_id]["text"] == [text]
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == 0
    assert {node_id, preview_id}.isdisjoint(cached(entries[0]))
    assert {node_id, preview_id} <= cached(entries[1])
    export_workflow(ui, ui.artifacts / "after-batch.json")


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("upstream", [True, False], ids=["before_wildcard", "after_wildcard"])
def test_control_bridge_stop_with_constant(ui, node_type, upstream, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    text = f"bridge {uuid4().hex}"
    node_id, preview_id, source_id = constant_graph(ui, node_type, text)
    ui.choose(node_type, "control_after_generate", "increment")
    ui.set_number(node_type, "seed", 731)
    bridge_id = ui.add_node("Control Bridge", "ImpactControlBridge", 950, 700)
    if upstream:
        ui.connect("PrimitiveString", "ImpactControlBridge")
        ui.connect("ImpactControlBridge", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    else:
        ui.connect(node_type, "ImpactControlBridge", source_slot=3 if node_type == "ImpactWildcardEncode" else 0)
        ui.connect("ImpactControlBridge", "PreviewAny")
    native_roundtrip(ui, node_id, bridge_id)
    for index in range(2):
        entry = run(ui, preview_id, text, f"active-{index}")
        inputs = entry["prompt"][2][node_id]["inputs"]
        # Fixed mode pauses the selected increment policy; direct constants
        # still use the backend's normalized static cache key.
        assert inputs["seed"] == (731 if upstream else 0)
        assert inputs["populated_text"] == [bridge_id if upstream else source_id, 0]
        if index:
            assert node_id in cached(entry)
        show_node(ui, node_id)
        values = {widget["name"]: widget["value"] for widget in widget_state(ui, node_id)["widgets"]}
        assert (values["seed"], values["control_after_generate"]) == (731, "increment")

    show_node(ui, bridge_id)
    toggle(ui, "ImpactControlBridge", "mode", False)
    for label in ("stopped", "stopped-new-text"):
        if label == "stopped-new-text":
            show_node(ui, source_id)
            set_text(ui, "PrimitiveString", "value", text + " changed")
        entry = ui.run()
        (ui.artifacts / f"{label}.json").write_text(json.dumps(entry, indent=2) + "\n")
        assert preview_id not in entry["outputs"]
        assert preview_id not in cached(entry)
    native_roundtrip(ui, node_id, bridge_id)
    show_node(ui, bridge_id)
    toggle(ui, "ImpactControlBridge", "mode", True)
    run(ui, preview_id, text + " changed", "reactivated")


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_switch_constant_is_not_folded(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    text = f"switch {uuid4().hex}"
    node_id, preview_id, _ = constant_graph(ui, node_type, text)
    other_id = ui.add_node("Text (Multiline)", "PrimitiveStringMultiline", 100, 850)
    show_node(ui, other_id)
    ui.page.get_by_placeholder("value", exact=True).fill("{red|green|blue}")
    show_node(ui, node_id)
    switch_id = ui.add_node("Switch (Any)", "ImpactSwitch", 950, 700)
    ui.connect("PrimitiveString", "ImpactSwitch", target_slot=input_slot(ui, "ImpactSwitch", "input1"))
    ui.connect("PrimitiveStringMultiline", "ImpactSwitch", target_slot=input_slot(ui, "ImpactSwitch", "input2"))
    ui.connect("ImpactSwitch", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    for select_on_prompt in (False, True):
        show_node(ui, switch_id)
        toggle(ui, "ImpactSwitch", "sel_mode", select_on_prompt)
        native_roundtrip(ui, node_id, switch_id)
        for select, seed, expected in ((1, 2, text), (2, 2, "red"), (2, 5, "blue"), (1, 5, text)):
            show_node(ui, switch_id)
            ui.set_number("ImpactSwitch", "select", select)
            show_node(ui, node_id)
            ui.set_number(node_type, "seed", seed)
            entry = run(ui, preview_id, expected, f"switch-{select_on_prompt}-{select}-{seed}")
            assert entry["prompt"][2][node_id]["inputs"]["seed"] == seed
            assert entry["prompt"][2][node_id]["inputs"]["populated_text"] == [switch_id, 0]


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_remote_seed_precedes_constant_check(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id, source_id = constant_graph(ui, node_type, "{red|green|blue}")
    remote_id = ui.add_node("Remote Int (on prompt)", "ImpactRemoteInt", 950, 700)
    ui.set_number("ImpactRemoteInt", "node_id", int(node_id))
    set_text(ui, "ImpactRemoteInt", "widget_name", "seed")
    for seed, expected in ((2, "red"), (5, "blue")):
        show_node(ui, remote_id)
        ui.set_number("ImpactRemoteInt", "value", seed)
        entry = run(ui, preview_id, expected, f"remote-syntax-{seed}")
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == seed
    text = f"remote {uuid4().hex}"
    show_node(ui, source_id)
    set_text(ui, "PrimitiveString", "value", text)
    native_roundtrip(ui, node_id, remote_id)
    for index, seed in enumerate((731, 732)):
        show_node(ui, remote_id)
        ui.set_number("ImpactRemoteInt", "value", seed)
        entry = run(ui, preview_id, text, f"remote-plain-{seed}")
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == 0
        show_node(ui, node_id)
        assert next(w["value"] for w in widget_state(ui, node_id)["widgets"] if w["name"] == "seed") == seed
        if index:
            assert {node_id, preview_id} <= cached(entry)


def queue_runs(ui, count, label):
    """Click Run once and observe the frontend's bounded automatic requeues."""
    requests = []
    prompt_ids = []
    history = {}

    def is_submission(request):
        return urlsplit(request.url).path in ("/prompt", "/api/prompt") and request.method == "POST"

    def requested(request):
        if is_submission(request):
            requests.append(request)

    def submitted(response):
        if is_submission(response.request):
            assert response.ok, response.text()
            prompt_ids.append(response.json()["prompt_id"])

    ui.page.on("request", requested)
    ui.page.on("response", submitted)
    try:
        ui.page.get_by_test_id("queue-button").click()
        deadline = time.monotonic() + ui.timeout
        while time.monotonic() < deadline:
            ui.page.wait_for_timeout(100)
            for prompt_id in tuple(prompt_ids):
                if prompt_id not in history:
                    response = ui.page.request.get(f"{ui.base_url}/history/{prompt_id}")
                    assert response.ok, response.text()
                    history.update(response.json())
            queue = ui.page.request.get(f"{ui.base_url}/queue").json()
            frontend_idle = ui.app.evaluate("app => app.processingQueue === false && app.queueItems.length === 0")
            assert len(requests) <= count, f"Expected {count} submissions, observed {len(requests)}"
            if (len(prompt_ids) == count and all(key in history for key in prompt_ids)
                    and not any(queue.values()) and frontend_idle):
                entries = [history[key] for key in prompt_ids]
                (ui.artifacts / f"{label}.json").write_text(json.dumps(entries, indent=2) + "\n")
                return entries
        raise AssertionError(f"Expected {count} completed frontend submissions, got {prompt_ids}")
    finally:
        ui.page.remove_listener("request", requested)
        ui.page.remove_listener("response", submitted)


def test_queue_runs_rejects_extra_submission_before_response(ui):
    ui.add_node("Queue Trigger (Countdown)", "ImpactQueueTriggerCountdown", 400, 300)
    ui.set_number("ImpactQueueTriggerCountdown", "total", 4)
    requests = []
    held = []

    def delay_fourth(route):
        if route.request.method != "POST":
            route.continue_()
            return
        requests.append(route.request)
        if len(requests) == 4:
            held.append(route)
        else:
            route.continue_()

    ui.page.route("**/prompt", delay_fourth)
    try:
        with pytest.raises(AssertionError, match="Expected 3 submissions, observed 4"):
            queue_runs(ui, 3, "unexpected-fourth")
        assert len(held) == 1
    finally:
        for route in held:
            route.abort()
        ui.page.unroute("**/prompt", delay_fourth)


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
@pytest.mark.parametrize("behavior,node_mode,bypass_input", [
    pytest.param("Mute", 2, False, id="Mute"),
    pytest.param("Bypass", 4, False, id="Bypass-populated-only"),
    pytest.param("Bypass", 4, True, id="Bypass-connected-first-input"),
])
def test_control_bridge_mode_change_requeues(ui, node_type, behavior, node_mode, bypass_input, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    node_id, preview_id, _ = constant_graph(ui, node_type, "{red|green|blue}")
    ui.set_number(node_type, "seed", 2)
    bridge_id = ui.add_node("Control Bridge", "ImpactControlBridge", 950, 700)
    ui.choose("ImpactControlBridge", "behavior", behavior)
    ui.connect("PrimitiveString", "ImpactControlBridge")
    ui.connect("ImpactControlBridge", node_type, target_slot=input_slot(ui, node_type, "populated_text"))
    if bypass_input:
        # Native Bypass forwards the first matching input, wildcard_text.
        ui.connect("PrimitiveString", node_type, target_slot=input_slot(ui, node_type, "wildcard_text"))
    native_roundtrip(ui, node_id, bridge_id)
    run(ui, preview_id, "red", "active")
    show_node(ui, bridge_id)
    toggle(ui, "ImpactControlBridge", "mode", False)
    entries = queue_runs(ui, 2, "deactivate")
    assert any(event == "execution_interrupted" for event, _ in entries[0]["status"]["messages"])
    assert entries[1]["status"]["status_str"] == "success"
    assert node_id not in entries[1]["prompt"][2]
    assert ui.app.evaluate("(app, id) => app.graph.getNodeById(id).mode", node_id) == node_mode
    if behavior == "Mute":
        assert preview_id not in entries[1]["outputs"]
    else:
        if (node_type == "ImpactWildcardProcessor" and not bypass_input
                and entries[1]["prompt"][2][preview_id]["inputs"] == {}
                and preview_id not in entries[1]["outputs"]):
            pytest.xfail("Pre-existing on Main + #1269: Processor Bypass drops PreviewAny.source when only populated_text is linked")
        assert entries[1]["outputs"][preview_id]["text"] == ["{red|green|blue}"]
    native_roundtrip(ui, node_id, bridge_id)
    assert ui.app.evaluate("(app, id) => app.graph.getNodeById(id).mode", node_id) == node_mode
    show_node(ui, bridge_id)
    toggle(ui, "ImpactControlBridge", "mode", True)
    entries = queue_runs(ui, 2, "reactivate")
    assert node_id not in entries[0]["prompt"][2]
    # With a muted target, the bridge can be the last executable node: the
    # requeue still happens, but no following node polls the interrupt flag.
    if entries[0]["status"]["status_str"] != "success":
        assert any(event == "execution_interrupted" for event, _ in entries[0]["status"]["messages"])
    assert entries[1]["status"]["status_str"] == "success"
    assert entries[1]["outputs"][preview_id]["text"] == ["red"]
    assert entries[1]["prompt"][2][node_id]["inputs"]["seed"] == 2
    assert ui.app.evaluate("(app, id) => app.graph.getNodeById(id).mode", node_id) == 0


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_widget_feedback_and_countdown_requeue(ui, node_type, restore_by_name):
    setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    text = f"feedback {uuid4().hex}"
    node_id, preview_id, source_id = constant_graph(ui, node_type, text)
    ui.choose(node_type, "control_after_generate", "increment")
    ui.set_number(node_type, "seed", 731)
    replacement_id = ui.add_node("Text (Multiline)", "PrimitiveStringMultiline", 100, 850)
    show_node(ui, replacement_id)
    ui.page.get_by_placeholder("value", exact=True).fill(text + " changed")
    show_node(ui, node_id)
    setter_id = ui.add_node("Set Widget Value", "ImpactSetWidgetValue", 950, 650)
    ui.set_number("ImpactSetWidgetValue", "node_id", int(source_id))
    set_text(ui, "ImpactSetWidgetValue", "widget_name", "value")
    ui.connect(node_type, "ImpactSetWidgetValue", source_slot=3 if node_type == "ImpactWildcardEncode" else 0)
    ui.connect("PrimitiveStringMultiline", "ImpactSetWidgetValue",
               target_slot=input_slot(ui, "ImpactSetWidgetValue", "string_value"))
    countdown_id = ui.add_node("Queue Trigger (Countdown)", "ImpactQueueTriggerCountdown", 1300, 700)
    ui.set_number("ImpactQueueTriggerCountdown", "total", 3)
    ui.connect("ImpactSetWidgetValue", "ImpactQueueTriggerCountdown",
               target_slot=input_slot(ui, "ImpactQueueTriggerCountdown", "signal"))
    native_roundtrip(ui, node_id, setter_id, countdown_id)
    entries = queue_runs(ui, 3, "feedback-countdown")
    for index, (entry, expected) in enumerate(zip(entries, (text, text + " changed", text + " changed"))):
        assert entry["status"]["status_str"] == "success", entry["status"]
        assert entry["outputs"][preview_id]["text"] == [expected]
        assert entry["prompt"][2][source_id]["inputs"]["value"] == expected
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == 0
        assert entry["prompt"][2][countdown_id]["inputs"]["count"] == index
        assert (node_id in cached(entry)) == (index == 2)
        assert (preview_id in cached(entry)) == (index == 2)
    for target, widget, expected in ((source_id, "value", text + " changed"), (countdown_id, "count", 0)):
        show_node(ui, target)
        assert next(w["value"] for w in widget_state(ui, target)["widgets"] if w["name"] == widget) == expected

    # A runtime widget update can introduce wildcard syntax. The next queued
    # prompt must keep its requested seed instead of reusing the old static key.
    syntax = text + " {red|green|blue}"
    show_node(ui, replacement_id)
    ui.page.get_by_placeholder("value", exact=True).fill(syntax)
    show_node(ui, node_id)
    ui.choose(node_type, "control_after_generate", "fixed")
    ui.set_number(node_type, "seed", 2)
    entries = queue_runs(ui, 3, "feedback-syntax-countdown")
    for index, entry in enumerate(entries):
        assert entry["status"]["status_str"] == "success", entry["status"]
        assert entry["outputs"][preview_id]["text"] == [text + (" changed" if index == 0 else " red")]
        assert entry["prompt"][2][source_id]["inputs"]["value"] == (text + " changed" if index == 0 else syntax)
        assert entry["prompt"][2][node_id]["inputs"]["seed"] == (0 if index == 0 else 2)
        assert entry["prompt"][2][countdown_id]["inputs"]["count"] == index
        assert (node_id in cached(entry)) == (index != 1)
    native_roundtrip(ui, node_id, setter_id, countdown_id)
