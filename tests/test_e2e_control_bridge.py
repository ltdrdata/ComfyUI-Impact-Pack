"""Control Bridge inside subgraphs bypasses the node behind it and queues the workflow again, in Chrome."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from test_e2e_switches import ui as ui

WORKFLOWS = Path(__file__).parent / "workflows" / "control-bridge"

BYPASSED = """app => {
    const found = [];
    const walk = (graph, path) => graph._nodes.forEach(node => {
        if (node.mode === 4) found.push(path + node.id);
        if (node.subgraph) walk(node.subgraph, `${path}${node.id}:`);
    });
    walk(app.graph, '');
    return found;
}"""
RAN = """app => { window.impactRan = []; window.impactListens ??= app.api.addEventListener('executing', event => event.detail == null || window.impactRan.push(String(event.detail))) ?? true; }"""
MODE = "({app, mode}) => app.graph._nodes.find(n => n.type === 'ImpactControlBridge').widgets.find(w => w.name === 'mode').value === mode"


@pytest.mark.parametrize("name,behind", [("in-subgraph", "22:12"), ("last-two-subgraphs-deep", "5")],
                         ids=["node_in_the_same_subgraph", "node_behind_two_subgraphs"])
def test_bridge_in_a_subgraph_bypasses_the_node_behind_it(ui, name, behind):
    browser, _ = ui
    page = browser.page
    workflow = WORKFLOWS / f"{name}.json"
    browser.new_workflow()
    browser.load_workflow(str(workflow), len(json.loads(workflow.read_text())["nodes"]))
    assert browser.app.evaluate(BYPASSED) == []

    prompts = []
    page.on("response", lambda response: prompts.append(response.json()["prompt_id"]) if response.request.method == "POST"
            and urlsplit(response.url).path in ("/prompt", "/api/prompt") else None)
    page.get_by_test_id("queue-button").click()

    def ended():
        history = [page.request.get(f"{browser.base_url}/history/{prompt}").json().get(prompt) for prompt in prompts]
        return [[kind for kind, _ in entry["status"]["messages"] if kind.startswith("execution_") and kind not in ("execution_start", "execution_cached")]
                for entry in history if entry], history

    for _ in range(150):
        page.wait_for_timeout(200)
        ends, history = ended()
        if ["execution_error"] in ends or len(ends) == 2:
            break
    assert ends == [["execution_interrupted"], ["execution_success"]], history[-1]["status"] if history and history[-1] else prompts
    assert behind in history[0]["prompt"][2] and behind not in history[1]["prompt"][2]
    assert browser.app.evaluate(BYPASSED) == [behind]


def test_a_stopped_bridge_does_not_run_the_nodes_before_it(ui):
    browser, _ = ui
    page = browser.page
    browser.new_workflow()
    browser.add_node("Empty Image", "EmptyImage", 150, 250)
    before = browser.add_node("Invert Image Colors", "ImageInvert", 450, 250)
    browser.add_node("Control Bridge", "ImpactControlBridge", 750, 250)
    preview = browser.add_node("Preview Image", "PreviewImage", 1050, 250)
    browser.set_number("EmptyImage", "width", 64 + uuid4().int % 1000)
    browser.connect("EmptyImage", "ImageInvert")
    browser.connect("ImageInvert", "ImpactControlBridge")
    browser.connect("ImpactControlBridge", "PreviewImage")

    page.mouse.click(*browser.point("ImpactControlBridge", widget="mode"))
    page.wait_for_function(MODE, arg={"app": browser.app, "mode": False})
    browser.app.evaluate(RAN)
    stopped = browser.run()
    assert preview not in stopped["outputs"]
    assert before not in page.evaluate("window.impactRan")

    page.mouse.click(*browser.point("ImpactControlBridge", widget="mode"))
    page.wait_for_function(MODE, arg={"app": browser.app, "mode": True})
    browser.app.evaluate(RAN)
    active = browser.run()
    assert len(active["outputs"][preview]["images"]) == 1
    assert before in page.evaluate("window.impactRan")
