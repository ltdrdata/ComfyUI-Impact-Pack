"""Control Bridge inside subgraphs bypasses the node behind it and queues the workflow again, in Chrome."""
import json
from pathlib import Path
from urllib.parse import urlsplit

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
