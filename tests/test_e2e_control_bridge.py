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
MODE_OF = "({app, type}) => app.graph._nodes.find(n => n.type === type).mode"


def run_until_ended(browser, count):
    """Clicks Run and waits for the prompt and the ones the page queues after it. Returns how each ended, and their entries in the history."""
    page = browser.page
    prompts = []

    def queued(response):
        if response.request.method == "POST" and urlsplit(response.url).path in ("/prompt", "/api/prompt"):
            prompts.append(response.json()["prompt_id"])

    page.on("response", queued)
    page.get_by_test_id("queue-button").click()
    for _ in range(150):
        page.wait_for_timeout(200)
        history = [page.request.get(f"{browser.base_url}/history/{prompt}").json().get(prompt) for prompt in prompts]
        ends = [[kind for kind, _ in entry["status"]["messages"] if kind.startswith("execution_") and kind not in ("execution_start", "execution_cached")]
                for entry in history if entry]
        if ["execution_error"] in ends or len(ends) == count:
            break
    page.remove_listener("response", queued)
    return ends, history


def mute_by_hand(browser, node_type, muted):
    """Selects the node and presses the key of the page that mutes it or makes it active again."""
    arg = {"app": browser.app, "type": node_type, "mode": 2 if muted else 0}
    browser.page.mouse.click(*browser.page.evaluate("""({app, type}) => {
        const node = app.graph._nodes.find(n => n.type === type);
        return app.canvasPosToClientPos([node.pos[0] + 40, node.pos[1] - 12]);
    }""", arg))
    browser.page.keyboard.press("Control+m")
    browser.page.wait_for_function("({app, type, mode}) => app.graph._nodes.find(n => n.type === type).mode === mode", arg=arg)


def set_mode(browser, mode):
    browser.page.mouse.click(*browser.point("ImpactControlBridge", widget="mode"))
    browser.page.wait_for_function(MODE, arg={"app": browser.app, "mode": mode})


@pytest.mark.parametrize("name,behind", [("in-subgraph", "22:12"), ("last-two-subgraphs-deep", "5")],
                         ids=["node_in_the_same_subgraph", "node_behind_two_subgraphs"])
def test_bridge_in_a_subgraph_bypasses_the_node_behind_it(ui, name, behind):
    browser, _ = ui
    workflow = WORKFLOWS / f"{name}.json"
    browser.new_workflow()
    browser.load_workflow(str(workflow), len(json.loads(workflow.read_text())["nodes"]))
    assert browser.app.evaluate(BYPASSED) == []

    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history[-1]["status"] if history and history[-1] else history
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

    set_mode(browser, False)
    browser.app.evaluate(RAN)
    stopped = browser.run()
    assert preview not in stopped["outputs"]
    assert before not in page.evaluate("window.impactRan")

    set_mode(browser, True)
    browser.app.evaluate(RAN)
    active = browser.run()
    assert len(active["outputs"][preview]["images"]) == 1
    assert before in page.evaluate("window.impactRan")


@pytest.mark.parametrize("behavior,mode_behind", [("Mute", 2), ("Bypass", 4)])
def test_a_bridge_that_mutes_or_bypasses_does_not_run_the_nodes_before_it(ui, behavior, mode_behind):
    browser, _ = ui
    page = browser.page
    browser.new_workflow()
    browser.add_node("Empty Image", "EmptyImage", 150, 250)
    before = browser.add_node("Invert Image Colors", "ImageInvert", 450, 250)
    browser.add_node("Control Bridge", "ImpactControlBridge", 750, 250)
    preview = browser.add_node("Preview Image", "PreviewImage", 1050, 250)
    width = 64 + uuid4().int % 1000
    browser.set_number("EmptyImage", "width", width)
    browser.connect("EmptyImage", "ImageInvert")
    browser.connect("ImageInvert", "ImpactControlBridge")
    browser.connect("ImpactControlBridge", "PreviewImage")
    browser.choose("ImpactControlBridge", "behavior", behavior)

    set_mode(browser, False)
    browser.app.evaluate(RAN)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history
    assert page.evaluate(MODE_OF, {"app": browser.app, "type": "PreviewImage"}) == mode_behind
    assert before not in page.evaluate("window.impactRan")

    browser.set_number("EmptyImage", "width", width + 1)
    browser.app.evaluate(RAN)
    browser.run()
    assert before not in page.evaluate("window.impactRan")

    set_mode(browser, True)
    browser.app.evaluate(RAN)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history
    assert len(history[1]["outputs"][preview]["images"]) == 1
    assert page.evaluate("window.impactRan").count(before) == 1


def test_a_value_that_goes_on_through_a_bypassed_node_is_made_once(ui):
    browser, _ = ui
    page = browser.page
    browser.new_workflow()
    browser.add_node("Empty Image", "EmptyImage", 150, 250)
    before = browser.add_node("Invert Image Colors", "ImageInvert", 450, 250)
    browser.add_node("Control Bridge", "ImpactControlBridge", 750, 250)
    browser.add_node("Flip Image", "ImageFlip", 450, 600)
    preview = browser.add_node("Preview Image", "PreviewImage", 750, 600)
    browser.set_number("EmptyImage", "width", 64 + uuid4().int % 1000)
    browser.connect("EmptyImage", "ImageInvert")
    browser.connect("ImageInvert", "ImpactControlBridge")
    browser.connect("ImpactControlBridge", "ImageFlip")
    browser.connect("ImageFlip", "PreviewImage")
    browser.choose("ImpactControlBridge", "behavior", "Bypass")

    set_mode(browser, False)
    browser.app.evaluate(RAN)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history
    assert page.evaluate(MODE_OF, {"app": browser.app, "type": "ImageFlip"}) == 4
    assert len(history[1]["outputs"][preview]["images"]) == 1
    assert page.evaluate("window.impactRan").count(before) == 1


def test_a_node_behind_a_muting_bridge_that_is_made_active_by_hand_is_muted_again(ui):
    browser, _ = ui
    page = browser.page
    browser.new_workflow()
    browser.add_node("Empty Image", "EmptyImage", 150, 250)
    before = browser.add_node("Invert Image Colors", "ImageInvert", 450, 250)
    browser.add_node("Control Bridge", "ImpactControlBridge", 750, 250)
    browser.add_node("Preview Image", "PreviewImage", 1050, 250)
    browser.set_number("EmptyImage", "width", 64 + uuid4().int % 1000)
    browser.connect("EmptyImage", "ImageInvert")
    browser.connect("ImageInvert", "ImpactControlBridge")
    browser.connect("ImpactControlBridge", "PreviewImage")
    browser.choose("ImpactControlBridge", "behavior", "Mute")
    set_mode(browser, False)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history

    mute_by_hand(browser, "PreviewImage", False)
    browser.app.evaluate(RAN)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history
    assert page.evaluate(MODE_OF, {"app": browser.app, "type": "PreviewImage"}) == 2
    assert before not in page.evaluate("window.impactRan")


def test_a_bridge_that_makes_one_of_two_nodes_behind_it_active_gives_the_other_its_value(ui):
    browser, _ = ui
    page = browser.page
    browser.new_workflow()
    browser.add_node("Empty Image", "EmptyImage", 150, 250)
    before = browser.add_node("Invert Image Colors", "ImageInvert", 450, 250)
    browser.add_node("Control Bridge", "ImpactControlBridge", 750, 250)
    preview = browser.add_node("Preview Image", "PreviewImage", 1050, 250)
    browser.add_node("Flip Image", "ImageFlip", 750, 600)
    browser.set_number("EmptyImage", "width", 64 + uuid4().int % 1000)
    browser.connect("EmptyImage", "ImageInvert")
    browser.connect("ImageInvert", "ImpactControlBridge")
    browser.connect("ImpactControlBridge", "PreviewImage")
    browser.connect("ImpactControlBridge", "ImageFlip")
    browser.choose("ImpactControlBridge", "behavior", "Mute")
    mute_by_hand(browser, "ImageFlip", True)

    browser.app.evaluate(RAN)
    ends, history = run_until_ended(browser, 2)
    assert ends == [["execution_interrupted"], ["execution_success"]], history
    assert page.evaluate(MODE_OF, {"app": browser.app, "type": "ImageFlip"}) == 0
    assert len(history[1]["outputs"][preview]["images"]) == 1
    assert page.evaluate("window.impactRan").count(before) == 1
