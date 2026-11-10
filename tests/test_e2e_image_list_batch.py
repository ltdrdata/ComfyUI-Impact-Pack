"""Image list batching through native UI creation, execution and workflow reload."""
import json

import pytest

from e2e_ui import ui as ui
from test_e2e_widget_catalog import (
    assert_catalog, export_workflow, full_zoom, setting, show_node, widget_state,
)


def reload_and_check_workflow(browser, workflow, expected_widgets):
    saved = json.loads(workflow.read_text())
    browser.new_workflow()
    browser.load_workflow(workflow, len(expected_widgets))
    assert_catalog(browser, expected_widgets)
    roundtrip = workflow.with_stem("roundtrip")
    export_workflow(browser, roundtrip)
    restored = json.loads(roundtrip.read_text())
    for field, default in (("widgets_values", []), ("widgets_values_named", {})):
        before = {n["id"]: n.get(field, default) for n in saved["nodes"]}
        after = {n["id"]: n.get(field, default) for n in restored["nodes"]}
        # Keep bool/number distinctions that Python's dictionary equality loses.
        assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
    assert sorted(restored["links"]) == sorted(saved["links"])


def run_and_record(browser, artifacts, phase):
    with browser.page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/prompt")) as submitted:
        entry = browser.run()
    request = submitted.value.post_data_json
    (artifacts / f"{phase}-request.json").write_text(json.dumps(request, indent=2) + "\n")
    (artifacts / f"{phase}-history.json").write_text(json.dumps(entry, indent=2) + "\n")
    return entry, request["prompt"]


def assert_uncached(entry, *node_ids):
    events = [data["nodes"] for event, data in entry["status"]["messages"] if event == "execution_cached"]
    assert len(events) == 1, entry["status"]
    assert set(node_ids).isdisjoint(map(str, events[0]))


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("mask_size", [64, 128], ids=["uniform", "resize"])
def test_batch_pixels_order_and_native_workflow_compatibility(ui, restore_by_name, mask_size):
    browser, artifacts = ui
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    browser.new_workflow()
    full_zoom(browser)
    response = browser.page.request.get(f"{browser.base_url}/object_info")
    assert response.ok
    definitions = response.json()
    node_ids = []
    for node_type, x, y in [
        ("EmptyImage", 100, 170), ("SolidMask", 100, 560), ("MaskToImage", 500, 560),
        ("ImpactMakeImageList", 650, 180), ("ImageListToImageBatch", 1050, 180),
        ("PreviewImage", 1100, 570),
    ]:
        node_ids.append(browser.add_node(definitions[node_type]["display_name"], node_type, x, y))
    batch_id, preview_id = node_ids[4:]
    browser.set_number("EmptyImage", "width", 64)
    browser.set_number("EmptyImage", "height", 64)
    browser.set_number("EmptyImage", "batch_size", 2)
    browser.page.mouse.click(*browser.point("EmptyImage", widget="color"))
    browser.page.locator('input[type="color"]').fill("#ff0000")
    browser.page.keyboard.press("Escape")
    browser.set_number("SolidMask", "width", mask_size)
    browser.set_number("SolidMask", "height", mask_size)
    browser.set_number("SolidMask", "value", 1)
    browser.connect("EmptyImage", "ImpactMakeImageList")
    browser.connect("SolidMask", "MaskToImage")
    browser.connect("MaskToImage", "ImpactMakeImageList", target_slot=1)
    browser.connect("ImpactMakeImageList", "ImageListToImageBatch")
    browser.connect("ImageListToImageBatch", "PreviewImage")

    expected_widgets = []
    for node_id in node_ids:
        show_node(browser, node_id)
        expected_widgets.append(widget_state(browser, node_id))
    assert expected_widgets[4]["widgets"] == []
    assert list(definitions["ImageListToImageBatch"]["input"]["required"]) == ["images"]
    workflow = artifacts / "native.json"
    export_workflow(browser, workflow)

    for phase in ("initial", "reloaded", "edited"):
        if phase == "reloaded":
            reload_and_check_workflow(browser, workflow, expected_widgets)
        elif phase == "edited":
            show_node(browser, node_ids[1])
            browser.set_number("SolidMask", "value", 0)
        show_node(browser, preview_id)
        entry, prompt = run_and_record(browser, artifacts, phase)
        assert set(prompt[batch_id]["inputs"]) == {"images"}
        mask_inputs = prompt[node_ids[1]]["inputs"]
        assert (mask_inputs["width"], mask_inputs["height"]) == (mask_size, mask_size)
        images = browser.images(entry)[preview_id]
        assert [image.size for image in images] == [(64, 64)] * 3
        last = (0, 0, 0) if phase == "edited" else (255, 255, 255)
        assert [image.getextrema() for image in images] == [
            ((255, 255), (0, 0), (0, 0)), ((255, 255), (0, 0), (0, 0)),
            tuple((value, value) for value in last),
        ]
        if phase == "edited":
            assert_uncached(entry, batch_id, preview_id)


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_empty_list_skips_preview_preserves_other_output_and_recovers(ui, restore_by_name):
    browser, artifacts = ui
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    browser.new_workflow()
    full_zoom(browser)
    response = browser.page.request.get(f"{browser.base_url}/object_info")
    assert response.ok
    definitions = response.json()
    node_ids = []
    for node_type, x, y in [
        ("ImpactMakeImageList", 300, 170), ("ImageListToImageBatch", 650, 170),
        ("PreviewImage", 1100, 170), ("EmptyImage", 300, 600), ("SaveImage", 1100, 600),
    ]:
        node_ids.append(browser.add_node(definitions[node_type]["display_name"], node_type, x, y))
    list_id, batch_id, preview_id, image_id, save_id = node_ids
    browser.set_number("EmptyImage", "width", 64)
    browser.set_number("EmptyImage", "height", 64)
    browser.connect("ImpactMakeImageList", "ImageListToImageBatch")
    browser.connect("ImageListToImageBatch", "PreviewImage")
    browser.connect("EmptyImage", "SaveImage")
    expected_widgets = []
    for node_id in node_ids:
        show_node(browser, node_id)
        expected_widgets.append(widget_state(browser, node_id))
    workflow = artifacts / "empty-list.json"
    export_workflow(browser, workflow)

    for phase in ("empty", "reloaded-empty", "populated"):
        if phase == "reloaded-empty":
            reload_and_check_workflow(browser, workflow, expected_widgets)
        elif phase == "populated":
            show_node(browser, image_id)
            browser.page.mouse.click(*browser.point("EmptyImage", widget="color"))
            browser.page.locator('input[type="color"]').fill("#336699")
            browser.page.keyboard.press("Escape")
            show_node(browser, list_id)
            browser.connect("EmptyImage", "ImpactMakeImageList")
        show_node(browser, preview_id if phase == "populated" else save_id)
        entry, prompt = run_and_record(browser, artifacts, phase)
        assert set(prompt[batch_id]["inputs"]) == {"images"}
        images = browser.images(entry)
        assert [image.size for image in images[save_id]] == [(64, 64)]
        color = (51, 102, 153) if phase == "populated" else (0, 0, 0)
        assert images[save_id][0].getextrema() == tuple((value, value) for value in color)
        if phase == "populated":
            assert [image.size for image in images[preview_id]] == [(64, 64)]
            assert images[preview_id][0].tobytes() == images[save_id][0].tobytes()
            assert_uncached(entry, list_id, batch_id, preview_id)
        else:
            assert prompt[list_id]["inputs"] == {}
            assert not entry["outputs"].get(preview_id, {}).get("images")
