"""Original SAM tutorial PNG execution and Mask Pointer JSON restoration.

Set IMPACT_TUTORIALS_DIR to the tutorial checkout and IMPACT_E2E_TUTORIAL_IMAGE
to its detectors-sam.png workflow's original ComfyUI_temp_xltgv_00001_.png input.
Requires Impact Subpack, face_yolov8m.pt, SAM ViT-B and --cache-none.
Mask Pointer's external painted Clipspace mask is not bundled; that JSON case
checks native loading/serialization only and does not claim diffusion execution.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from e2e_ui import ui
from test_e2e_legacy_sam import source_image
from test_e2e_sam_masks import assert_saved_controls, controls, saved_values
from test_e2e_widget_catalog import export_workflow, setting


@pytest.mark.parametrize("named", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("filename,execute", [
    ("images/detectors-sam.png", True), ("workflow/maskpointer.json", False),
], ids=["detector-png", "mask-pointer-json"])
def test_original_sam_tutorial(ui, named, filename, execute):
    browser, artifacts = ui
    path = Path(os.environ["IMPACT_TUTORIALS_DIR"]) / "ComfyUI-Impact-Pack" / filename
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".png":
        with Image.open(path) as image:
            original = json.loads(image.info["workflow"])
    else:
        original = json.loads(path.read_text())
    relevant = [node for node in original["nodes"] if node["type"] in {
        "SAMLoader", "SAMDetectorCombined", "MaskToSEGS", "BboxDetectorSEGS", "UltralyticsDetectorProvider"}]
    node_ids = [str(node["id"]) for node in relevant]
    input_source = None
    if execute:
        input_path = Path(os.environ["IMPACT_E2E_TUTORIAL_IMAGE"])
        input_id = next(str(node["id"]) for node in original["nodes"] if node["type"] == "LoadImage")
        browser.new_workflow()
        source = source_image(browser, input_path)
        uploaded_name = next(widget["value"] for widget in controls(browser, [source])[0]["widgets"]
                             if widget["name"] == "image")
        input_source = {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "uploaded_name": uploaded_name}
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", named)
    browser.new_workflow()
    browser.load_workflow(path, len(original["nodes"]), check_node_errors=execute)
    if execute:
        browser.choose("LoadImage", "image", uploaded_name)
    before = controls(browser, node_ids)
    trusted = json.loads((Path(__file__).parent / "workflows/all-impact-nodes.widgets.json").read_text())["nodes"]
    names = {node["type"]: [widget["name"] for widget in node["widgets"]] for node in trusted}
    # Subpack nodes are outside the Impact Pack catalog.
    names["UltralyticsDetectorProvider"] = ["model_name"]
    for node, actual in zip(relevant, before, strict=True):
        assert [widget["name"] for widget in actual["widgets"]] == names[node["type"]]
        saved = node["widgets_values"]
        assert json.dumps([widget["value"] for widget in actual["widgets"]][:len(saved)]) == json.dumps(saved)
    results = []
    previous_files = None
    for phase in ("before", "reloaded"):
        if phase == "reloaded":
            browser.new_workflow()
            browser.load_workflow(artifacts / "workflow.json", len(original["nodes"]), check_node_errors=execute)
            assert json.dumps(controls(browser, node_ids)) == json.dumps(before)
        if execute:
            entry = browser.run()
            assert entry["prompt"][2][input_id]["inputs"]["image"] == uploaded_name, (
                "Tutorial executed a different image than the uploaded source")
            events = [data["nodes"] for event, data in entry["status"]["messages"] if event == "execution_cached"]
            assert events and not any(events), "Run with --cache-none to execute SAM and BBOX on every load"
            images = browser.images(entry)
            assert images.keys() == {"7", "25"}, "Both BBOX and SAM previews must execute"
            filenames = {node: value["images"] for node, value in entry["outputs"].items()}
            if previous_files is not None:
                assert all(filenames[node] != previous_files[node] for node in filenames)
            previous_files = filenames
            hashes = {}
            for node, batch in images.items():
                assert batch
                hashes[node] = []
                for index, image in enumerate(batch):
                    pixels = np.asarray(image)
                    assert np.any(pixels) and not np.all(pixels), "Tutorial mask must be non-degenerate"
                    image.save(artifacts / f"{phase}-{node}-{index}.png")
                    hashes[node].append(hashlib.sha256(image.tobytes()).hexdigest())
            assert len(images["25"]) == 1
            bbox_union = np.maximum.reduce([np.asarray(image) for image in images["7"]])
            assert np.any(np.asarray(images["25"][0]) != bbox_union), "SAM must refine the rectangular BBOX masks"
            results.append({"phase": phase, "prompt_id": entry["prompt"][1], "pixel_sha256": hashes})
        assert json.dumps(controls(browser, node_ids)) == json.dumps(before)
        destination = artifacts / ("workflow.json" if phase == "before" else "reloaded-workflow.json")
        export_workflow(browser, destination)
        assert_saved_controls(destination, before)
        if phase == "reloaded":
            assert json.dumps(saved_values(destination), sort_keys=True) == json.dumps(
                saved_values(artifacts / "workflow.json"), sort_keys=True)
        browser.page.screenshot(path=str(artifacts / f"{phase}.png"))
    if execute:
        assert results[0]["pixel_sha256"] == results[1]["pixel_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    (artifacts / "results.json").write_text(json.dumps({
        "source": str(path), "source_sha256": source_hash, "named_restore": named,
        "input_source": input_source, "executed": execute, "runs": results, "widgets_before": before,
        "widgets_after": controls(browser, node_ids),
    }, indent=2) + "\n")
