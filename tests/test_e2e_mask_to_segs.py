"""Bicubic mask execution and native workflow restoration through Chrome.

Use an isolated ComfyUI server with Impact Pack; no models are required.
The reference branch uses core MaskComposite to clamp the same resized mask.
"""
from argparse import Namespace
import json
import os
from pathlib import Path

import pytest

from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, setting, show_node, widget_state


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_bicubic_mask_pixels_and_native_widgets(tmp_path, restore_by_name):
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / str(restore_by_name))
    prompt = {
        "base": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}},
        "patch": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 32, "height": 32}},
        "pattern": {"class_type": "MaskComposite", "inputs": {
            "destination": ["base", 0], "source": ["patch", 0], "x": 16, "y": 16, "operation": "add"}},
        "image": {"class_type": "MaskToImage", "inputs": {"mask": ["pattern", 0]}},
        "scale": {"class_type": "ImageScaleBy", "inputs": {
            "image": ["image", 0], "upscale_method": "bicubic", "scale_by": 4.0}},
        "mask": {"class_type": "ImageToMask", "inputs": {"image": ["scale", 0], "channel": "red"}},
        "zero": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 256, "height": 256}},
        "clamp": {"class_type": "MaskComposite", "inputs": {
            "destination": ["mask", 0], "source": ["zero", 0], "x": 0, "y": 0, "operation": "add"}},
    }
    expected_widgets = [{"name": name, "value": value} for name, value in [
        ("combined", False), ("crop_factor", 2.2), ("bbox_fill", False),
        ("drop_size", 7), ("contour_fill", True),
    ]]
    for branch, source in [("actual", "mask"), ("reference", "clamp")]:
        prompt.update({
            f"{branch}_segs": {"class_type": "MaskToSEGS", "inputs": {
                "mask": [source, 0], **{widget["name"]: widget["value"] for widget in expected_widgets}}},
            f"{branch}_mask": {"class_type": "SegsToCombinedMask", "inputs": {"segs": [f"{branch}_segs", 0]}},
            f"{branch}_image": {"class_type": "MaskToImage", "inputs": {"mask": [f"{branch}_mask", 0]}},
            f"{branch}_preview": {"class_type": "PreviewImage", "inputs": {"images": [f"{branch}_image", 0]}},
        })
    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        ui.new_workflow()
        ui.load_workflow({"name": "bicubic-mask-api.json", "mimeType": "application/json",
                          "buffer": json.dumps(prompt).encode()}, len(prompt))
        before = {}
        for node_id in ("actual_segs", "reference_segs"):
            show_node(ui, node_id)
            before[node_id] = widget_state(ui, node_id)
            # Preserve JSON types: Python equality alone accepts False == 0.
            assert json.dumps(before[node_id]["widgets"], sort_keys=True) == json.dumps(expected_widgets, sort_keys=True)
        expected_values = {node_id: [widget["value"] for widget in expected_widgets] for node_id in before}
        workflow = args.artifacts / "bicubic-mask-native.json"
        export_workflow(ui, workflow)
        saved = json.loads(workflow.read_text())
        values = {str(node["id"]): node["widgets_values"] for node in saved["nodes"] if str(node["id"]) in before}
        assert json.dumps(values, sort_keys=True) == json.dumps(expected_values, sort_keys=True)
        previous_outputs = None
        reload_scale = 3.0 if restore_by_name else 2.0
        for reload_workflow in (False, True):
            if reload_workflow:
                ui.new_workflow()
                ui.load_workflow(workflow, len(prompt))
                for node_id, state in before.items():
                    show_node(ui, node_id)
                    assert json.dumps(widget_state(ui, node_id), sort_keys=True) == json.dumps(state, sort_keys=True)
                reexported = args.artifacts / "bicubic-mask-reloaded.json"
                export_workflow(ui, reexported)
                restored = json.loads(reexported.read_text())
                values = {str(node["id"]): node["widgets_values"] for node in restored["nodes"] if str(node["id"]) in before}
                assert json.dumps(values, sort_keys=True) == json.dumps(expected_values, sort_keys=True)
                show_node(ui, "scale")
                ui.set_number("ImageScaleBy", "scale_by", reload_scale)
            entry = ui.run()
            (args.artifacts / f"execution-{int(reload_workflow)}.json").write_text(json.dumps(entry, indent=2) + "\n")
            if reload_workflow:
                assert entry["prompt"][2]["scale"]["inputs"]["scale_by"] == reload_scale
                cached = {node for event, data in entry["status"]["messages"]
                          if event == "execution_cached" for node in data["nodes"]}
                changed = {"scale", "mask", "clamp", "actual_segs", "reference_segs",
                           "actual_mask", "reference_mask", "actual_image", "reference_image",
                           "actual_preview", "reference_preview"}
                assert cached.isdisjoint(changed), f"Changed mask path reused execution cache: {cached & changed}"
                for node_id in ("actual_preview", "reference_preview"):
                    filename = entry["outputs"][node_id]["images"][0]["filename"]
                    assert filename != previous_outputs[node_id]["images"][0]["filename"]
            ui.page.get_by_role("button", name="Fit View (.)", exact=True).click()
            images = ui.images(entry)
            actual = images["actual_preview"][0]
            reference = images["reference_preview"][0]
            size = int(64 * reload_scale) if reload_workflow else 256
            assert actual.size == reference.size == (size, size)
            if not reload_workflow:
                # Pixel baseline: PyTorch 2.10.0 and OpenCV 4.13.0.
                # Investigate dependency changes before updating these expectations.
                assert reference.getbbox() == (59, 59, 197, 197)
                assert reference.convert("L").histogram()[255] == 17460
            pixels_match = actual.tobytes() == reference.tobytes()
            assert pixels_match, "Bicubic mask contour includes wrapped out-of-range pixels"
            previous_outputs = entry["outputs"]
