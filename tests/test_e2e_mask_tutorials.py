"""Native PNG/JSON compatibility using the original extension tutorials.

Set IMPACT_TUTORIALS_DIR to a ComfyUI-extension-tutorials checkout.
This checks loading and saved widgets; it does not run model inference.
"""
from argparse import Namespace
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, setting, show_node, widget_state


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_native_mask_tutorial_widgets(tmp_path, restore_by_name):
    tutorials_dir = os.environ.get("IMPACT_TUTORIALS_DIR")
    if not tutorials_dir:
        pytest.fail("Set IMPACT_TUTORIALS_DIR to a ComfyUI-extension-tutorials checkout.", pytrace=False)
    tutorials = Path(tutorials_dir) / "ComfyUI-Impact-Pack/workflow"
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / f"tutorial-{restore_by_name}")
    names = ["combined", "crop_factor", "bbox_fill", "drop_size", "contour_fill"]
    cases = [
        ("mask_to_segs.png", 5, {"3": [False, 1, False, 50, False]}),
        # Four saved values: the absent contour_fill must retain its False default.
        ("maskpointer.json", 4, {"13": [True, 2, False, 10, False], "57": [False, 2, False, 10, False]}),
        ("segs-manipulation.png", 5, {"2": [False, 3, False, 10, False]}),
    ]
    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        for filename, saved_count, expected in cases:
            path = tutorials / filename
            if path.suffix == ".png":
                with Image.open(path) as image:
                    original = json.loads(image.info["workflow"])
            else:
                original = json.loads(path.read_text())
            original_values = {str(node["id"]): node["widgets_values"] for node in original["nodes"] if str(node["id"]) in expected}
            expected_original = {node_id: values[:saved_count] for node_id, values in expected.items()}
            assert json.dumps(original_values, sort_keys=True) == json.dumps(expected_original, sort_keys=True), (
                f"Tutorial saved controls changed: {filename}")
            for loading_original in (True, False):
                ui.new_workflow()
                # Missing media/models do not prevent checking the saved controls.
                ui.load_workflow(path, len(original["nodes"]), check_node_errors=False)
                for node_id, values in expected.items():
                    show_node(ui, node_id)
                    state = widget_state(ui, node_id)
                    assert state["type"] == "MaskToSEGS"
                    wanted = [{"name": name, "value": value} for name, value in zip(names, values)]
                    assert json.dumps(state["widgets"], sort_keys=True) == json.dumps(wanted, sort_keys=True)
                exported = args.artifacts / f"{Path(filename).stem}-{int(loading_original)}.json"
                export_workflow(ui, exported)
                saved = json.loads(exported.read_text())
                values = {str(node["id"]): node["widgets_values"] for node in saved["nodes"] if str(node["id"]) in expected}
                # Preserve JSON types: Python equality alone accepts False == 0.
                assert json.dumps(values, sort_keys=True) == json.dumps(expected, sort_keys=True)
                path = exported
