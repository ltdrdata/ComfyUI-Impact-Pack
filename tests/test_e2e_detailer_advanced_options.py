"""Load embedded-workflow PNGs and run the UI-created Detailer tutorials.

Use an isolated --cache-none ComfyUI server. IMPACT_TUTORIALS_DIR points to the
companion tutorial checkout; its original example image must have been uploaded.
No model, sampler or VAE is replaced in these tests.
"""
from argparse import Namespace
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from detailer_hook_cases import expected_widgets, targets
from e2e_ui import browser_session
from test_e2e_widget_catalog import export_workflow, full_zoom, setting, show_node, widget_state


def tutorial_png(relative_png, workflow_name):
    root = Path(os.environ["IMPACT_TUTORIALS_DIR"]) / "ComfyUI-Impact-Pack"
    path = root / relative_png
    exported = (root / "workflow" / f"{workflow_name}.json").read_text()
    with Image.open(path) as image:
        assert image.info["workflow"] == exported, "PNG differs from the trusted native Export"
    return path, json.loads(exported)


def displayed_images(ui, entry):
    # The frontend starts loading previews when their nodes are rendered.
    for node_id, output in entry["outputs"].items():
        if output.get("images"):
            show_node(ui, node_id)
    return ui.images(entry)


def workflow_node_id(workflow, kind):
    node, = [n for n in workflow["nodes"] if n["type"] == kind]
    return str(node["id"])


def preview_node_id(workflow, source):
    previews = {n["id"] for n in workflow["nodes"] if n["type"] == "PreviewImage"}
    target, = [link[3] for link in workflow["links"]
               if str(link[1]) == source and link[2] == 0 and link[3] in previews]
    return str(target)


def assert_widgets(ui, expected):
    for node_id, widgets in expected.items():
        show_node(ui, node_id)
        assert json.dumps(widget_state(ui, node_id)["widgets"], sort_keys=True) == json.dumps(widgets, sort_keys=True)


def export_and_reload(ui, path, original, expected):
    export_workflow(ui, path)
    saved = json.loads(path.read_text())
    assert sorted(saved["links"]) == sorted(original["links"])
    assert sorted((n["id"], n["type"]) for n in saved["nodes"]) == sorted((n["id"], n["type"]) for n in original["nodes"])
    for node in saved["nodes"]:
        if str(node["id"]) in expected:
            assert json.dumps(node.get("widgets_values") or []) == json.dumps([w["value"] for w in expected[str(node["id"])]])
    ui.new_workflow()
    ui.load_workflow(path, len(original["nodes"]))
    full_zoom(ui)
    assert_widgets(ui, expected)


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_cycle_selection_and_final_image_survive_native_reload(tmp_path, restore_by_name):
    path, original = tutorial_png("workflow/detailer-advanced-options.png", "detailer-advanced-options")
    options_id = workflow_node_id(original, "DetailerAdvancedOptionsHook")
    selection_id = workflow_node_id(original, "ImageFromBatch")
    cycles_preview = preview_node_id(original, workflow_node_id(original, "MaskDetailerPipe"))
    selection_preview = preview_node_id(original, selection_id)
    assert next(n for n in original["nodes"] if str(n["id"]) == options_id)["widgets_values"] == [True, "inherit", "inherit", 512, 64, False]
    expected = {str(n["id"]): expected_widgets(n) for n in targets(original)}
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=600,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / f"advanced-{restore_by_name}")
    with browser_session(args) as ui:
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        ui.load_workflow(path, len(original["nodes"]))
        full_zoom(ui)
        assert_widgets(ui, expected)

        show_node(ui, options_id)
        ui.page.mouse.click(*ui.point("DetailerAdvancedOptionsHook", widget="return_by_cycle_step"))
        show_node(ui, selection_id)
        ui.set_number("ImageFromBatch", "batch_index", 0)
        baseline = ui.run()
        single = displayed_images(ui, baseline)
        assert len(single[cycles_preview]) == len(single[selection_preview]) == 1
        assert single[cycles_preview][0].tobytes() == single[selection_preview][0].tobytes()

        show_node(ui, options_id)
        ui.page.mouse.click(*ui.point("DetailerAdvancedOptionsHook", widget="return_by_cycle_step"))
        show_node(ui, selection_id)
        ui.set_number("ImageFromBatch", "batch_index", 1)
        cycled = ui.run()
        images = displayed_images(ui, cycled)
        assert len(images[cycles_preview]) == 3 and len(images[selection_preview]) == 1
        assert all(im.size == (512, 512) for im in images[cycles_preview])
        assert images[cycles_preview][-1].tobytes() == single[cycles_preview][0].tobytes(), "Final cycle changed when collecting intermediates"
        assert images[selection_preview][0].tobytes() == images[cycles_preview][1].tobytes(), "ImageFromBatch did not select cycle 2"
        assert len({im.tobytes() for im in images[cycles_preview]}) == 3, "Repeated final image instead of cycle snapshots"
        for index, im in enumerate(images[cycles_preview]):
            im.save(args.artifacts / f"cycle-{index + 1}.png")
        single[cycles_preview][0].save(args.artifacts / "disabled.png")

        export_and_reload(ui, args.artifacts / "export.json", original, expected)
        reloaded = ui.run()
        restored = displayed_images(ui, reloaded)
        assert [im.tobytes() for im in restored[cycles_preview]] == [im.tobytes() for im in images[cycles_preview]]
        assert restored[selection_preview][0].tobytes() == images[cycles_preview][1].tobytes()
        for name, entry in [("disabled", baseline), ("cycles", cycled), ("reloaded", reloaded)]:
            (args.artifacts / f"{name}-history.json").write_text(json.dumps(entry, indent=2) + "\n")


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["normal-positional", "inpaint-named"])
def test_tiled_vae_with_cycle_selection(tmp_path, restore_by_name):
    path, original = tutorial_png("workflow/detailer-advanced-options-vae.png", "detailer-advanced-options-vae")
    options_id = workflow_node_id(original, "DetailerAdvancedOptionsHook")
    detailer_id = workflow_node_id(original, "MaskDetailerPipe")
    cycles_preview = preview_node_id(original, detailer_id)
    selected_preview = preview_node_id(original, workflow_node_id(original, "ImageFromBatch"))
    expected = {str(n["id"]): expected_widgets(n) for n in targets(original)}
    overrides = {options_id: {"vae_tile_overlap": 256}, detailer_id: {"inpaint_model": restore_by_name}}
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=600,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / f"vae-{restore_by_name}")
    with browser_session(args) as ui:
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        ui.load_workflow(path, len(original["nodes"]))
        full_zoom(ui)
        assert_widgets(ui, expected)
        for node_id, widgets in expected.items():
            for widget in widgets:
                if widget["name"] in overrides.get(node_id, {}):
                    widget["value"] = overrides[node_id][widget["name"]]
        show_node(ui, options_id)
        assert [w["value"] for w in widget_state(ui, options_id)["widgets"]] == [True, "tiled", "tiled", 512, 64, False]
        ui.set_number("DetailerAdvancedOptionsHook", "vae_tile_overlap", 256)
        if restore_by_name:
            show_node(ui, detailer_id)
            ui.page.mouse.click(*ui.point("MaskDetailerPipe", widget="inpaint_model"))
        tiled = ui.run()
        images = displayed_images(ui, tiled)
        assert len(images[cycles_preview]) == 3
        assert len({im.tobytes() for im in images[cycles_preview]}) == 3
        assert images[selected_preview][0].tobytes() == images[cycles_preview][1].tobytes()
        for index, im in enumerate(images[cycles_preview]):
            im.save(args.artifacts / f"cycle-{index + 1}.png")
        export_and_reload(ui, args.artifacts / "export.json", original, expected)
        reloaded = ui.run()
        after = displayed_images(ui, reloaded)
        assert [im.tobytes() for im in after[cycles_preview]] == [im.tobytes() for im in images[cycles_preview]]
        assert after[selected_preview][0].tobytes() == images[cycles_preview][1].tobytes()
        for name, entry in [("tiled", tiled), ("reloaded", reloaded)]:
            (args.artifacts / f"{name}-history.json").write_text(json.dumps(entry, indent=2) + "\n")


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("detail", ["hooks", "selection"])
def test_detail_reference_png_survives_native_reload(tmp_path, restore_by_name, detail):
    path, original = tutorial_png(f"images/detailer-advanced-options-{detail}.png", "detailer-advanced-options")
    expected = {str(n["id"]): expected_widgets(n) for n in targets(original)}
    args = Namespace(base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"),
                     browser_channel="chrome", headed=False, timeout=30,
                     artifacts=Path(os.environ.get("IMPACT_E2E_ARTIFACTS", str(tmp_path))) / f"png-{detail}-{restore_by_name}")
    with browser_session(args) as ui:
        setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
        ui.load_workflow(path, len(original["nodes"]))
        full_zoom(ui)
        assert_widgets(ui, expected)
        export_and_reload(ui, args.artifacts / "export.json", original, expected)
