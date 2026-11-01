"""Check Impact widget restoration in the four original AnimateDiff tutorial PNGs.

Set IMPACT_TUTORIALS_DIR to ComfyUI-extension-tutorials. This checks native
PNG import and JSON export/reload in both restoration modes, without inference.
Missing external nodes are recorded; the Impact nodes under test must be real.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from PIL import Image

from e2e_ui import browser_session, options
from test_e2e_widget_catalog import WIDGETS, export_workflow, full_zoom, setting, show_node, widget_state


WORKFLOWS = ["cinemagraph", "detailer-for-animatediff", "MakeTileSEGS_upscale_AnimateDiff",
             "MakeTileSEGS_upscale_AnimateDiff_ControlNet"]
TYPES = {"SEGSDetailerForAnimateDiff", "DetailerForEachPipeForAnimateDiff", "SEGSPaste",
         "MaskToSEGS", "ImpactSimpleDetectorSEGS_for_AD", "ImpactMakeTileSEGS",
         "ImpactControlNetApplySEGS", "PreviewDetailerHookProvider"}


def check_widgets(ui, expected):
    full_zoom(ui)
    for node in expected:
        show_node(ui, node["id"])
        actual = widget_state(ui, node["id"])
        assert json.dumps(actual, sort_keys=True) == json.dumps(node, sort_keys=True), (actual, node)


def main():
    args = options(__doc__)
    tutorials = Path(os.environ.get("IMPACT_TUTORIALS_DIR",
                                   Path(__file__).resolve().parents[4] / "ComfyUI-extension-tutorials"))
    catalog = {node["type"]: node["widgets"] for node in json.loads(WIDGETS.read_text())["nodes"]}
    artifacts = args.artifacts or Path(tempfile.mkdtemp(prefix="impact-animatediff-tutorials-"))
    for name in WORKFLOWS:
        path = tutorials / "ComfyUI-Impact-Pack/workflow" / f"{name}.png"
        with Image.open(path) as image:
            workflow = json.loads(image.info["workflow"])
        expected = []
        for node in workflow["nodes"]:
            if node["type"] not in TYPES:
                continue
            # Names/order and defaults come from the unchanged trusted catalog;
            # saved values come from the original PNG, never the current runtime.
            widgets = [dict(widget) for widget in catalog[node["type"]]]
            saved = node["widgets_values"]
            assert len(saved) <= len(widgets)
            for widget, value in zip(widgets, saved):
                widget["value"] = value
            expected.append(dict(id=str(node["id"]), type=node["type"], widgets=widgets))
        assert any(node["type"] in {"SEGSDetailerForAnimateDiff", "DetailerForEachPipeForAnimateDiff"} for node in expected)
        for restore_by_name in (False, True):
            label = f"{name}-{'named' if restore_by_name else 'positional'}"
            args.artifacts = artifacts / label
            result_path = args.artifacts / "result.json"
            result_path.unlink(missing_ok=True)
            with browser_session(args) as ui:
                ui.page.keyboard.press("Escape")
                setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
                ui.new_workflow()
                ui.load_workflow(path, len(workflow["nodes"]), check_node_errors=False)
                check_widgets(ui, expected)
                missing = ui.app.evaluate("""app => app.graph._nodes.filter(node =>
                    !LiteGraph.registered_node_types[node.type]).map(node => node.type)""")
                assert not TYPES.intersection(missing)
                # Export through the native UI; no hand-built workflow JSON.
                exported = args.artifacts / "exported.json"
                export_workflow(ui, exported)
                ui.new_workflow()
                ui.load_workflow(exported, len(workflow["nodes"]), check_node_errors=False)
                check_widgets(ui, expected)
                report = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              restore_by_name=restore_by_name, widgets=expected, missing_external_nodes=missing,
                              png_import=True, native_export_reload=True, inference=False)
            # Session exit also checks browser errors and trace/close failures.
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"PASS: {label}, {len(expected)} Impact nodes, original PNG and native JSON reload", flush=True)


if __name__ == "__main__":
    main()
