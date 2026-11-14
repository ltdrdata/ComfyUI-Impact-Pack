"""Unmodified hook tutorials from extension-tutorials ca31a3b."""
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cycles": ("workflow/detailer-hook.png", "40e10ea74169253bc61ee54faeec056247f419118658139cf1ffa0b22c35ee84"),
    "preview": ("workflow/PreviewDetailerHookProvider.png", "8dbceeee95aa9109786ab800aaa5ff84723d8082175823839229590378cb529b"),
    "sampler": ("workflow/detailer-with-detail-daemon.png", "d4e8b33fe6f89b24c369f3aae276497c7b10d0f5592ec8d93c87a47eba19c534"),
    "handfix": ("workflow/meshgraphomer_dwpose_tile-segs-handfix.png", "561d7e1c6a543336f0f2a127f45b5e8dadf3f59ab5fee4de473fb400521dcf6f"),
    "animatediff": ("workflow/MakeTileSEGS_upscale_AnimateDiff_ControlNet.png", "147078341423c890c917a2d39a60acf14369d5e90c9899bfa781938600cc8412"),
    "pixel_schedule": ("workflow/pk_hook.png", "9e705aa9219d6da29dc771ea3792938f232f81994194670614121b18907ef0a0"),
    "upscale_preview": ("tutorial/extreme-upscale-original.png", "8af79e6ad0428dabbf1ef98cfbc685dc34f7485065b8c266466b3cc90417885a"),
}


def tutorial(case):
    relative, digest = CASES[case]
    root = Path(os.environ["IMPACT_TUTORIALS_DIR"])
    path = root / "ComfyUI-Impact-Pack" / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Original tutorial changed: {path}"
    with Image.open(path) as image:
        workflow = json.loads(image.info["workflow"])
    return path, workflow


def expected_widgets(node):
    catalog = json.loads((ROOT / "tests/workflows/all-impact-nodes.widgets.json").read_text())
    reference = next(n for n in catalog["nodes"] if n["type"] == node["type"])
    saved = node.get("widgets_values") or []
    assert len(saved) <= len(reference["widgets"]), node["type"]
    return [{"name": widget["name"], "value": saved[i] if i < len(saved) else widget["value"]}
            for i, widget in enumerate(reference["widgets"])]


def saved_inputs(case, node_id):
    _, workflow = tutorial(case)
    node = next(n for n in workflow["nodes"] if n["id"] == node_id)
    return {widget["name"]: widget["value"] for widget in expected_widgets(node)
            if widget["name"] != "control_after_generate"}


def targets(workflow):
    return [node for node in workflow["nodes"] if (
        "Hook" in node["type"] or "Detailer" in node["type"] or
        any("hook" in slot["name"] for slot in node.get("inputs", [])))]
