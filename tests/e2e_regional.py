"""Chrome regression for the regional-sampler-img2img tutorial.

Set IMPACT_REGIONAL_WORKFLOW to its PNG or JSON with the original embedded
source image and color map restored. Requires Inspire Pack, Essentials and
the checkpoint named by the workflow. No images or models are substituted.
"""
import base64
import copy
import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from e2e_ui import browser_session, options


REGIONS = (("39", "43", (0, 255, 0)), ("9", "44", (255, 0, 0)), ("10", "45", (0, 0, 255)))


def check_masks(images, expected):
    for node, mask in expected.items():
        assert len(images[node]) == 1
        actual = np.asarray(images[node][0])
        assert actual.shape == (*mask.shape, 3), f"Mask {node} dimensions differ"
        assert np.array_equal(actual, np.repeat(mask[:, :, None], 3, axis=2) * 255), f"Mask {node} pixels differ"


def region_effect(baseline, changed, mask):
    assert baseline.size == changed.size
    # VAEEncode center-crops this 850px sample to 848px. Exclude boundary
    # pixels where mask overlap and VAE decoding can affect neighbouring regions.
    width, height = baseline.size
    y, x = (mask.shape[0] - height) // 2, (mask.shape[1] - width) // 2
    mask = mask[y:y + height, x:x + width]
    inside = minimum_filter(mask, size=65, mode="nearest")
    outside = ~maximum_filter(mask, size=129, mode="nearest")
    assert inside.any() and outside.any(), "Region needs an interior and a distant exterior"
    difference = np.abs(np.asarray(baseline, dtype=np.float32) - np.asarray(changed, dtype=np.float32)).mean(axis=2)
    inner, outer = float(difference[inside].mean()), float(difference[outside].mean())
    assert inner > 1.0, f"Regional prompt had no meaningful effect: {inner=}"
    assert inner > 2 * outer, f"Prompt change was not localized to its region: {inner=}, {outer=}"
    return {"inside_mae": inner, "outside_mae": outer}


def main():
    args = options(__doc__)
    path = Path(os.environ["IMPACT_REGIONAL_WORKFLOW"])
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            workflow = json.loads(image.info["workflow"])
    else:
        workflow = json.loads(path.read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node, index in ((35, 3), (49, 1)):
        assert nodes[node]["widgets_values"][index].startswith("data:image/"), "Restore the original embedded images first"
        # Embedded data is authoritative; the old session's file need not exist.
        nodes[node]["widgets_values"][0] = "#DATA"
    with Image.open(io.BytesIO(base64.b64decode(nodes[35]["widgets_values"][3].split(",", 1)[1]))) as image:
        color_map = np.asarray(image.convert("RGB"))
    masks = {preview: np.all(color_map == color, axis=2) for _, preview, color in REGIONS}
    assert all(mask.any() and not mask.all() for mask in masks.values())

    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        ui.new_workflow()
        ui.load_workflow({"name": "regional-tutorial.json", "mimeType": "application/json",
                          "buffer": json.dumps(workflow).encode()}, len(workflow["nodes"]))
        with ui.page.expect_request(lambda request: urlsplit(request.url).path in ("/prompt", "/api/prompt")
                                    and request.method == "POST") as submitted:
            original = ui.images(ui.run())
        assert original.keys() == {"25", "43", "44", "45"}
        assert all(len(images) == 1 for images in original.values())
        check_masks(original, masks)
        prompt = submitted.value.post_data_json["prompt"]
        prompt["roundtrip"] = {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["30", 3]}}
        prompt["reference"] = {"class_type": "PreviewImage", "inputs": {"images": ["roundtrip", 0]}}
        ui.new_workflow()
        ui.load_workflow({"name": "regional-reference.json", "mimeType": "application/json",
                          "buffer": json.dumps(prompt).encode()}, len(prompt))
        baseline = ui.images(ui.run())
        assert baseline.keys() == original.keys() | {"reference"}
        assert all(len(images) == 1 for images in baseline.values())
        check_masks(baseline, masks)
        assert baseline["25"][0].tobytes() == original["25"][0].tobytes(), "Import changed the sampled result"
        assert baseline["25"][0].size == baseline["reference"][0].size
        difference = np.abs(np.asarray(baseline["25"][0], dtype=np.float32)
                            - np.asarray(baseline["reference"][0], dtype=np.float32)).mean()
        assert difference > 1.0, "Sampling returned the unprocessed VAE roundtrip"

        effects = {}
        for node, preview, _ in REGIONS:
            changed = copy.deepcopy(prompt)
            changed[node]["inputs"]["wildcard_prompt"] = "a bright yellow stone wall, geometric blocks, no people"
            ui.new_workflow()
            ui.load_workflow({"name": f"regional-prompt-{node}.json", "mimeType": "application/json",
                              "buffer": json.dumps(changed).encode()}, len(changed))
            result = ui.images(ui.run())
            assert result.keys() == baseline.keys()
            assert all(len(images) == 1 for images in result.values())
            check_masks(result, masks)
            effects[node] = region_effect(baseline["25"][0], result["25"][0], masks[preview])
        if args.artifacts:
            (args.artifacts / "regional-effects.json").write_text(json.dumps(effects, indent=2))
        print("PASS: exact color masks, VAE roundtrip comparison, localized effect of all three prompts, browser pixels", effects)


if __name__ == "__main__":
    main()
