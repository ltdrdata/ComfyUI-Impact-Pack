"""Real SAM execution, downstream masks and native workflow reload in Chrome.

Start an isolated ComfyUI server with Impact Pack, local SAM ViT-B weights and
--cache-none. Set IMPACT_E2E_SAM_IMAGE to a local subject image and
IMPACT_E2E_SAM_MODEL to the same sam_vit_b_01ec64.pth used by that server.
No models are downloaded. API-shaped input is only graph setup; Run and native
Export/import use the UI. Reference predictions come directly from SAM.
"""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from segment_anything import SamPredictor, sam_model_registry

from e2e_ui import ui
from test_e2e_legacy_sam import source_image
from test_e2e_widget_catalog import export_workflow, setting


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    image_path = Path(os.environ["IMPACT_E2E_SAM_IMAGE"])
    model_path = Path(os.environ["IMPACT_E2E_SAM_MODEL"])
    assert model_path.name == "sam_vit_b_01ec64.pth", "This test uses the SAM ViT-B architecture"
    with Image.open(image_path) as original:
        subject = original.convert("RGB")
    subject.thumbnail((256, 448))
    width, height = subject.size
    image = Image.new("RGB", (width * 2, height))
    image.paste(subject, (0, 0))
    image.paste(subject, (width, 0))
    path = tmp_path_factory.mktemp("sam-reference") / "sam-two-subjects.png"
    image.save(path)
    model = sam_model_registry["vit_b"](checkpoint=str(model_path))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(model)
    predictor.set_image(np.asarray(image))
    predictions = []
    for index in range(2):
        box = [index * width, 0, (index + 1) * width, height]
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[(index + .5) * width, height / 2]]),
            point_labels=np.array([1]), box=np.array([box]))
        predictions.append((masks, scores))
    model.to("cpu")
    return path, width, height, predictions


def workflow(filename, width, height, threshold):
    graph = {}

    def add(node_type, **inputs):
        node_id = str(len(graph) + 1)
        graph[node_id] = {"class_type": node_type, "inputs": inputs}
        return node_id

    image = add("LoadImage", image=filename)
    model = add("SAMLoader", model_name="sam_vit_b_01ec64.pth", device_mode="AUTO")
    empty = add("SolidMask", value=0., width=width * 2, height=height)
    half = add("SolidMask", value=1., width=width, height=height)
    segments = []
    for x in (0, width):
        mask = add("MaskComposite", destination=[empty, 0], source=[half, 0], x=x, y=0, operation="add")
        segments.append(add("MaskToSEGS", mask=[mask, 0], combined=False, crop_factor=1.,
                            bbox_fill=False, drop_size=1, contour_fill=False))
    joined = add("ImpactSEGSConcat", segs1=[segments[0], 0], segs2=[segments[1], 0])
    inputs = dict(sam_model=[model, 0], segs=[joined, 0], image=[image, 0], detection_hint="center-1",
                  dilation=3, threshold=threshold, bbox_expansion=0,
                  mask_hint_threshold=.61, mask_hint_use_negative="False")
    segmented = add("SAMDetectorSegmented", **inputs)
    combined = add("SAMDetectorCombined", **inputs)
    per_seg = add("ImpactSegsAndMaskForEach", segs=[joined, 0], masks=[segmented, 1])
    combined_segs = add("ImpactSegsAndMask", segs=[joined, 0], mask=[combined, 0])
    per_seg_masks = add("ImpactSEGSToMaskList", segs=[per_seg, 0])
    combined_seg_masks = add("ImpactSEGSToMaskList", segs=[combined_segs, 0])
    outputs = {}
    prefix = "sam-mask-e2e-" + uuid4().hex
    for name, source in {"batch": [segmented, 1], "segmented_combined": [segmented, 0],
                         "combined": [combined, 0], "per_seg": [per_seg_masks, 0],
                         "combined_segs": [combined_seg_masks, 0]}.items():
        converted = add("MaskToImage", mask=source)
        outputs[name] = add("SaveImage", images=[converted, 0], filename_prefix=prefix + "/" + name)
    return graph, [model, *segments, segmented, combined], outputs


def controls(browser, node_ids):
    return browser.app.evaluate("""(app, ids) => ids.map(id => {
        const node = app.graph.getNodeById(id);
        return {id: String(node.id), type: node.type,
                widgets: (node.widgets ?? []).map(widget => ({name: widget.name, value: widget.value ?? null}))};
    })""", node_ids)


def saved_values(path):
    return {str(node["id"]): {"positional": node.get("widgets_values", []),
                              "named": node.get("widgets_values_named")}
            for node in json.loads(path.read_text())["nodes"]}


def assert_saved_controls(path, controls):
    saved = saved_values(path)
    for node in controls:
        expected = {"positional": [widget["value"] for widget in node["widgets"]],
                    "named": {widget["name"]: widget["value"] for widget in node["widgets"]}}
        assert json.dumps(saved[node["id"]], sort_keys=True) == json.dumps(expected, sort_keys=True), (
            f"Native positional/named widget mappings differ from live {node['type']} controls")


@pytest.mark.parametrize("named", [False, True], ids=["positional", "named"])
@pytest.mark.parametrize("threshold", [.93, 1., 0.], ids=["filtered", "fallback", "all-candidates"])
def test_sam_masks_execute_after_native_reload(ui, reference, named, threshold):
    browser, artifacts = ui
    path, width, height, predictions = reference
    expected_batch, counts = [], []
    for masks, scores in predictions:
        selected = np.flatnonzero(scores >= threshold)
        counts.append(len(selected))
        if not len(selected):
            selected = [int(np.argmax(scores))]
        expected_batch.append(np.logical_or.reduce(masks[selected]))
    if threshold == .93:
        assert any(1 < count < 3 for count in counts), "The fixture must exercise multiple passing candidates"
        assert counts[0] != counts[1], "The fixture must exercise unequal candidate counts"
    elif threshold == 1.:
        assert counts == [0, 0], "The fixture must exercise best-score fallback"
    else:
        assert counts == [3, 3], "The fixture must exercise all three candidates"
    assert all(mask.any() for mask in expected_batch)
    assert not np.array_equal(*expected_batch), "Detection masks must differ so order assertions are meaningful"
    expected_combined = cv2.dilate((expected_batch[0] | expected_batch[1]).astype(np.uint8),
                                   np.ones((3, 3), np.uint8))
    expected = {"batch": expected_batch, "segmented_combined": [expected_combined], "combined": [expected_combined]}
    for name, masks in {"per_seg": expected_batch, "combined_segs": [expected_combined] * 2}.items():
        expected[name] = []
        for index, mask in enumerate(masks):
            cropped = np.zeros_like(mask)
            cropped[:, index * width:(index + 1) * width] = mask[:, index * width:(index + 1) * width]
            expected[name].append(cropped)

    browser.new_workflow()
    source = source_image(browser, path)
    filename = next(widget["value"] for widget in controls(browser, [source])[0]["widgets"]
                    if widget["name"] == "image")
    graph, control_ids, outputs = workflow(filename, width, height, threshold)
    browser.new_workflow()
    browser.load_workflow({"name": "sam-setup-api.json", "mimeType": "application/json",
                           "buffer": json.dumps(graph).encode()}, len(graph))
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", named)
    before = controls(browser, control_ids)
    results = []
    previous_files = None

    for phase in ("before", "reloaded"):
        if phase == "reloaded":
            browser.new_workflow()
            browser.load_workflow(artifacts / "workflow.json", len(graph))
            assert json.dumps(controls(browser, control_ids)) == json.dumps(before), (
                "Live widget names/order/values changed on native load")
            export_workflow(browser, artifacts / "reloaded-workflow.json")
            assert_saved_controls(artifacts / "reloaded-workflow.json", before)
            assert json.dumps(saved_values(artifacts / "reloaded-workflow.json"), sort_keys=True) == json.dumps(
                saved_values(artifacts / "workflow.json"), sort_keys=True), (
                "Native widgets_values changed on reload")
        entry = browser.run()
        cached_events = [data["nodes"] for event, data in entry["status"]["messages"] if event == "execution_cached"]
        assert cached_events, "Execution history must expose cache status"
        cached = {str(node) for event in cached_events for node in event}
        assert not cached, "Every node must run again; start the test server with --cache-none"
        images = browser.images(entry)
        filenames = {name: entry["outputs"][node_id]["images"] for name, node_id in outputs.items()}
        if previous_files is not None:
            assert all(filenames[name] != previous_files[name] for name in outputs), "Reload returned stale output files"
        previous_files = filenames
        hashes = {}
        for name, masks in expected.items():
            actual = images[outputs[name]]
            assert len(actual) == len(masks), f"{phase}: {name} count {len(actual)} != {len(masks)}"
            hashes[name] = []
            for index, (image, mask) in enumerate(zip(actual, masks, strict=True)):
                pixels = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
                np.testing.assert_array_equal(np.asarray(image), pixels, err_msg=f"{phase}: {name}[{index}] pixels")
                image.save(artifacts / f"{phase}-{name}-{index}.png")
                hashes[name].append(hashlib.sha256(image.tobytes()).hexdigest())
        results.append({"phase": phase, "prompt_id": entry["prompt"][1], "pixel_sha256": hashes, "cached_nodes": sorted(cached)})
        after = controls(browser, control_ids)
        assert json.dumps(after) == json.dumps(before), "Execution changed saved SAM controls"
        browser.page.get_by_role("button", name="Fit View (.)", exact=True).click()
        browser.page.screenshot(path=str(artifacts / f"{phase}.png"))
        if phase == "before":
            export_workflow(browser, artifacts / "workflow.json")
            assert_saved_controls(artifacts / "workflow.json", before)
    (artifacts / "results.json").write_text(json.dumps({
        "threshold": threshold, "passing_candidates": counts, "named_restore": named,
        "scores": [scores.tolist() for _, scores in predictions], "runs": results,
        "widgets_before": before, "widgets_after": after,
    }, indent=2) + "\n")
