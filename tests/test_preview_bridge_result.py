import pytest
import torch

from impact import bridge_nodes, core


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
def test_execution_feedback_identifies_the_submitted_image_on_refresh_and_reuse(tmp_path, monkeypatch, node_type):
    monkeypatch.setattr(bridge_nodes.folder_paths, "temp_directory", str(tmp_path))
    for name in ("preview_bridge_cache", "preview_bridge_last_mask_cache", "preview_bridge_image_id_map", "preview_bridge_image_name_map"):
        monkeypatch.setattr(core, name, {})
    node = getattr(bridge_nodes, node_type)()
    inputs = {"image": "$submitted-image", "unique_id": "feedback-test"}
    if node_type == "PreviewBridge":
        inputs["images"] = torch.zeros((1, 64, 64, 3))
    else:
        inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
    refreshed = node.doit(**inputs)
    assert refreshed["ui"]["source_image"] == ["$submitted-image"]
    assert refreshed["ui"]["images"][0]["filename"]
    reused = node.doit(**inputs)
    assert reused["ui"]["source_image"] == ["$submitted-image"]
    assert reused["ui"]["images"] == refreshed["ui"]["images"]
