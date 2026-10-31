import pytest
import torch

from impact import bridge_nodes, core


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_nodes.folder_paths, "temp_directory", str(tmp_path))
    for name in ("preview_bridge_cache", "preview_bridge_last_mask_cache", "preview_bridge_image_id_map"):
        monkeypatch.setattr(core, name, {})
    return bridge_nodes.PreviewBridgeLatent()


@pytest.mark.parametrize("restore,width,restored", [
    ("always", 32, True), ("if_same_size", 32, True), ("always", 40, True),
    ("never", 32, False), ("never", 40, False), ("if_same_size", 40, False),
])
def test_latent_refresh_propagates_only_the_applicable_restored_mask(bridge, restore, width, restored):
    mask = torch.zeros((1, 256, 256))
    mask[:, 48:128, 32:160] = 1
    core.preview_bridge_last_mask_cache["restore-test"] = mask
    latent = {"samples": torch.zeros((1, 4, 32, width)), "metadata": "preserve"}
    result = bridge.doit(latent, "$restore-test", "Latent2RGB-SD15", unique_id="restore-test", restore_mask=restore)
    output, returned_mask = result["result"]
    assert output["samples"] is latent["samples"]
    assert output["metadata"] == "preserve"
    assert "noise_mask" not in latent
    if restored:
        assert output is not latent
        torch.testing.assert_close(returned_mask, mask)
        torch.testing.assert_close(output["noise_mask"], mask)
    else:
        assert "noise_mask" not in output
        assert torch.all(returned_mask == 1)


def test_incoming_noise_mask_takes_priority_over_the_cached_mask(bridge):
    core.preview_bridge_last_mask_cache["incoming-test"] = torch.zeros((1, 256, 256))
    latent = {"samples": torch.zeros((1, 4, 32, 32)), "noise_mask": torch.full((1, 256, 256), 0.5)}
    result = bridge.doit(latent, "$incoming-test", "Latent2RGB-SD15", unique_id="incoming-test", restore_mask="always")
    output, mask = result["result"]
    assert output is latent
    torch.testing.assert_close(mask, latent["noise_mask"].squeeze(0))
