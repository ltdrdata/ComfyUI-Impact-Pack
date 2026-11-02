from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from PIL import Image
import pytest
import torch

from impact import bridge_nodes, core


@pytest.fixture
def bridge_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_nodes.folder_paths, "temp_directory", str(tmp_path))
    for name in ("preview_bridge_cache", "preview_bridge_last_mask_cache", "preview_bridge_image_id_map"):
        monkeypatch.setattr(core, name, {})
    return tmp_path


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
def test_execution_feedback_identifies_the_submitted_image_on_refresh_and_reuse(bridge_files, node_type):
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


def test_concurrent_registrations_keep_distinct_image_references(bridge_files, monkeypatch):
    first = bridge_files / "first.png"
    second = bridge_files / "second.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(first)
    Image.new("RGBA", (2, 2), (0, 0, 255, 128)).save(second)
    opened = Barrier(2)
    open_image = Image.open

    def concurrent_open(*args, **kwargs):
        image = open_image(*args, **kwargs)
        opened.wait(timeout=5)
        return image

    monkeypatch.setattr(core.Image, "open", concurrent_open)

    def register(path):
        return core.set_previewbridge_image("same-node", str(path), {"filename": path.name, "subfolder": "", "type": "input"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        references = list(executor.map(register, [first, second]))
    assert references[0] != references[1]
    assert [core.preview_bridge_image_id_map[reference][0] for reference in references] == [str(first), str(second)]


@pytest.mark.parametrize("value,rgb,mask_shape,mask_value", [
    (0, [1, 0, 0], (1, 3, 2), 127 / 255),
    (1, [1, 1, 1], (64, 64), 0),
])
def test_recomputed_source_preserves_edits_only_when_pixels_match(bridge_files, value, rgb, mask_shape, mask_value):
    node = bridge_nodes.PreviewBridge()
    source = torch.zeros((1, 3, 2, 3))
    node.doit(source, "$source", unique_id="recomputed")
    edited = bridge_files / "edited.png"
    Image.new("RGBA", (2, 3), (255, 0, 0, 128)).save(edited)
    reference = core.set_previewbridge_image("recomputed", str(edited), {
        "filename": edited.name, "subfolder": "", "type": "temp",
    })

    pixels, mask = node.doit(torch.full_like(source, value), reference, unique_id="recomputed")["result"]

    torch.testing.assert_close(pixels, torch.tensor(rgb, dtype=torch.float32).expand(1, 3, 2, 3))
    torch.testing.assert_close(mask, torch.full(mask_shape, mask_value, dtype=torch.float32))
