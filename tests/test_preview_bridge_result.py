import asyncio
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
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
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
@pytest.mark.parametrize("block", [False, True])
def test_saved_clear_survives_input_refresh(bridge_files, node_type, restore, block):
    from comfy_execution.graph import ExecutionBlocker
    node = getattr(bridge_nodes, node_type)()
    inputs = {"unique_id": "erase", "restore_mask": restore}
    if node_type == "PreviewBridge":
        inputs["images"] = torch.zeros((1, 64, 64, 3))
    else:
        inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
    node.doit(image="$initial", **inputs)
    for filename, alpha in [("painted.png", 128), ("cleared.png", 255)]:
        path = bridge_files / filename
        Image.new("RGBA", (64, 64), (0, 0, 0, alpha)).save(path)
        reference = core.set_previewbridge_image("erase", str(path), {
            "filename": filename, "subfolder": "", "type": "temp",
        })
        if alpha == 255:
            if node_type == "PreviewBridge":
                inputs["images"] = torch.ones_like(inputs["images"])
            else:
                inputs["latent"] = {"samples": torch.ones_like(inputs["latent"]["samples"])}
        output, mask = node.doit(image=reference, block=block, **inputs)["result"]
        if block and alpha == 255:
            assert isinstance(output, ExecutionBlocker)
            assert isinstance(mask, ExecutionBlocker)
            continue
        torch.testing.assert_close(mask, torch.full((1, 64, 64), 1 - alpha / 255))
        if node_type == "PreviewBridgeLatent":
            torch.testing.assert_close(output["noise_mask"], mask)
            assert "noise_mask" not in inputs["latent"]


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
@pytest.mark.parametrize("first_alpha", [191, 255], ids=["painted", "cleared"])
def test_queued_refresh_uses_its_submitted_mask(bridge_files, node_type, restore, first_alpha):
    node = getattr(bridge_nodes, node_type)()
    inputs = {"unique_id": "queued", "restore_mask": restore}
    if node_type == "PreviewBridge":
        inputs["images"] = torch.zeros((1, 64, 64, 3))
    else:
        inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
    node.doit(image="$initial", **inputs)

    # Both prompts register before either executes; the node cache now holds B.
    references = []
    for filename, alpha in [("a.png", first_alpha), ("b.png", 64)]:
        path = bridge_files / filename
        Image.new("RGBA", (64, 64), (0, 0, 0, alpha)).save(path)
        references.append(core.set_previewbridge_image("queued", str(path), {
            "filename": filename, "subfolder": "", "type": "temp",
        }))

    for value, reference, alpha in zip([0.25, 0.75], references, [first_alpha, 64]):
        if node_type == "PreviewBridge":
            inputs["images"] = torch.full_like(inputs["images"], value)
        else:
            inputs["latent"] = {"samples": torch.full_like(inputs["latent"]["samples"], value)}
        output, mask = node.doit(image=reference, **inputs)["result"]
        torch.testing.assert_close(mask, torch.full((1, 64, 64), 1 - alpha / 255))
        if node_type == "PreviewBridgeLatent":
            torch.testing.assert_close(output["noise_mask"], mask)
            assert "noise_mask" not in inputs["latent"]
        else:
            torch.testing.assert_close(output, inputs["images"])


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
@pytest.mark.parametrize("first_restore", ["always", "never"])
@pytest.mark.parametrize("alpha", [128, 255], ids=["painted", "cleared"])
def test_shared_registration_preserves_mask_and_refreshed_preview(bridge_files, node_type, first_restore, alpha):
    node = getattr(bridge_nodes, node_type)()

    def source(size, value):
        if node_type == "PreviewBridge":
            return {"images": torch.full((1, size, size, 3), value)}
        return {"latent": {"samples": torch.full((1, 4, size // 8, size // 8), value)},
                "preview_method": "Latent2RGB-SD15"}

    node.doit(image="$shared-0", unique_id="shared", **source(64, 0.0))
    path = bridge_files / "shared.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, alpha)).save(path)
    reference = core.set_previewbridge_image("shared", str(path), {
        "filename": path.name, "subfolder": "", "type": "temp",
    })

    # Both queues use the same saved file, but different input sizes/policies.
    node.doit(image=reference, unique_id="shared", restore_mask=first_restore, **source(128, 0.25))
    inputs = source(64, 1.0)
    expected_mask = torch.full((1, 64, 64), 1 - alpha / 255)
    refreshed_rgb = None
    for _ in range(2):
        # Reuse must keep the refreshed pixels, not reload the original red image.
        result = node.doit(image=reference, unique_id="shared", restore_mask="if_same_size", **inputs)
        output, mask = result["result"]
        item, = result["ui"]["images"]
        with Image.open(bridge_files / item["subfolder"] / item["filename"]) as preview:
            rgb = preview.convert("RGB").tobytes()
        if refreshed_rgb is None:
            assert rgb != Image.new("RGB", (64, 64), "red").tobytes()
            refreshed_rgb = rgb
        else:
            assert rgb == refreshed_rgb, "Reusing a latent must retain its refreshed preview pixels"
        torch.testing.assert_close(mask, expected_mask)
        if node_type == "PreviewBridge":
            torch.testing.assert_close(output, inputs["images"])
        else:
            torch.testing.assert_close(output["noise_mask"], expected_mask)
            assert output["samples"] is inputs["latent"]["samples"]
            assert "noise_mask" not in inputs["latent"]


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


@pytest.mark.parametrize("first_type", ["PreviewBridge", "PreviewBridgeLatent"])
def test_reusing_a_node_id_for_the_other_bridge_discards_its_cache(bridge_files, first_type):
    images = torch.zeros((1, 64, 64, 3))
    latent = {"samples": torch.zeros((1, 4, 8, 8))}

    def execute(node_type):
        inputs = {"image": "$reused-id", "unique_id": "reused-id", "restore_mask": "always"}
        if node_type == "PreviewBridge":
            inputs["images"] = images
        else:
            inputs.update(latent=latent, preview_method="Latent2RGB-SD15")
        return getattr(bridge_nodes, node_type)().doit(**inputs)

    execute(first_type)
    # An edited mask from the other node type must not leak into the new node.
    core.preview_bridge_last_mask_cache["reused-id"] = torch.full((1, 64, 64), 0.5)
    second_type = "PreviewBridgeLatent" if first_type == "PreviewBridge" else "PreviewBridge"
    result = execute(second_type)
    output, mask = result["result"]
    if second_type == "PreviewBridge":
        torch.testing.assert_close(output, images)
        assert torch.all(mask == 0)
    else:
        assert output is latent
        assert "noise_mask" not in output
        assert torch.all(mask == 1)
    assert result["ui"]["source_image"] == ["$reused-id"]


@pytest.mark.parametrize("restore", ["never", "always", "if_same_size"])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_recomputed_image_batch_keeps_every_frame(bridge_files, restore, batch_size):
    node = bridge_nodes.PreviewBridge()
    images = torch.linspace(0, 1, batch_size).reshape(batch_size, 1, 1, 1).expand(-1, 64, 64, 3)
    inputs = {"image": "$batch", "unique_id": "batch", "restore_mask": restore}
    node.doit(images, **inputs)

    result = node.doit(images.clone(), **inputs)

    torch.testing.assert_close(result["result"][0], images)
    assert len(result["ui"]["images"]) == batch_size


@pytest.mark.parametrize("restore,width,restored", [
    ("always", 32, True), ("if_same_size", 32, True), ("always", 40, True),
    ("never", 32, False), ("never", 40, False), ("if_same_size", 40, False),
])
def test_latent_refresh_propagates_only_the_applicable_restored_mask(bridge_files, restore, width, restored):
    mask = torch.zeros((1, 256, 256))
    mask[:, 48:128, 32:160] = 1
    core.preview_bridge_last_mask_cache["restore-test"] = mask
    latent = {"samples": torch.zeros((1, 4, 32, width)), "metadata": "preserve"}
    result = bridge_nodes.PreviewBridgeLatent().doit(
        latent, "$restore-test", "Latent2RGB-SD15", unique_id="restore-test", restore_mask=restore)
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


def test_incoming_noise_mask_takes_priority_over_the_cached_mask(bridge_files):
    core.preview_bridge_last_mask_cache["incoming-test"] = torch.zeros((1, 256, 256))
    latent = {"samples": torch.zeros((1, 4, 32, 32)), "noise_mask": torch.full((1, 256, 256), 0.5)}
    result = bridge_nodes.PreviewBridgeLatent().doit(
        latent, "$incoming-test", "Latent2RGB-SD15", unique_id="incoming-test", restore_mask="always")
    output, mask = result["result"]
    assert output is latent
    torch.testing.assert_close(mask, latent["noise_mask"].squeeze(0))


@pytest.mark.parametrize("block", [False, True])
def test_unedited_latent_requeue_keeps_no_mask_and_block(bridge_files, block):
    from comfy_execution.graph import ExecutionBlocker
    node = bridge_nodes.PreviewBridgeLatent()
    latent = {"samples": torch.zeros((1, 4, 8, 12))}
    image = "$unedited-0"
    for queue in range(3):
        result = node.doit(latent, image, "Latent2RGB-SD15", unique_id="unedited", block=block)
        output, mask = result["result"]
        if block:
            assert isinstance(output, ExecutionBlocker)
            assert isinstance(mask, ExecutionBlocker)
        else:
            assert "noise_mask" not in output
            assert output["samples"] is latent["samples"]
            torch.testing.assert_close(mask, torch.ones((1, 8, 12)))
        if queue == 0:
            # First feedback changes the filename; the third queue keeps this ID.
            item, = result["ui"]["images"]
            image = core.set_previewbridge_image("unedited", str(bridge_files / item["subfolder"] / item["filename"]), item)
    assert "noise_mask" not in latent


@pytest.mark.parametrize("alpha", [None, 255, 128, 0], ids=["no-alpha", "clear", "painted", "full"])
@pytest.mark.parametrize("route", ["reuse", "refresh", "incoming"])
@pytest.mark.parametrize("block", [False, True])
def test_latent_mask_presence_controls_blocking(bridge_files, alpha, route, block):
    from comfy_execution.graph import ExecutionBlocker
    node = bridge_nodes.PreviewBridgeLatent()
    latent = {"samples": torch.zeros((1, 4, 8, 12))}
    image = "$edit-0"
    if route == "incoming":
        if alpha is not None:
            latent["noise_mask"] = torch.full((1, 64, 96), 1 - alpha / 255)
    else:
        if route == "reuse":
            # A saved edit must replace an incoming mask without mutating it.
            latent["noise_mask"] = torch.full((1, 64, 96), 0.5)
        node.doit(latent, image, "Latent2RGB-SD15", unique_id="edit")
        if route == "refresh":
            latent = {"samples": torch.ones_like(latent["samples"])}
        path = bridge_files / "edited.png"
        Image.new("RGB" if alpha is None else "RGBA", (96, 64),
                  (0, 0, 0) if alpha is None else (0, 0, 0, alpha)).save(path)
        image = core.set_previewbridge_image("edit", str(path), {"filename": path.name, "subfolder": "", "type": "temp"})
    before = {key: value.clone() for key, value in latent.items()}
    output, mask = node.doit(latent, image, "Latent2RGB-SD15", unique_id="edit",
                             restore_mask="always", block=block)["result"]
    if block and alpha in (None, 255):
        assert isinstance(output, ExecutionBlocker)
        assert isinstance(mask, ExecutionBlocker)
    else:
        assert output["samples"] is latent["samples"]
        if alpha is None:
            assert "noise_mask" not in output
            torch.testing.assert_close(mask, torch.ones((1, 8, 12)))
        else:
            expected = torch.full((1, 64, 96), 1 - alpha / 255)
            torch.testing.assert_close(mask, expected.squeeze(0) if route == "incoming" else expected)
            torch.testing.assert_close(output["noise_mask"], expected)
    assert latent.keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(latent[key], value)


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
def test_never_refresh_does_not_revive_an_older_cached_mask(bridge_files, node_type):
    node = getattr(bridge_nodes, node_type)()
    def source(value):
        if node_type == "PreviewBridge":
            return {"images": torch.full((1, 64, 64, 3), value)}
        return {"latent": {"samples": torch.full((1, 4, 8, 8), value)}, "preview_method": "Latent2RGB-SD15"}
    path = bridge_files / "painted.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 128)).save(path)
    reference = core.set_previewbridge_image("policy", str(path), {"filename": path.name, "subfolder": "", "type": "temp"})
    node.doit(image=reference, unique_id="policy", restore_mask="always", **source(0.0))
    node.doit(image=reference, unique_id="policy", restore_mask="never", **source(0.5))
    # A legacy/missing reference has no saved alpha and must use the fallback cache.
    output, mask = node.doit(image="$unavailable", unique_id="policy", restore_mask="always", **source(1.0))["result"]
    if node_type == "PreviewBridge":
        assert torch.all(mask == 0)
    else:
        assert torch.all(mask == 1)
        assert "noise_mask" not in output


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
@pytest.mark.parametrize("saved_file", [True, False], ids=["saved", "missing"])
def test_legacy_clipspace_reference_loads_or_refreshes(bridge_files, monkeypatch, node_type, saved_file):
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(bridge_files))
    node = getattr(bridge_nodes, node_type)()
    inputs = {"unique_id": "legacy"}
    if node_type == "PreviewBridge":
        inputs["images"] = torch.zeros((1, 64, 64, 3))
    else:
        inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
    node.doit(image="$legacy-0", **inputs)
    path = bridge_files / "clipspace" / "legacy-edited.png"
    if saved_file:
        path.parent.mkdir()
        Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(path)
    result = node.doit(image="clipspace/legacy-edited.png [input]", **inputs)
    output, mask = result["result"]
    if saved_file:
        torch.testing.assert_close(mask, torch.full((1, 64, 64), 127 / 255))
        assert result["ui"]["images"] == [{"filename": path.name, "subfolder": "clipspace", "type": "input"}]
        if node_type == "PreviewBridge":
            torch.testing.assert_close(output, torch.tensor([1., 0., 0.]).expand(1, 64, 64, 3))
        else:
            torch.testing.assert_close(output["noise_mask"], mask)
    else:
        item, = result["ui"]["images"]
        assert item["type"] == "temp"
        assert (bridge_files / item["subfolder"] / item["filename"]).is_file()
        assert torch.all(mask == (0 if node_type == "PreviewBridge" else 1))
        if node_type == "PreviewBridge":
            torch.testing.assert_close(output, inputs["images"])
        else:
            assert "noise_mask" not in output
    if node_type == "PreviewBridgeLatent":
        assert output["samples"] is inputs["latent"]["samples"]
        assert "noise_mask" not in inputs["latent"]


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
@pytest.mark.parametrize("restore,size,restored", [
    ("never", 64, False), ("always", 64, True), ("if_same_size", 64, True),
    ("always", 96, True), ("if_same_size", 96, False),
])
@pytest.mark.parametrize("alpha", [128, 255, None], ids=["painted", "clear", "missing"])
def test_raw_clipspace_mask_obeys_restoration_on_changed_input(
        bridge_files, monkeypatch, node_type, restore, size, restored, alpha):
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(bridge_files))
    node = getattr(bridge_nodes, node_type)()

    def source(size, value):
        if node_type == "PreviewBridge":
            return {"images": torch.full((1, size, size, 3), value)}
        return {"latent": {"samples": torch.full((1, 4, size // 8, size // 8), value)},
                "preview_method": "Latent2RGB-SD15"}

    node.doit(image="$raw-0", unique_id="raw", **source(64, 0.0))
    path = bridge_files / "clipspace" / "raw.png"
    if alpha is not None:
        path.parent.mkdir()
        Image.new("RGBA", (64, 64), (255, 0, 0, alpha)).save(path)
    image = "clipspace/raw.png [input]"
    assert image not in core.preview_bridge_image_id_map
    inputs = source(size, 1.0)
    result = node.doit(image=image, unique_id="raw", restore_mask=restore, **inputs)
    output, mask = result["result"]
    restored = restored and alpha is not None
    if restored:
        torch.testing.assert_close(mask, torch.full((1, 64, 64), 1 - alpha / 255))
    else:
        expected = torch.zeros((64, 64)) if node_type == "PreviewBridge" else torch.ones((1, size // 8, size // 8))
        torch.testing.assert_close(mask, expected)
    if node_type == "PreviewBridge":
        torch.testing.assert_close(output, inputs["images"])
    else:
        assert output["samples"] is inputs["latent"]["samples"]
        if restored:
            torch.testing.assert_close(output["noise_mask"], mask)
        else:
            assert "noise_mask" not in output
        assert "noise_mask" not in inputs["latent"]
    item, = result["ui"]["images"]
    with Image.open(bridge_files / item["subfolder"] / item["filename"]) as preview:
        assert preview.size == (size, size)
        if restored:
            assert preview.getchannel("A").getextrema() == (alpha, alpha)


@pytest.mark.parametrize("kind", [
    "input-root", "nested", "legacy-basename", "absolute-inside",
    "no-space-suffix", "two-dot-filename", "cwd-shadow",
])
def test_raw_file_registration_stays_in_input_and_preserves_metadata(bridge_files, monkeypatch, kind):
    input_dir = bridge_files / "input"
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(input_dir))
    subfolder = "" if kind in ("input-root", "two-dot-filename") else "clipspace/edits" if kind in (
        "nested", "absolute-inside") else "clipspace"
    filename = "my..image.png" if kind == "two-dot-filename" else "mask.png"
    saved = input_dir / subfolder / filename
    saved.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(saved)
    relative = str(saved.relative_to(input_dir))
    if kind == "legacy-basename":
        relative = filename
    elif kind == "absolute-inside":
        relative = str(saved)
    elif kind == "cwd-shadow":
        shadow = bridge_files / "working" / "clipspace" / filename
        shadow.parent.mkdir(parents=True)
        Image.new("RGBA", (64, 64), (0, 255, 0, 64)).save(shadow)
        monkeypatch.chdir(shadow.parent.parent)
    reference = relative + ("[input]" if kind == "no-space-suffix" else " [input]")
    assert bridge_nodes.PreviewBridge.register_clipspace_image(reference, "allowed")
    pixels, mask, item = bridge_nodes.PreviewBridge.load_image(reference)
    torch.testing.assert_close(pixels, torch.tensor([1., 0., 0.]).expand(1, 64, 64, 3))
    torch.testing.assert_close(mask, torch.full((1, 64, 64), 127 / 255))
    assert item == {"filename": filename, "subfolder": subfolder, "type": "input"}
    assert (input_dir / item["subfolder"] / item["filename"]).read_bytes() == saved.read_bytes()


@pytest.mark.parametrize("kind", [
    "absolute-outside", "parent-traversal", "sibling-prefix", "null-byte",
])
def test_raw_file_registration_rejects_escape_without_reading(bridge_files, monkeypatch, kind):
    input_dir = bridge_files / "input"
    input_dir.mkdir()
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(input_dir))
    outside = bridge_files / "input-other" / "mask.png"
    outside.parent.mkdir()
    Image.new("RGBA", (64, 64), (0, 255, 0, 128)).save(outside)
    (input_dir / "clipspace").mkdir()
    fallback = input_dir / "clipspace" / "mask.png"
    # A rejected external reference must not silently select this other file.
    Image.new("RGBA", (64, 64), (255, 0, 0, 64)).save(fallback)
    if kind in ("absolute-outside", "sibling-prefix"):
        reference = str(outside) if kind == "absolute-outside" else "../input-other/mask.png"
    elif kind == "parent-traversal":
        reference = "clipspace/../../input-other/mask.png"
    else:
        reference = "bad\0mask.png"

    def unexpected_open(*args, **kwargs):
        pytest.fail("Rejected raw references must not read image contents")

    monkeypatch.setattr(Image, "open", unexpected_open)
    assert not bridge_nodes.PreviewBridge.register_clipspace_image(reference + " [input]", "rejected")
    assert not core.preview_bridge_image_id_map
    assert not core.preview_bridge_last_mask_cache


@pytest.mark.parametrize("kind", ["internal-file", "external-file", "external-directory", "legacy-fallback"])
def test_raw_symlinks_preserve_saved_mask_and_logical_metadata(bridge_files, monkeypatch, kind):
    input_dir = bridge_files / "input"
    input_dir.mkdir()
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(input_dir))
    target_dir = input_dir / "originals" if kind == "internal-file" else bridge_files / "other-disk"
    target_dir.mkdir()
    target = target_dir / "saved.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(target)
    if kind == "external-directory":
        (input_dir / "clipspace").symlink_to(target_dir, target_is_directory=True)
        relative = "clipspace/saved.png"
        filename, subfolder = "saved.png", "clipspace"
    else:
        subfolder = "clipspace" if kind == "legacy-fallback" else ""
        (input_dir / subfolder).mkdir(exist_ok=True)
        (input_dir / subfolder / "alias.png").symlink_to(target)
        relative = filename = "alias.png"
    reference = relative + " [input]"
    assert bridge_nodes.PreviewBridge.register_clipspace_image(reference, "linked")
    pixels, mask, item = bridge_nodes.PreviewBridge.load_image(reference)
    torch.testing.assert_close(pixels, torch.tensor([1., 0., 0.]).expand(1, 64, 64, 3))
    torch.testing.assert_close(mask, torch.full((1, 64, 64), 127 / 255))
    assert item == {"filename": filename, "subfolder": subfolder, "type": "input"}
    assert (input_dir / subfolder / filename).read_bytes() == target.read_bytes()


@pytest.mark.parametrize("node_type", ["PreviewBridge", "PreviewBridgeLatent"])
@pytest.mark.parametrize("reuse", [False, True], ids=["first-run", "reuse"])
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
def test_external_raw_file_does_not_reach_either_bridge(bridge_files, monkeypatch, node_type, reuse, restore):
    input_dir = bridge_files / "input"
    input_dir.mkdir()
    monkeypatch.setattr(bridge_nodes.folder_paths, "input_directory", str(input_dir))
    outside = bridge_files / "outside.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(outside)
    original_open = Image.open

    def checked_open(path, *args, **kwargs):
        assert str(path) != str(outside), "Bridge opened an external raw file"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", checked_open)
    node = getattr(bridge_nodes, node_type)()
    inputs = {"unique_id": "external", "restore_mask": restore}
    if node_type == "PreviewBridge":
        inputs["images"] = torch.zeros((1, 64, 64, 3))
    else:
        inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
    if reuse:
        node.doit(image="$external-0", **inputs)
    reference = str(outside) + " [input]"
    output, mask = node.doit(image=reference, **inputs)["result"]
    assert reference not in core.preview_bridge_image_id_map
    if node_type == "PreviewBridge":
        torch.testing.assert_close(output, inputs["images"])
        torch.testing.assert_close(mask, torch.zeros((64, 64)))
    else:
        assert output["samples"] is inputs["latent"]["samples"]
        assert "noise_mask" not in output
        torch.testing.assert_close(mask, torch.ones((1, 8, 8)))


@pytest.fixture
def registration_app(bridge_files, monkeypatch):
    from server import PromptServer
    monkeypatch.setattr(PromptServer, "instance", SimpleNamespace(
        routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
    from impact import impact_server
    for path_type in ("input", "output", "temp"):
        directory = bridge_files / path_type
        directory.mkdir()
        monkeypatch.setattr(bridge_nodes.folder_paths, path_type + "_directory", str(directory))
    app = web.Application()
    app.router.add_get("/impact/set/pb_id_image", impact_server.set_previewbridge_image)
    app.router.add_get("/impact/get/pb_id_image", impact_server.get_previewbridge_image)
    return app


@pytest.mark.parametrize("path_type", ["input", "output", "temp"])
@pytest.mark.parametrize("escape", ["absolute-subfolder", "parent-subfolder", "sibling-subfolder",
                                   "absolute-filename", "parent-filename"])
def test_registration_rejects_escape_before_reading(registration_app, bridge_files, monkeypatch, path_type, escape):
    outside = bridge_files / (path_type + "-other") / "mask.png"
    outside.parent.mkdir()
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(outside)
    filename, subfolder = outside.name, ""
    if escape == "absolute-subfolder":
        subfolder = str(outside.parent)
    elif escape == "parent-subfolder":
        subfolder = "clipspace/../../" + outside.parent.name
        (bridge_files / path_type / "clipspace").mkdir()
    elif escape == "sibling-subfolder":
        subfolder = "../" + outside.parent.name
    elif escape == "absolute-filename":
        filename = str(outside)
    else:
        filename = "../" + outside.parent.name + "/" + outside.name
    opened = []
    original_open = Image.open

    def record_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", record_open)
    async def register():
        async with TestClient(TestServer(registration_app)) as client:
            response = await client.get("/impact/set/pb_id_image", params={
                "node_id": "escape", "filename": filename, "subfolder": subfolder, "type": path_type,
            })
            return response.status

    status = asyncio.run(register())
    assert (status, len(opened), len(core.preview_bridge_image_id_map),
            len(core.preview_bridge_last_mask_cache)) == (400, 0, 0, 0)


@pytest.mark.parametrize("path_type", ["input", "output", "temp"])
@pytest.mark.parametrize("kind", ["nested", "annotated", "file-symlink", "directory-symlink"])
def test_registration_preserves_media_paths_and_masks(registration_app, bridge_files, path_type, kind):
    root = bridge_files / path_type
    directory = root / "clipspace" if kind in ("nested", "annotated") else bridge_files / "other-disk"
    directory.mkdir()
    saved = directory / "mask.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(saved)
    if kind == "file-symlink":
        (root / "clipspace").mkdir()
        (root / "clipspace" / saved.name).symlink_to(saved)
    elif kind == "directory-symlink":
        (root / "clipspace").symlink_to(directory, target_is_directory=True)
    item = {"filename": saved.name, "subfolder": "clipspace", "type": path_type}
    filename = saved.name + (f" [{path_type}]" if kind == "annotated" else "")
    async def register():
        async with TestClient(TestServer(registration_app)) as client:
            response = await client.get("/impact/set/pb_id_image", params={
                **item, "filename": filename, "node_id": "valid",
            })
            assert response.status == 200
            reference = await response.text()
            metadata = await client.get("/impact/get/pb_id_image", params={"id": reference})
            assert await metadata.json() == item
            return reference

    reference = asyncio.run(register())
    pixels, mask, _ = bridge_nodes.PreviewBridge.load_image(reference)
    torch.testing.assert_close(pixels, torch.tensor([1., 0., 0.]).expand(1, 64, 64, 3))
    expected = torch.full((1, 64, 64), 127 / 255)
    torch.testing.assert_close(mask, expected)
    for node_type in ("PreviewBridge", "PreviewBridgeLatent"):
        inputs = {"image": reference, "unique_id": node_type, "restore_mask": "always"}
        if node_type == "PreviewBridge":
            inputs["images"] = torch.zeros((1, 64, 64, 3))
        else:
            inputs.update(latent={"samples": torch.zeros((1, 4, 8, 8))}, preview_method="Latent2RGB-SD15")
        output, mask = getattr(bridge_nodes, node_type)().doit(**inputs)["result"]
        torch.testing.assert_close(mask, expected)
        if node_type == "PreviewBridgeLatent":
            torch.testing.assert_close(output["noise_mask"], expected)
