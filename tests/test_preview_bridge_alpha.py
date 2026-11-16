"""PreviewBridge and the alpha channel of its input, over every short sequence of runs.

`alpha_as_mask` decides what the alpha of an RGBA input is. The rules are checked as properties that hold after
every step of every sequence, not as hand-picked cases:

- an RGB input is not affected by the option at all;
- disabled: an RGBA input gives the mask its RGB part gives, the image output is the input with its alpha on
  every run, and the preview shows the transparency over a checkerboard;
- enabled: the alpha is the mask until the mask is edited, nothing is restored over it, and the image output
  carries the current mask as its alpha; an alpha without any transparency brings no mask;
- the input tensor is never written to.

Where the option meets `restore_mask` and the mask editor, the outcome is listed case by case in SCENARIOS.
Every sequence runs twice: with the reference an API client keeps sending, and with the reference the browser
sends, which follows the file the node displays (js/preview-bridge.js).
"""
from itertools import product

import numpy as np
from PIL import Image
import pytest
import torch

from impact import bridge_nodes, core


SIZE = 16
RESTORE = ["never", "always", "if_same_size"]
CLIENTS = ["api", "browser"]
# What can happen between two runs of the bridge.
EVENTS = ["same", "recomputed", "changed", "resized", "paint", "clear"]
SEQUENCES = list(product(EVENTS, repeat=3))


@pytest.fixture
def bridge_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_nodes.folder_paths, "temp_directory", str(tmp_path))
    for name in ("preview_bridge_cache", "preview_bridge_last_mask_cache", "preview_bridge_image_id_map"):
        monkeypatch.setattr(core, name, {})
    return tmp_path


def picture(value, size=SIZE, alpha=True, opaque=False):
    """A flat picture; with alpha it is transparent in its top-left quarter and half transparent below it."""
    image = torch.full((1, size, size, 4 if alpha else 3), value)
    if alpha:
        image[..., 3] = 1
        if not opaque:
            image[:, :size // 2, :size // 2, 3] = 0
            image[:, size // 2:, :size // 2, 3] = 0.5
    return image


def over_checkerboard(image):
    """What the preview of an RGBA picture must show while the option is off, as 8-bit RGB."""
    rows, columns = np.indices(image.shape[1:3]) // 16
    board = np.where((rows + columns) % 2 == 0, 1.0, 0.8)[..., None].astype(np.float32)
    rgb, alpha = image[0, ..., :3].numpy(), image[0, ..., 3:].numpy()
    return (255.0 * (rgb * alpha + board * (1 - alpha))).clip(0, 255).astype(np.uint8)


def as_file(picture):
    """A picture as the 8-bit RGBA of the file its preview is saved to."""
    data = np.clip(255.0 * picture[0].numpy(), 0, 255).astype(np.uint8)
    return data if data.shape[-1] == 4 else np.dstack((data, np.full(data.shape[:2], 255, np.uint8)))


def reset_bridges():
    for name in ("preview_bridge_cache", "preview_bridge_last_mask_cache", "preview_bridge_image_id_map"):
        getattr(core, name).clear()


def own_mask(image):
    return 1 - image[..., 3]


class Run:
    """One bridge driven through a sequence, with the files the frontend would register."""

    def __init__(self, folder, node_id, restore, alpha=True, rgb_twin=False, block=False, opaque=False, client="api"):
        self.node, self.folder, self.node_id = bridge_nodes.PreviewBridge(), folder, node_id
        self.restore, self.alpha, self.rgb_twin, self.block, self.opaque = restore, alpha, rgb_twin, block, opaque
        self.browser = client == "browser"
        self.reference, self.value, self.size, self.edits = f"${node_id}-0" if self.browser else "", 0.5, SIZE, 0
        self.images = self.given = self.displayed = self.preview = None

    def register(self, path, item):
        self.displayed = path
        self.reference = core.set_previewbridge_image(self.node_id, str(path), item)

    def supply(self):
        """A new tensor for the current picture; the twin gets the RGB part of the same picture."""
        self.given = picture(self.value, self.size, self.alpha, self.opaque)
        self.images = self.given[..., :3].clone() if self.rgb_twin else self.given.clone()

    def edit(self, cleared, mask=None, paint=None, opacity=1.0):
        """Save from the mask editor: a bar across the middle as the mask, nothing when cleared, or the mask given,
        0 to 1. `paint` is where its paint tool draws red at the given opacity.

        The editor hands back the file the node displays with the paint laid over it and the mask as its alpha,
        and saves the paint layer alone beside it."""
        height, width = self.images.shape[1:3]
        with Image.open(self.preview) as preview:
            canvas = np.asarray(preview.convert("RGBA")).copy()
        layer = np.zeros_like(canvas)
        if paint is not None:
            layer[paint.numpy().astype(bool)] = (255, 0, 0, round(255 * opacity))
        cover = layer[..., 3:] / 255
        canvas[..., :3] = (canvas[..., :3] * (1 - cover) + layer[..., :3] * cover).round()
        canvas[..., 3] = 255
        if mask is not None:
            canvas[..., 3] = 255 - (mask.numpy() * 255).round().astype(np.uint8)
        elif not cleared:
            canvas[height // 4:height // 2, width // 4:, 3] = 0
        self.edits += 1
        path = self.folder / f"clipspace-painted-masked-{self.node_id}{self.edits}.png"
        Image.fromarray(canvas, "RGBA").save(path)
        Image.fromarray(layer, "RGBA").save(self.folder / f"clipspace-paint-{self.node_id}{self.edits}.png")
        self.register(path, {"filename": path.name, "subfolder": "", "type": "temp"})
        return 1 - torch.from_numpy(canvas[..., 3].astype(np.float32) / 255)[None]

    def step(self, event, alpha_as_mask):
        edited = None
        if event == "first" or event == "recomputed":
            self.supply()
        elif event == "changed":
            self.value = round(1.25 - self.value, 2)
            self.supply()
        elif event == "resized":
            self.size = SIZE * 3 // 2 if self.size == SIZE else SIZE
            self.supply()
        elif event in ("paint", "clear"):
            edited = self.edit(event == "clear")
        result = self.node.doit(self.images, self.reference, unique_id=self.node_id, block=self.block,
                                restore_mask=self.restore, alpha_as_mask=alpha_as_mask)
        pixels, mask = result["result"]
        self.warned = result["ui"]["alpha_in_mask"]
        item = result["ui"]["images"][0]
        path = self.folder / item["subfolder"] / item["filename"]
        with Image.open(path) as preview:
            shown = np.asarray(preview.convert("RGBA")).copy()
        self.preview = path
        if self.browser and path != self.displayed:
            # The image widget takes the file the node displays, and the next prompt registers it.
            self.register(path, item)
        untouched = torch.equal(self.images, self.given[..., :3] if self.rgb_twin else self.given)
        return pixels, mask, shown, edited, untouched


def blocked(value):
    return not isinstance(value, torch.Tensor)


def same_output(first, second):
    if blocked(first) or blocked(second):
        return blocked(first) and blocked(second)
    return first.shape == second.shape and torch.equal(first, second)


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("block", [False, True], ids=["output", "block"])
def test_an_rgb_input_is_not_affected_by_the_option(bridge_files, restore, block, client):
    for sequence in SEQUENCES:
        runs = [Run(bridge_files, name, restore, alpha=False, block=block, client=client) for name in ("off", "on", "toggled")]
        for index, event in enumerate(("first",) + sequence):
            flags = (False, True, index % 2 == 1)
            (pixels, mask, shown, _, untouched), *others = [run.step(event, flag) for run, flag in zip(runs, flags)]
            assert untouched, (sequence, index)
            for other in others:
                assert same_output(other[0], pixels) and same_output(other[1], mask) and np.array_equal(other[2], shown), (sequence, index)
        reset_bridges()


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("block", [False, True], ids=["output", "block"])
def test_disabled_an_rgba_input_behaves_like_its_rgb_part_and_keeps_its_alpha(bridge_files, restore, block, client):
    for sequence in SEQUENCES:
        rgba = Run(bridge_files, "rgba", restore, block=block, client=client)
        twin = Run(bridge_files, "rgb", restore, rgb_twin=True, block=block, client=client)
        for index, event in enumerate(("first",) + sequence):
            pixels, mask, shown, _, untouched = rgba.step(event, False)
            twin_pixels, twin_mask, twin_shown, _, _ = twin.step(event, False)
            where = (sequence, index, event)
            assert untouched, where
            assert same_output(mask, twin_mask), where
            assert np.array_equal(shown[..., 3], twin_shown[..., 3]), where
            assert np.array_equal(shown[..., :3], over_checkerboard(rgba.images)), where
            assert blocked(pixels) == blocked(twin_pixels), where
            if not blocked(pixels):
                assert torch.equal(pixels, rgba.images), where
        reset_bridges()


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
def test_enabled_the_alpha_is_the_mask_until_it_is_edited(bridge_files, restore, client):
    for sequence in SEQUENCES:
        run = Run(bridge_files, "mask", restore, client=client)
        current = None
        for index, event in enumerate(("first",) + sequence):
            pixels, mask, shown, edited, untouched = run.step(event, True)
            where = (sequence, index, event)
            if edited is not None:
                current = edited
            elif event in ("first", "changed", "resized"):
                # A new picture brings its own mask; nothing is restored over it.
                current = own_mask(run.images)
            assert untouched, where
            torch.testing.assert_close(mask, current, atol=1 / 255, rtol=0, msg=str(where))
            assert pixels.shape[-1] == 4, where
            torch.testing.assert_close(pixels[..., 3], 1 - mask, atol=1 / 255, rtol=0, msg=str(where))
            torch.testing.assert_close(torch.from_numpy(shown[..., 3] / 255).float()[None], 1 - mask, atol=1 / 255, rtol=0, msg=str(where))
        reset_bridges()


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("block", [False, True], ids=["output", "block"])
def test_enabled_an_alpha_without_transparency_brings_no_mask(bridge_files, restore, block, client):
    for sequence in SEQUENCES:
        rgba = Run(bridge_files, "opaque", restore, block=block, opaque=True, client=client)
        twin = Run(bridge_files, "rgb", restore, rgb_twin=True, block=block, opaque=True, client=client)
        for index, event in enumerate(("first",) + sequence):
            pixels, mask, shown, _, untouched = rgba.step(event, True)
            twin_pixels, twin_mask, twin_shown, _, _ = twin.step(event, True)
            where = (sequence, index, event)
            assert untouched, where
            assert same_output(mask, twin_mask), where
            assert np.array_equal(shown, twin_shown), where
            assert blocked(pixels) == blocked(twin_pixels), where
            if not blocked(pixels):
                assert torch.equal(pixels[..., :3], twin_pixels), where
                # The current mask as the preview shows it: a restored mask is resized to the picture there.
                torch.testing.assert_close(pixels[..., 3], torch.from_numpy(shown[..., 3] / 255).float()[None], atol=1 / 255, rtol=0, msg=str(where))
        reset_bridges()


ON, OFF = True, False
RESTORED = ("always", "if_same_size")


def scene(kind, value, size=SIZE):
    """rgb, opaque (RGBA without transparency), A (transparent top-left quarter) or B (bottom-right quarter)."""
    image = torch.full((1, size, size, 3 if kind == "rgb" else 4), value)
    if kind != "rgb":
        image[..., 3] = 1
    if kind == "A":
        image[:, :size // 2, :size // 2, 3] = 0
    if kind == "B":
        image[:, size // 2:, size // 2:, 3] = 0
    return image


def stroke(size=SIZE):
    mask = torch.zeros(size, size)
    mask[size // 4:size // 2, size // 4:] = 1
    return mask


def named(mask):
    """Which of the masks a scenario can produce this one is."""
    plane = mask.reshape(-1, *mask.shape[-2:])[0]
    if not plane.any():
        return "empty"
    known = {"alpha A": 1 - scene("A", 0.0)[0, ..., 3], "alpha B": 1 - scene("B", 0.0)[0, ..., 3], "painted": stroke()}
    return next((name for name, expected in known.items() if plane.shape == expected.shape and torch.equal(plane, expected)), "other")


# Each step: (what happens, the picture given, the option) -> the mask the bridge returns and the alpha of its
# image output ("rgb" for a three-channel output). A dict gives the outcome per restore_mask value.
def by_restore(never, restored):
    return {"never": never, "always": restored, "if_same_size": restored}


SCENARIOS = {
    "the alpha of a new image replaces a painted mask": [
        (("first", "rgb", ON), ("empty", "rgb")),
        (("paint", "rgb", ON), ("painted", "rgb")),
        (("new", "A", ON), ("alpha A", "alpha A")),
    ],
    "an alpha that was the mask is restored onto an image without alpha, like any last mask": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("new", "rgb", ON), by_restore(("empty", "rgb"), ("alpha A", "rgb"))),
    ],
    "and onto an RGBA image without transparency, whose output then carries it": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("new", "opaque", ON), by_restore(("empty", "empty"), ("alpha A", "alpha A"))),
    ],
    "the alpha of a new image replaces an edit of the previous alpha": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("paint", "A", ON), ("painted", "painted")),
        (("new", "B", ON), ("alpha B", "alpha B")),
    ],
    "an edit of the alpha is restored onto an image without alpha": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("paint", "A", ON), ("painted", "painted")),
        (("new", "rgb", ON), by_restore(("empty", "rgb"), ("painted", "rgb"))),
    ],
    "switching off keeps an edited mask and gives the image its own alpha back": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("paint", "A", ON), ("painted", "painted")),
        (("same", "A", OFF), ("painted", "alpha A")),
        (("new", "B", OFF), by_restore(("empty", "alpha B"), ("painted", "alpha B"))),
    ],
    "switching off drops a mask that is still the alpha, and it is not restored afterwards": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("same", "A", OFF), ("empty", "alpha A")),
        (("new", "rgb", OFF), ("empty", "rgb")),
        (("new", "A", OFF), ("empty", "alpha A")),
    ],
    "it is still the alpha after the bridge ran again on the image": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("same", "A", ON), ("alpha A", "alpha A")),
        (("recompute", "A", ON), ("alpha A", "alpha A")),
        (("same", "A", OFF), ("empty", "alpha A")),
        (("same", "A", OFF), ("empty", "alpha A")),
    ],
    "switched off together with a new image, the previous alpha is not restored either": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("new", "B", OFF), ("empty", "alpha B")),
        (("new", "rgb", OFF), ("empty", "rgb")),
    ],
    "disabled, a painted mask is restored and the alpha of each image is passed on": [
        (("first", "A", OFF), ("empty", "alpha A")),
        (("paint", "A", OFF), ("painted", "alpha A")),
        (("new", "B", OFF), by_restore(("empty", "alpha B"), ("painted", "alpha B"))),
    ],
    "switching on reads the alpha of the image at once, over a painted mask": [
        (("first", "A", OFF), ("empty", "alpha A")),
        (("paint", "A", OFF), ("painted", "alpha A")),
        (("same", "A", ON), ("alpha A", "alpha A")),
        (("same", "A", ON), ("alpha A", "alpha A")),
    ],
    "a mask cleared in the editor stays cleared for that image; the next image brings its alpha again": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("clear", "A", ON), ("empty", "empty")),
        (("recompute", "A", ON), ("empty", "empty")),
        (("new", "A", ON), ("alpha A", "alpha A")),
    ],
    "an alpha without transparency does not replace a painted mask, switched on or off": [
        (("first", "opaque", ON), ("empty", "empty")),
        (("paint", "opaque", ON), ("painted", "painted")),
        (("same", "opaque", OFF), ("painted", "empty")),
        (("same", "opaque", ON), ("painted", "painted")),
        (("new", "opaque", ON), by_restore(("empty", "empty"), ("painted", "painted"))),
    ],
    "an RGB input is not affected by switching": [
        (("first", "rgb", OFF), ("empty", "rgb")),
        (("paint", "rgb", OFF), ("painted", "rgb")),
        (("same", "rgb", ON), ("painted", "rgb")),
        (("new", "rgb", ON), by_restore(("empty", "rgb"), ("painted", "rgb"))),
        (("same", "rgb", OFF), by_restore(("empty", "rgb"), ("painted", "rgb"))),
    ],
    "a restored alpha is the bridge's mask from then on and survives switching off": [
        (("first", "A", ON), ("alpha A", "alpha A")),
        (("new", "opaque", ON), by_restore(("empty", "empty"), ("alpha A", "alpha A"))),
        (("same", "opaque", OFF), by_restore(("empty", "empty"), ("alpha A", "empty"))),
    ],
}
# The reference an API client keeps sending still names the painted file after another image brought its alpha;
# the browser's reference follows the displayed file, so there the alpha is the last mask.
LAST_MASK = {
    "api": by_restore(("empty", "empty"), ("painted", "painted")),
    "browser": by_restore(("empty", "empty"), ("alpha A", "alpha A")),
}
AFTER_AN_ALPHA = [
    (("first", "opaque", ON), ("empty", "empty")),
    (("paint", "opaque", ON), ("painted", "painted")),
    (("new", "A", ON), ("alpha A", "alpha A")),
    (("new", "opaque", ON), LAST_MASK),
]


def play(folder, steps, restore, client, block=False):
    """Run a scenario; return (mask name, image alpha name) per step, or "blocked"."""
    run = Run(folder, "scenario", restore, client=client, block=block)
    value, outcome = 0.5, []
    for event, kind, flag in steps:
        if event == "new":
            value = round(1.25 - value, 2)
        if event in ("first", "new", "recompute"):
            run.given = scene(kind, value)
            run.images = run.given.clone()
        pixels, mask, shown, _, untouched = run.step(event if event in ("paint", "clear") else "same", flag)
        assert untouched, (event, kind, flag)
        if blocked(pixels):
            outcome.append("blocked")
            continue
        # The preview is what the mask editor opens: its alpha is the mask.
        assert named(1 - torch.from_numpy(shown[..., 3] / 255).float()) == named(mask), (event, kind, flag)
        outcome.append((named(mask), "rgb" if pixels.shape[-1] == 3 else named(1 - pixels[..., 3])))
    return outcome


def expected(steps, restore, client):
    outcome = []
    for _, result in steps:
        if isinstance(result, dict) and client in result:
            result = result[client]
        outcome.append(result[restore] if isinstance(result, dict) else result)
    return outcome


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("title", SCENARIOS)
def test_the_option_with_restore_mask_and_the_editor(bridge_files, title, restore, client):
    steps = SCENARIOS[title]
    assert play(bridge_files, [step for step, _ in steps], restore, client) == expected(steps, restore, client)


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
def test_where_a_painted_mask_and_the_transparent_area_overlap(bridge_files, restore, client):
    """The stroke crosses the edge of the transparent area: half of it lies on it, half beside it."""
    alpha, painted = 1 - scene("A", 0.0)[0, ..., 3], stroke()
    both, beside = (alpha * painted).bool(), (painted * (1 - alpha)).bool()
    assert both.any() and beside.any() and (alpha * (1 - painted)).any()

    # Disabled: the mask is what was painted, also on the transparent area, and the image's alpha is not touched by it.
    run = Run(bridge_files, "apart", restore, client=client)
    run.given = scene("A", 0.5)
    run.images = run.given.clone()
    run.step("same", False)
    run.edit(False)
    pixels, mask, _, _, untouched = run.step("same", False)
    assert untouched and torch.equal(mask[0], painted) and torch.equal(1 - pixels[0, ..., 3], alpha)

    # Enabled, the editor opens with the transparent area as its mask. Painting adds to it: the image becomes
    # transparent beside the area too, and stays transparent where the stroke lies on it.
    run = Run(bridge_files, "one", restore, client=client)
    run.given = scene("A", 0.5)
    run.images = run.given.clone()
    run.step("same", True)
    run.edit(False, mask=torch.maximum(alpha, painted))
    pixels, mask, _, _, untouched = run.step("same", True)
    assert untouched and torch.equal(mask[0], torch.maximum(alpha, painted)) and torch.equal(1 - pixels[0, ..., 3], mask[0])
    # Erasing takes from it: the image is opaque again where the mask was erased.
    run.edit(False, mask=alpha * (1 - painted))
    pixels, mask, _, _, untouched = run.step("same", True)
    assert untouched and torch.equal(mask[0], alpha * (1 - painted))
    assert pixels[0, ..., 3][both].eq(1).all() and pixels[0, ..., 3][(alpha * (1 - painted)).bool()].eq(0).all()


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
def test_a_mask_saved_while_enabled_stays_as_saved_when_the_option_is_switched_off(bridge_files, restore, client):
    """The editor opens with the alpha as its mask and saves the two as one mask; that mask is the user's."""
    alpha, painted = 1 - scene("A", 0.0)[0, ..., 3], stroke()
    saved = torch.maximum(alpha, painted)
    run = Run(bridge_files, "saved", restore, client=client)
    run.given = scene("A", 0.5)
    run.images = run.given.clone()
    run.step("same", True)
    run.edit(False, mask=saved)
    run.step("same", True)
    assert run.warned == [False]
    for warned in (True, False):
        pixels, mask, shown, _, untouched = run.step("same", False)
        # The alpha area is still in the mask; the image has its own alpha back.
        assert untouched and torch.equal(mask[0], saved) and torch.equal(1 - pixels[0, ..., 3], alpha)
        assert torch.equal(1 - torch.from_numpy(shown[..., 3] / 255).float(), saved)
        # The file is now what the editor gets while the option is off: the transparency over a checkerboard.
        assert np.array_equal(shown[..., :3], over_checkerboard(run.images))
        # The frontend is told once, on the run that follows the switch.
        assert run.warned == [warned]
    # Not saved in the editor, the alpha is dropped instead, and there is nothing to warn about.
    fresh = Run(bridge_files, "unsaved", restore, client=client)
    fresh.given = scene("A", 0.5)
    fresh.images = fresh.given.clone()
    fresh.step("same", True)
    assert not fresh.step("same", False)[1].any() and fresh.warned == [False]


def painted_over(image, paint, opacity):
    """Red laid over an RGBA picture at the given opacity where `paint` is set: what the image output must be."""
    cover = paint[None, ..., None] * opacity
    alpha = image[..., 3:]
    red = torch.tensor([1.0, 0.0, 0.0])
    both = cover + alpha * (1 - cover)
    colour = (red * cover + image[..., :3] * alpha * (1 - cover)) / both.clamp(min=1e-6)
    return torch.where(cover > 0, torch.cat((colour, both), dim=-1), image)


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("switched", [False, True], ids=["disabled all along", "switched off"])
@pytest.mark.parametrize("opacity", [1.0, 0.4], ids=["solid", "soft"])
def test_disabled_the_image_output_is_the_input_with_the_paint_of_the_editor_laid_over_it(bridge_files, restore, client, switched, opacity):
    """The file shows a checkerboard where the picture is transparent. None of it reaches the image output: the
    image is the input, and what the editor's paint tool drew is taken from the paint layer it saves."""
    run = Run(bridge_files, "drawn", restore, client=client)
    run.step("first", switched)
    drawn = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    # Across the transparent quarter, the half transparent quarter and the opaque half.
    drawn[2:4, 2:14] = drawn[10:12, 2:14] = True
    saved = run.edit(False, paint=drawn, opacity=opacity)
    expected = painted_over(run.images, drawn, round(255 * opacity) / 255)
    for _ in range(3):
        pixels, mask, shown, _, untouched = run.step("same", False)
        assert untouched and torch.equal(mask, saved)
        assert torch.equal(pixels[0][~drawn], run.images[0][~drawn])
        torch.testing.assert_close(pixels, expected, atol=1e-6, rtol=0)
        # The file is the one the editor saved; the paint cannot be told from the picture in a file written again.
        assert run.preview == run.displayed


def test_disabled_without_a_paint_layer_the_image_output_is_the_input(bridge_files):
    """A file that an API client refers to, or one of an editor without paint tools, has no layer beside it."""
    run = Run(bridge_files, "plain", "never")
    run.step("first", False)
    drawn = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    drawn[2:4, 2:14] = True
    saved = run.edit(False, paint=drawn)
    layer = bridge_files / f"clipspace-paint-plain{run.edits}.png"
    # A layer that does not fit the image is not laid over it either.
    Image.new("RGBA", (SIZE * 2, SIZE * 2), (255, 0, 0, 255)).save(layer)
    pixels, mask, _, _, _ = run.step("same", False)
    assert torch.equal(pixels, run.images) and torch.equal(mask, saved)
    layer.unlink()
    pixels, mask, _, _, _ = run.step("recomputed", False)
    assert torch.equal(pixels, run.images) and torch.equal(mask, saved)


def test_switching_off_over_a_file_of_another_size_leaves_that_file_alone(bridge_files):
    """An API client can refer to any file. One that does not fit the image is passed on as before, and is not
    written again or warned about."""
    run = Run(bridge_files, "size", "never")
    run.step("first", True)
    path = bridge_files / "other-size.png"
    canvas = np.full((SIZE * 2, SIZE * 2, 4), 128, dtype=np.uint8)
    canvas[:SIZE, :, 3] = 0
    Image.fromarray(canvas, "RGBA").save(path)
    run.register(path, {"filename": path.name, "subfolder": "", "type": "temp"})
    run.step("same", True)
    pixels, mask, _, _, untouched = run.step("same", False)
    assert untouched and pixels.shape == (1, SIZE * 2, SIZE * 2, 3) and mask.shape == (1, SIZE * 2, SIZE * 2)
    assert run.preview == path and run.warned == [False]


def test_disabled_the_image_of_a_batch_that_the_editor_was_given_is_passed_on(bridge_files):
    """Every image of a batch is shown over the checkerboard, and the editor is given the one that is selected."""
    node = bridge_nodes.PreviewBridge()
    images = torch.cat((scene("A", 0.25), scene("B", 0.75), scene("A", 0.5)))
    original = images.clone()
    shown = node.doit(images, "", unique_id="batch")["ui"]["images"]
    assert len(shown) == 3
    for index, item in enumerate(shown):
        with Image.open(bridge_files / item["subfolder"] / item["filename"]) as preview:
            assert np.array_equal(np.asarray(preview.convert("RGB")), over_checkerboard(images[index:index + 1]))
    # The second image is opened in the editor, and a mask is saved over it.
    with Image.open(bridge_files / shown[1]["subfolder"] / shown[1]["filename"]) as preview:
        canvas = np.asarray(preview.convert("RGBA")).copy()
    canvas[:4, :, 3] = 0
    path = bridge_files / "clipspace-painted-masked-7.png"
    Image.fromarray(canvas, "RGBA").save(path)
    reference = core.set_previewbridge_image("batch", str(path), {"filename": path.name, "subfolder": "", "type": "temp"})
    pixels, mask = node.doit(images, reference, unique_id="batch")["result"]
    assert torch.equal(images, original) and torch.equal(pixels, images[1:2])
    assert mask.shape == (1, SIZE, SIZE) and mask[0, :4].all() and not mask[0, 4:].any()


@pytest.mark.parametrize("restore", RESTORE)
def test_disabled_values_outside_the_range_of_a_file_are_passed_on_too(bridge_files, restore):
    node = bridge_nodes.PreviewBridge()
    images = picture(0.5)
    images[:, :, :4, :3] = 1.5
    images[:, :, 4:8, :3] = -0.25
    original = images.clone()
    for _ in range(3):
        pixels, mask = node.doit(images, "", unique_id="range", restore_mask=restore)["result"]
        assert torch.equal(images, original) and torch.equal(pixels, images) and not mask.any()


def test_the_checkerboard_has_squares_of_sixteen_pixels():
    shown = bridge_nodes.over_checkerboard(torch.zeros(1, 48, 48, 4))[0, ..., 0]
    cells = shown.reshape(3, 16, 3, 16)
    light, grey = torch.tensor(1.0), torch.tensor(0.8)
    assert all(torch.equal(cells[row, :, column], (light if (row + column) % 2 == 0 else grey).expand(16, 16))
               for row in range(3) for column in range(3))


@pytest.mark.parametrize("client", CLIENTS)
def test_a_mask_kept_after_switching_off_keeps_its_levels(bridge_files, client):
    """The file is written again when the option is switched off; the mask in it stays the one that was saved."""
    run = Run(bridge_files, "levels", "never", client=client)
    run.step("first", True)
    levels = torch.arange(SIZE * SIZE, dtype=torch.float32).reshape(SIZE, SIZE) / 255
    saved = run.edit(False, mask=levels)
    run.step("same", True)
    # The picture shown as soon as the option is switched has the mask at the same levels.
    assert np.array_equal(as_file(bridge_nodes.PreviewBridge.alpha_preview("levels", False, 0))[..., 3], 255 - np.arange(SIZE * SIZE, dtype=np.uint8).reshape(SIZE, SIZE))
    for _ in range(3):
        _, mask, shown, _, _ = run.step("same", False)
        assert torch.equal(mask, saved)
        assert np.array_equal(shown[..., 3], 255 - np.arange(SIZE * SIZE, dtype=np.uint8).reshape(SIZE, SIZE))


@pytest.mark.parametrize("kind,area,warned", [("A", "beside", False), ("A", "on", True), ("opaque", "on", False), ("rgb", "on", False)],
                         ids=["mask beside the transparent area", "mask on it", "no transparency", "rgb"])
def test_the_warning_is_for_a_kept_mask_that_covers_transparent_pixels(bridge_files, kind, area, warned):
    run = Run(bridge_files, "warn", "never")
    run.given = scene(kind, 0.5)
    run.images = run.given.clone()
    run.step("same", True)
    mask = torch.zeros(SIZE, SIZE)
    if area == "on":
        mask[:4, :4] = 1
    else:
        mask[SIZE // 2:, SIZE // 2:] = 1
    run.edit(False, mask=mask)
    run.step("same", True)
    saved = run.preview
    run.step("same", False)
    assert run.warned == [warned]
    # Only a picture with transparency is written again to show it.
    assert (run.preview != saved) == (kind == "A")


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
def test_the_last_mask_after_an_image_brought_its_alpha(bridge_files, restore, client):
    assert play(bridge_files, [step for step, _ in AFTER_AN_ALPHA], restore, client) == expected(AFTER_AN_ALPHA, restore, client)


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("title", SCENARIOS)
def test_blocking_follows_the_mask_in_every_scenario(bridge_files, title, restore, client):
    steps = SCENARIOS[title]
    stopped = ["blocked" if result[0] == "empty" else result for result in expected(steps, restore, client)]
    assert play(bridge_files, [step for step, _ in steps], restore, client, block=True) == stopped


@pytest.mark.parametrize("flag,alpha", [(False, "alpha A"), (True, "empty")], ids=["disabled", "enabled"])
def test_a_reference_to_a_file_without_alpha_is_no_mask(bridge_files, flag, alpha):
    images = scene("A", 0.5)
    node = bridge_nodes.PreviewBridge()
    node.doit(images, "", unique_id="plain", alpha_as_mask=flag)
    path = bridge_files / "plain.png"
    Image.new("RGB", (SIZE, SIZE), (128, 128, 128)).save(path)
    reference = core.set_previewbridge_image("plain", str(path), {"filename": path.name, "subfolder": "", "type": "temp"})
    pixels, mask = node.doit(images, reference, unique_id="plain", alpha_as_mask=flag)["result"]
    # Enabled, the image output carries the current mask, which is empty; disabled, the input's alpha.
    assert (named(mask), named(1 - pixels[..., 3])) == ("empty", alpha)


@pytest.mark.parametrize("restore", RESTORE)
def test_the_kind_of_input_can_change_between_runs(bridge_files, restore):
    node = bridge_nodes.PreviewBridge()
    rgb, rgba = picture(0.25, alpha=False), picture(0.75)
    for index, images in enumerate((rgb, rgba, rgb, rgba)):
        pixels, mask = node.doit(images, "", unique_id="kind", restore_mask=restore, alpha_as_mask=True)["result"]
        assert torch.equal(pixels, images)
        if images is rgba:
            torch.testing.assert_close(mask, own_mask(rgba))
        elif index and restore != "never":
            # The alpha the RGBA input brought is the last mask the bridge had.
            torch.testing.assert_close(mask, own_mask(rgba), atol=1 / 255, rtol=0)
        else:
            assert not mask.any()


@pytest.mark.parametrize("flag", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("restore", RESTORE)
def test_every_frame_of_an_rgba_batch_keeps_its_own_alpha(bridge_files, restore, flag):
    images = torch.cat((picture(0.25), picture(0.75).flip(1)))
    original = images.clone()
    pixels, mask = bridge_nodes.PreviewBridge().doit(images, "", unique_id="batch", restore_mask=restore, alpha_as_mask=flag)["result"]
    assert torch.equal(images, original) and torch.equal(pixels, original)
    if flag:
        # The mask is one mask whatever the size of the batch: that of the first image, which the node shows.
        torch.testing.assert_close(mask, own_mask(original)[:1])
    else:
        assert not mask.any()


def test_enabled_every_image_of_a_batch_is_shown_with_its_own_alpha(bridge_files):
    """The first image is opaque here, so the mask is empty; the second brings its mask once it is selected."""
    node = bridge_nodes.PreviewBridge()
    images = torch.cat((scene("opaque", 0.25), scene("B", 0.75)))
    result = node.doit(images, "", unique_id="shown", alpha_as_mask=True)
    pixels, mask = result["result"]
    assert torch.equal(pixels, images) and mask.shape == (1, SIZE, SIZE) and not mask.any()
    files = [bridge_files / item["subfolder"] / item["filename"] for item in result["ui"]["images"]]
    for index, path in enumerate(files):
        with Image.open(path) as preview:
            assert named(1 - torch.from_numpy(np.asarray(preview.convert("RGBA"))[..., 3] / 255).float()) == ("empty", "alpha B")[index]
    item = result["ui"]["images"][1]
    reference = core.set_previewbridge_image("shown", str(files[1]), item)
    _, mask = node.doit(images, reference, unique_id="shown", alpha_as_mask=True)["result"]
    assert mask.shape == (1, SIZE, SIZE) and named(mask) == "alpha B"


@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("selected", [0, 1], ids=["first image", "second image"])
def test_the_alpha_of_the_image_selected_in_a_batch_is_dropped_when_the_option_is_switched_off(bridge_files, restore, selected):
    node = bridge_nodes.PreviewBridge()
    images = torch.cat((scene("A", 0.25), scene("B", 0.75)))
    item = node.doit(images, "", unique_id="picked", restore_mask=restore, alpha_as_mask=True)["ui"]["images"][selected]
    reference = core.set_previewbridge_image("picked", str(bridge_files / item["subfolder"] / item["filename"]), item)
    _, mask = node.doit(images, reference, unique_id="picked", restore_mask=restore, alpha_as_mask=True)["result"]
    assert named(mask) == ("alpha A", "alpha B")[selected]
    result = node.doit(images, reference, unique_id="picked", restore_mask=restore, alpha_as_mask=False)
    pixels, mask = result["result"]
    assert not mask.any() and result["ui"]["alpha_in_mask"] == [False]
    assert torch.equal(pixels, images)


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", ["always", "if_same_size"])
@pytest.mark.parametrize("kind", ["rgb", "opaque"])
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_the_alpha_a_batch_brought_is_restored_as_the_mask_of_its_first_image(bridge_files, restore, kind, batch, client):
    run = Run(bridge_files, "batch", restore, client=client)
    run.given = torch.cat((scene("A", 0.25), scene("B", 0.75)))
    run.images = run.given.clone()
    _, mask, _, _, _ = run.step("same", True)
    assert [named(frame) for frame in mask] == ["alpha A"]
    run.given = scene(kind, 0.5).repeat(batch, 1, 1, 1)
    run.images = run.given.clone()
    pixels, mask, shown, _, untouched = run.step("same", True)
    assert untouched and pixels.shape[0] == batch
    assert (mask.shape[0], named(mask), named(1 - torch.from_numpy(shown[..., 3] / 255).float())) == (1, "alpha A", "alpha A")
    if kind == "opaque":
        # Enabled, every image of the output carries the restored mask as its alpha.
        assert [named(1 - frame[..., 3]) for frame in pixels] == ["alpha A"] * batch


@pytest.mark.parametrize("flag", [False, True], ids=["disabled", "enabled"])
def test_an_rgba_batch_the_bridge_runs_on_again_passes_on_its_first_image(bridge_files, flag):
    node = bridge_nodes.PreviewBridge()
    images = torch.cat((scene("A", 0.25), scene("B", 0.75)))
    original = images.clone()
    node.doit(images, "", unique_id="again", alpha_as_mask=flag)
    pixels, mask = node.doit(images, "", unique_id="again", alpha_as_mask=flag)["result"]
    assert torch.equal(images, original)
    assert pixels.shape == images[:1].shape and named(mask) == ("alpha A" if flag else "empty")
    if flag:
        # Enabled, the image comes from the file and carries the mask as its alpha, as a single image does.
        torch.testing.assert_close(pixels, images[:1], atol=1 / 255, rtol=0)
        assert torch.equal(pixels[..., 3], 1 - mask)
    else:
        # Disabled, the image is taken from the input: the first one, which the node displays.
        assert torch.equal(pixels, images[:1])


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("restore", RESTORE)
@pytest.mark.parametrize("edited", [False, True], ids=["no edit", "after an edit"])
@pytest.mark.parametrize("flag", [False, True], ids=["switched on", "switched off"])
def test_the_picture_shown_when_the_option_is_switched_is_the_one_the_next_run_shows(bridge_files, flag, edited, restore, client):
    run = Run(bridge_files, "switch", restore, client=client)
    run.step("first", flag)
    if edited:
        run.step("paint", flag)
    files = sorted(bridge_files.rglob("*.png"))
    picture = bridge_nodes.PreviewBridge.alpha_preview("switch", not flag, 0)
    # Nothing is written for it, and a mask is in it only where the next run keeps one.
    assert sorted(bridge_files.rglob("*.png")) == files and picture.shape[-1] == (4 if not flag or edited else 3)
    _, _, shown, _, _ = run.step("same", not flag)
    assert np.array_equal(as_file(picture), shown)


def test_the_picture_shown_when_the_option_is_switched_is_of_the_image_asked_for(bridge_files):
    node = bridge_nodes.PreviewBridge()
    images = torch.cat((scene("A", 0.25), scene("B", 0.75)))
    assert node.doit(images, "", unique_id="pick", alpha_as_mask=False)["ui"]["alpha_as_mask"] == [False]
    for index in range(2):
        assert torch.equal(bridge_nodes.PreviewBridge.alpha_preview("pick", True, index), images[index:index + 1])
    assert node.doit(images, "", unique_id="pick", alpha_as_mask=True)["ui"]["alpha_as_mask"] == [True]
    for index in range(2):
        assert np.array_equal(as_file(bridge_nodes.PreviewBridge.alpha_preview("pick", False, index))[..., :3], over_checkerboard(images[index:index + 1]))
    assert bridge_nodes.PreviewBridge.alpha_preview("pick", False, 2) is None
    assert bridge_nodes.PreviewBridge.alpha_preview("pick", False, -1) is None


def test_a_mask_kept_at_the_switch_is_shown_on_the_image_of_the_batch_it_was_saved_for(bridge_files):
    run = Run(bridge_files, "kept", "never")
    run.given = torch.cat((scene("A", 0.25), scene("B", 0.75)))
    run.images = run.given.clone()
    second = run.node.doit(run.images, "", unique_id="kept", alpha_as_mask=True)["ui"]["images"][1]
    run.preview = bridge_files / second["subfolder"] / second["filename"]
    run.edit(False)
    run.step("same", True)
    picture = bridge_nodes.PreviewBridge.alpha_preview("kept", False, 0)
    _, _, shown, _, _ = run.step("same", False)
    assert np.array_equal(as_file(picture), shown) and np.array_equal(shown[..., :3], over_checkerboard(run.images[1:]))


@pytest.mark.parametrize("kept", ["paint", "size"])
def test_a_file_the_next_run_leaves_as_it_is_has_no_other_picture_to_show(bridge_files, kept):
    run = Run(bridge_files, "left", "never")
    run.step("first", True)
    if kept == "paint":
        run.edit(False, paint=stroke())
    else:
        # A client other than the editor can register a file of any size.
        path = bridge_files / "clipspace-painted-masked-other.png"
        Image.new("RGBA", (SIZE * 2, SIZE * 2), (10, 20, 30, 128)).save(path)
        run.register(path, {"filename": path.name, "subfolder": "", "type": "temp"})
    run.step("same", True)
    held = run.preview
    assert bridge_nodes.PreviewBridge.alpha_preview("left", False, 0) is None
    run.step("same", False)
    assert run.preview == held


def test_the_picture_of_the_setting_a_node_ran_with_is_the_plain_one(bridge_files):
    # The frontend shows the file of the run for that setting; asked for anyway, no mask of the editor is in it.
    run = Run(bridge_files, "plain", "never")
    run.step("first", False)
    run.step("paint", False)
    assert np.array_equal(as_file(bridge_nodes.PreviewBridge.alpha_preview("plain", False, 0))[..., :3], over_checkerboard(run.images))
    assert bridge_nodes.PreviewBridge.alpha_preview("plain", False, 0).shape[-1] == 3


def test_a_latent_bridge_that_ran_under_the_same_id_has_no_picture_to_show(bridge_files):
    core.preview_bridge_cache["shared"] = ({"samples": torch.zeros((1, 4, 8, 8))}, "Latent2RGB-SD15", None, "", "$shared-1")
    assert bridge_nodes.PreviewBridge.alpha_preview("shared", True, 0) is None


@pytest.mark.parametrize("kind", ["rgb", "opaque"])
@pytest.mark.parametrize("flag", [False, True], ids=["disabled", "enabled"])
def test_an_input_without_transparency_has_no_other_picture_to_show(bridge_files, kind, flag):
    bridge_nodes.PreviewBridge().doit(scene(kind, 0.5), "", unique_id="plain", alpha_as_mask=flag)
    assert bridge_nodes.PreviewBridge.alpha_preview("plain", not flag, 0) is None
    assert bridge_nodes.PreviewBridge.alpha_preview("never ran", not flag, 0) is None


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("edited", [False, True], ids=["no edit", "after an edit"])
def test_disabled_a_run_on_the_same_image_writes_no_new_file(bridge_files, client, edited):
    run = Run(bridge_files, "same", "never", client=client)
    run.step("first", False)
    if edited:
        run.step("paint", False)
    shown = run.displayed if edited else run.preview
    files = sorted(bridge_files.rglob("*.png"))
    for _ in range(2):
        run.step("same", False)
        assert run.preview == shown
    assert sorted(bridge_files.rglob("*.png")) == files


@pytest.mark.parametrize("client", CLIENTS)
def test_the_displayed_file_is_recognised_whatever_form_the_temp_path_has(bridge_files, monkeypatch, client):
    # The frontend registers the displayed file by its absolute path; the bridge joins the temp directory as given.
    monkeypatch.setattr(bridge_nodes.folder_paths, "temp_directory", f"{bridge_files}/./")
    run = Run(bridge_files, "path", "always", client=client)
    run.given = scene("A", 0.5)
    run.images = run.given.clone()
    assert named(run.step("same", True)[1]) == "alpha A"
    assert named(run.step("same", False)[1]) == "empty"


def test_an_alpha_that_is_only_partly_transparent_is_a_mask_too(bridge_files):
    images = scene("opaque", 0.5)
    images[..., 3] = 0.75
    pixels, mask = bridge_nodes.PreviewBridge().doit(images, "", unique_id="partly", alpha_as_mask=True)["result"]
    torch.testing.assert_close(mask, torch.full((1, SIZE, SIZE), 0.25))
    assert torch.equal(pixels, images)


@pytest.mark.parametrize("flag,stops", [(False, True), (True, False)], ids=["disabled", "enabled"])
def test_blocking_follows_the_mask_the_option_selects(bridge_files, flag, stops):
    run = Run(bridge_files, "block", "never", block=True)
    pixels, mask, _, _, _ = run.step("first", flag)
    assert (blocked(pixels), blocked(mask)) == (stops, stops)
    opaque = Run(bridge_files, "opaque", "never", block=True, opaque=True)
    pixels, mask, _, _, _ = opaque.step("first", flag)
    assert blocked(pixels) and blocked(mask)
    pixels, mask, _, _, _ = opaque.step("paint", flag)
    assert not blocked(pixels) and mask.any()


def test_the_option_is_the_last_widget_and_defaults_to_disabled(bridge_files):
    optional = bridge_nodes.PreviewBridge.INPUT_TYPES()["optional"]
    assert list(optional) == ["block", "restore_mask", "alpha_as_mask"]
    assert optional["alpha_as_mask"][0] == "BOOLEAN" and optional["alpha_as_mask"][1]["default"] is False
    # A prompt saved before the option existed does not send it.
    images = picture(0.5)
    without = bridge_nodes.PreviewBridge().doit(images, "", unique_id="old")["result"]
    disabled = bridge_nodes.PreviewBridge().doit(images, "", unique_id="new", alpha_as_mask=False)["result"]
    assert all(torch.equal(a, b) for a, b in zip(without, disabled))
