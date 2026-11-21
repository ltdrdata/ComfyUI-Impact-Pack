import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import impact_server  # noqa: F401
        from impact import segs_nodes
        yield SimpleNamespace(
            combined=lambda segs: segs_nodes.SegsToCombinedMask.execute(segs).result[0],
            segs=lambda mask: segs_nodes.MaskToSEGS.execute(mask, False, 1.5, False, 3).result[0],
            video_segs=lambda mask: segs_nodes.MaskToSEGS_for_AnimateDiff.execute(mask, False, 1.5, False, 3).result[0],
            merge=lambda segs: segs_nodes.SEGSMerge.execute(segs).result[0])


def two_blobs():
    mask = torch.zeros((96, 160))
    mask[10:30, 20:50] = 1.0
    mask[50:80, 100:140] = 1.0
    return mask


def test_segs_of_a_mask_give_the_mask_back(runtime):
    mask = two_blobs()

    combined = runtime.combined(runtime.segs(mask))

    assert combined.shape == (1, 96, 160)
    assert torch.equal(combined[0], mask)


def test_the_seg_of_segs_merge_gives_the_mask_of_the_segs_it_merged(runtime):
    mask = two_blobs()
    segs = runtime.segs(mask)
    merged = runtime.merge(segs)
    assert len(segs[1]) == 2 and len(merged[1]) == 1

    assert torch.equal(runtime.combined(merged), runtime.combined(segs))


def test_segs_of_a_video_give_what_the_masks_of_all_frames_cover(runtime):
    frames = torch.zeros((3, 96, 160))
    frames[0, 10:30, 20:50] = 1.0
    frames[1, 15:35, 30:60] = 1.0
    frames[2, 50:80, 100:140] = 1.0
    segs = runtime.video_segs(frames)
    assert [len(seg.cropped_mask) for seg in segs[1]] == [3, 3]

    combined = runtime.combined(segs)

    assert combined.shape == (1, 96, 160)
    assert torch.equal(combined[0], frames.max(dim=0).values)
