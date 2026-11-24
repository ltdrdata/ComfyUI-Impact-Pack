import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def masks():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        from impact import core, segs_nodes

        def tiles(mode):
            segs = segs_nodes.MakeTileSEGS.execute(torch.zeros(1, 512, 640, 3), bbox_size=512, crop_factor=1.5, min_overlap=64, filter_segs_dilation=20,
                                                   mask_irregularity=0.7, irregular_mask_mode=mode)[0]
            return [seg.cropped_mask for seg in segs[1]]

        def edge(seed):
            mask = np.zeros((128, 128), dtype=np.float32)
            core.random_mask(mask, (0, 0, 128, 128), factor=0.7, seed=seed)
            return mask

        yield SimpleNamespace(tiles=tiles, edge=edge)


@pytest.mark.parametrize("mode", ["Reuse fast", "All random fast"])
def test_irregular_masks_are_the_same_in_every_run(masks, mode):
    first, again = masks.tiles(mode), masks.tiles(mode)
    assert len(first) == len(again) > 1
    assert all(np.array_equal(mask, other) for mask, other in zip(first, again))


def test_a_tile_of_all_random_gets_an_edge_of_its_own(masks):
    assert np.array_equal(masks.edge(0), masks.edge(0))
    assert not np.array_equal(masks.edge(0), masks.edge(1))
