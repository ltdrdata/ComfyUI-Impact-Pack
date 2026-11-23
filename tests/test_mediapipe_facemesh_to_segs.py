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
        from impact import segs_nodes
        yield SimpleNamespace(segs=lambda image: segs_nodes.MediaPipeFaceMeshToSEGS.execute(
            image, 3.0, False, 50, 1, 0, True, False, False, False, False, False, False, False).result[0])


def mesh(frames):
    # The face oval in the colour comfyui_controlnet_aux draws it in.
    image = torch.zeros((frames, 64, 64, 3))
    image[:, 6:58, 10:54] = (torch.tensor((0x0A, 0xC8, 0x0A)) + 0.5) / 255
    return image


def test_one_frame_gives_the_seg_of_its_face(runtime):
    shape, (face,) = runtime.segs(mesh(1))

    assert shape == (64, 64)
    assert (face.label, tuple(face.bbox)) == ("face", (10, 6, 54, 58))


def test_a_batch_of_frames_is_refused_with_the_name_of_the_node(runtime):
    with pytest.raises(Exception, match="MediaPipeFaceMeshToSEGS does not allow image batches"):
        runtime.segs(mesh(2))
