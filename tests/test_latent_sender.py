import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def sender(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        import folder_paths
        patch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
        from impact import impact_server  # noqa: F401
        from impact import impact_pack

        def send(latent, preview_method):
            with torch.inference_mode():
                sent = impact_pack.LatentSender().doit({"samples": latent}, link_id=7, preview_method=preview_method)["ui"]["images"][0]
            assert server.send_sync.call_args.args == ("latent-send", {"link_id": 7, "images": [sent]})
            return impact_pack.LatentReceiver.load_preview_latent(str(tmp_path / sent["subfolder"] / sent["filename"]))["samples"]

        yield SimpleNamespace(send=send, methods=impact_pack.LatentSender.INPUT_TYPES()["required"]["preview_method"][0])


METHODS = ["Latent2RGB-FLUX.1", "Latent2RGB-SDXL", "Latent2RGB-SD15", "Latent2RGB-SD3", "Latent2RGB-SD-X4", "Latent2RGB-Playground-2.5",
           "Latent2RGB-SC-Prior", "Latent2RGB-SC-B", "Latent2RGB-LTXV", "TAEF1", "TAESDXL", "TAESD15", "TAESD3",
           "Latent2RGB-Mochi", "Latent2RGB-Wan2.1", "Latent2RGB-Wan2.2", "None"]
SHAPES = [(1, 3, 8, 8), (1, 4, 8, 8), (2, 4, 8, 8), (1, 8, 8, 8), (1, 16, 8, 8), (1, 32, 8, 8), (1, 64, 8, 8), (1, 128, 8, 8),
          (1, 4, 3, 8, 8), (1, 12, 3, 8, 8), (1, 16, 3, 8, 8), (1, 24, 3, 8, 8), (1, 32, 3, 8, 8), (1, 48, 3, 8, 8), (1, 128, 3, 8, 8),
          (1, 64, 32), (1, 8, 16, 3, 8, 8)]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("method", METHODS)
def test_a_latent_reaches_the_receiver_whatever_the_preview_method(sender, shape, method):
    latent = torch.randn(shape)

    assert torch.equal(sender.send(latent, method), latent)


def test_a_latent_that_the_preview_method_fits_reaches_the_receiver(sender):
    latent = torch.randn(1, 4, 8, 8)

    assert torch.equal(sender.send(latent, "Latent2RGB-SD15"), latent)


def test_the_new_methods_stand_behind_the_existing_ones(sender):
    assert sender.methods == METHODS
