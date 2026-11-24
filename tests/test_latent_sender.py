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
        from impact import segs_nodes  # noqa: F401
        from impact import impact_pack

        def send(latent, preview_method):
            sent = impact_pack.LatentSender().doit({"samples": latent}, link_id=7, preview_method=preview_method)["ui"]["images"][0]
            assert server.send_sync.call_args.args == ("latent-send", {"link_id": 7, "images": [sent]})
            return impact_pack.LatentReceiver.load_preview_latent(str(tmp_path / sent["subfolder"] / sent["filename"]))["samples"]

        yield SimpleNamespace(send=send, methods=impact_pack.LatentSender.INPUT_TYPES()["required"]["preview_method"][0])


def test_a_latent_that_no_preview_method_fits_is_sent_with_none(sender):
    latent_of_a_video_model = torch.randn(1, 48, 3, 8, 8)
    assert torch.equal(sender.send(latent_of_a_video_model, "None"), latent_of_a_video_model)
    for method in sender.methods[:-1]:
        with pytest.raises(RuntimeError):
            sender.send(latent_of_a_video_model, method)


def test_a_latent_with_a_preview_is_sent_as_before(sender):
    latent = torch.randn(1, 4, 8, 8)
    assert torch.equal(sender.send(latent, "Latent2RGB-SD15"), latent)


def test_none_is_added_behind_the_existing_methods(sender):
    assert sender.methods == ["Latent2RGB-FLUX.1", "Latent2RGB-SDXL", "Latent2RGB-SD15", "Latent2RGB-SD3", "Latent2RGB-SD-X4", "Latent2RGB-Playground-2.5",
                              "Latent2RGB-SC-Prior", "Latent2RGB-SC-B", "Latent2RGB-LTXV", "TAEF1", "TAESDXL", "TAESD15", "TAESD3", "None"]
