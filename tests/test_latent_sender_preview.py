import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def sender():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        from server import PromptServer
        server = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None, send_sync=Mock(), client_id="page")
        patch.setattr(PromptServer, "instance", server, raising=False)
        import nodes
        from impact import impact_server  # noqa: F401
        from impact import impact_pack

        loaded = []

        def load_vae(self, vae_name):
            loaded.append(vae_name)
            return (SimpleNamespace(decode=lambda samples: torch.stack([torch.stack([one.mean(), torch.tensor(1.0), torch.tensor(0.0)]).expand(samples.shape[-2] * 8, samples.shape[-1] * 8, 3)
                                                                        for one in samples])),)

        patch.setattr(nodes.VAELoader, "load_vae", load_vae)
        files = [f"{name}_{part}.pth" for name in ("taesd", "taesdxl", "taesd3", "taef1") for part in ("encoder", "decoder")]
        patch.setattr(impact_pack.folder_paths, "get_filename_list", lambda folder: files)

        def picture(latent, method):
            preview = impact_pack.LatentSender.prepare_preview(latent, method)
            return preview.crop((0, 0, preview.width, preview.height - 50))

        yield SimpleNamespace(picture=picture, loaded=loaded, files=files)


@pytest.mark.parametrize("method, decoder, channels", [("TAESD15", "taesd", 4), ("TAESDXL", "taesdxl", 4), ("TAESD3", "taesd3", 16), ("TAEF1", "taef1", 16)])
def test_a_tae_method_shows_what_its_decoder_makes_of_the_latent(sender, method, decoder, channels):
    picture = sender.picture(torch.zeros(1, channels, 16, 16), method)

    assert sender.loaded == [decoder]
    assert picture.getcolors() == [(picture.width * picture.height, (0, 255, 0))]


def test_a_tae_method_shows_the_first_latent_of_a_batch(sender):
    picture = sender.picture(torch.cat([torch.zeros(1, 4, 16, 16), torch.ones(1, 4, 16, 16)]), "TAESD15")

    assert picture.getcolors() == [(picture.width * picture.height, (0, 255, 0))]


@pytest.mark.parametrize("method, format_name", [("Latent2RGB-SD15", "SD15"), ("Latent2RGB-SDXL", "SDXL"), ("Latent2RGB-SD3", "SD3"), ("Latent2RGB-FLUX.1", "Flux")])
def test_a_latent2rgb_method_shows_a_stored_latent_as_the_sampler_shows_it_while_it_samples(sender, method, format_name):
    import comfy.latent_formats
    from latent_preview import Latent2RGBPreviewer

    latent_format = getattr(comfy.latent_formats, format_name)()
    while_sampling = torch.full((1, latent_format.latent_channels, 16, 16), 0.5)
    shown_while_sampling = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias).decode_latent_to_preview(while_sampling)

    picture = sender.picture(latent_format.process_out(while_sampling), method)

    assert picture.getcolors() == [(picture.width * picture.height, shown_while_sampling.getpixel((0, 0)))]


@pytest.mark.parametrize("method, decoder, latent2rgb, channels", [("TAESD15", "taesd", "Latent2RGB-SD15", 4), ("TAESDXL", "taesdxl", "Latent2RGB-SDXL", 4),
                                                                   ("TAESD3", "taesd3", "Latent2RGB-SD3", 16), ("TAEF1", "taef1", "Latent2RGB-FLUX.1", 16)])
@pytest.mark.parametrize("left, said", [([], "needs {0}_encoder and {0}_decoder in"), (["{0}_decoder.pth"], "needs {0}_encoder in"), (["{0}_encoder.safetensors"], "needs {0}_decoder in")])
def test_a_tae_method_without_its_files_shows_latent2rgb_and_names_the_missing_files(sender, caplog, method, decoder, latent2rgb, channels, left, said):
    sender.files[:] = [file.format(decoder) for file in left]
    latent = torch.linspace(-1, 1, channels * 256).reshape(1, channels, 16, 16)

    picture = sender.picture(latent, method)

    assert sender.loaded == []
    assert picture.tobytes() == sender.picture(latent, latent2rgb).tobytes()
    assert f"showed {latent2rgb}: {method} {said.format(decoder)} models/vae_approx" in caplog.text
