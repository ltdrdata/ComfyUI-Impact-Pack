import os
from types import SimpleNamespace

import pytest
import torch


class EachPositionByItself(torch.nn.Module):
    dtype = torch.float32

    def forward(self, x, timestep, **kwargs):
        return torch.tanh(x)


@pytest.fixture
def samplers():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        import comfy.latent_formats
        import comfy.model_base
        import comfy.model_patcher
        import comfy.sample
        import comfy.supported_models_base
        from thirdparty import tiled_sampler

        cpu = torch.device("cpu")
        kinds = {"sd15": (comfy.model_base.ModelType.EPS, comfy.latent_formats.SD15),
                 "flux": (comfy.model_base.ModelType.FLOW, comfy.latent_formats.Flux),
                 "wan": (comfy.model_base.ModelType.FLOW, comfy.latent_formats.Wan21)}

        def model(kind):
            model_type, latent_format = kinds[kind]
            config = comfy.supported_models_base.BASE({"disable_unet_model_creation": True})
            config.latent_format = latent_format()
            model = comfy.model_base.BaseModel(config, model_type=model_type)
            model.diffusion_model = EachPositionByItself()
            return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)

        def prompt():
            return [[torch.zeros(1, 1, 8), {}]]

        def tiled(kind, latent, strategy, denoise, steps=4):
            return tiled_sampler.sample(model(kind), 7, 64, 64, strategy, steps, 2.0, "euler", "simple", prompt(), prompt(), latent, denoise)[0]["samples"]

        def plain(kind, latent, denoise, steps=4):
            total = int(steps / denoise)
            noise = comfy.sample.prepare_noise(latent["samples"], 7)
            return comfy.sample.sample(model(kind), noise, total, 2.0, "euler", "simple", prompt(), prompt(), latent["samples"],
                                       start_step=total - steps, last_step=total, noise_mask=latent.get("noise_mask"), seed=7)

        yield SimpleNamespace(tiled=tiled, plain=plain)


def latent_of(kind, batch=1):
    generator = torch.Generator().manual_seed(3)
    shape = {"sd15": (batch, 4, 20, 28), "flux": (batch, 16, 20, 28), "wan": (batch, 16, 1, 20, 28)}[kind]
    return {"samples": torch.randn(shape, generator=generator)}


@pytest.mark.parametrize("strategy", ["random", "padded", "simple"])
@pytest.mark.parametrize("kind", ["sd15", "flux", "wan"])
def test_tiles_of_a_model_that_takes_each_position_by_itself_give_what_the_plain_sampler_gives(samplers, kind, strategy):
    latent = latent_of(kind)
    result = samplers.tiled(kind, latent, strategy, 0.5)
    assert torch.allclose(result, samplers.plain(kind, latent, 0.5), atol=1e-4)


@pytest.mark.parametrize("strategy", ["random", "padded", "simple"])
@pytest.mark.parametrize("kind", ["flux", "wan"])
def test_a_flow_model_is_sampled_from_noise_alone_at_full_denoise(samplers, kind, strategy):
    latent = latent_of(kind)
    result = samplers.tiled(kind, latent, strategy, 1.0)
    assert torch.allclose(result, samplers.plain(kind, latent, 1.0), atol=1e-4)


@pytest.mark.parametrize("strategy", ["random", "padded", "simple"])
@pytest.mark.parametrize("kind", ["sd15", "flux", "wan"])
def test_a_batch_of_latents_is_tiled_latent_by_latent(samplers, kind, strategy):
    latent = latent_of(kind, batch=2)
    result = samplers.tiled(kind, latent, strategy, 0.5)
    assert torch.allclose(result, samplers.plain(kind, latent, 0.5), atol=1e-4)


@pytest.mark.parametrize("strategy", ["random", "simple"])
@pytest.mark.parametrize("kind", ["sd15", "flux", "wan"])
def test_only_the_masked_part_of_the_latent_is_sampled(samplers, kind, strategy):
    latent = latent_of(kind)
    latent["noise_mask"] = torch.zeros(1, 1, 20, 28)
    latent["noise_mask"][..., 4:14, 6:20] = 1.0
    result = samplers.tiled(kind, latent, strategy, 0.5)
    assert torch.allclose(result, samplers.plain(kind, latent, 0.5), atol=1e-4)
