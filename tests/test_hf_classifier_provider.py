import os

import pytest
from PIL import Image


@pytest.fixture
def provide():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(os.environ["IMPACT_COMFY_ROOT"])
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from impact import hf_nodes
        yield hf_nodes.HF_TransformersClassifierProvider().doit


def test_a_vit_repo_gives_a_classifier_that_scores_a_picture(provide, tmp_path):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    config = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=16,
                       id2label={0: "female", 1: "male"}, label2id={"female": 0, "male": 1})
    ViTForImageClassification(config).save_pretrained(tmp_path)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)

    classifier = provide("Manual repo id", str(tmp_path), "CPU")[0]

    scores = classifier(Image.new("RGB", (48, 48), (120, 90, 80)))
    assert {score["label"] for score in scores} == {"female", "male"}
    assert sum(score["score"] for score in scores) == pytest.approx(1.0)
