"""Chrome UI E2E for the DifferentialDiffusion cross-version compat shim.

Exercises the `utils.apply_differential_diffusion` helper end-to-end through
a real SEGSDetailer inference with `noise_mask_feather > 0`. This is the
smallest graph that deterministically triggers the helper without relying on
FaceDetailer's KSampler-generated face pipeline.

Prerequisites:
- ComfyUI running on http://127.0.0.1:18188 with impact-pack AND
  impact-subpack whitelisted:
    python main.py --disable-all-custom-nodes \
        --whitelist-custom-nodes comfyui-impact-pack comfyui-impact-subpack \
        --port 18188
- Models available:
    models/checkpoints/SD1.5/realcartoonPixar_v8.safetensors
    models/ultralytics/bbox/face_yolov8m.pt
- Input image with a visible face at input/ComfyUI_00156_.png
- Python Playwright and Google Chrome

What it verifies:
1. SEGSDetailer inference runs without AttributeError on DifferentialDiffusion.
2. Output image differs slightly from input (detailer actually edited the face).
3. UI workflow import, Run, and image preview updates succeed.

Use --base-url, --browser-channel, --headed, and --artifacts to configure the run.
"""
import io
import json
import time

import numpy as np
from PIL import Image

from e2e_ui import browser_session, options

_SEED = int(time.time()) & 0xFFFFFFFF

PROMPT = {
    "ckpt": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "SD1.5/realcartoonPixar_v8.safetensors"},
    },
    "pos": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["ckpt", 1], "text": "a detailed face, high quality, sharp focus"},
    },
    "neg": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["ckpt", 1], "text": "blurry, low quality"},
    },
    "pipe": {
        "class_type": "ToBasicPipe",
        "inputs": {
            "model": ["ckpt", 0],
            "clip": ["ckpt", 1],
            "vae": ["ckpt", 2],
            "positive": ["pos", 0],
            "negative": ["neg", 0],
        },
    },
    "img": {
        "class_type": "LoadImage",
        "inputs": {"image": "ComfyUI_00156_.png"},
    },
    "detector": {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": "bbox/face_yolov8m.pt"},
    },
    "bbox_segs": {
        "class_type": "BboxDetectorSEGS",
        "inputs": {
            "bbox_detector": ["detector", 0],
            "image": ["img", 0],
            "threshold": 0.30,
            "dilation": 10,
            "crop_factor": 3.0,
            "drop_size": 10,
            "labels": "all",
        },
    },
    # Non-zero noise_mask_feather is the critical knob — this is what
    # activates the DifferentialDiffusion path inside enhance_detail and
    # SEGSDetailer.do_detail.
    "detail": {
        "class_type": "SEGSDetailer",
        "inputs": {
            "image": ["img", 0],
            "segs": ["bbox_segs", 0],
            "guide_size": 512,
            "guide_size_for": True,
            "max_size": 1024,
            "seed": _SEED,
            "steps": 10,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.5,
            "noise_mask": True,
            "force_inpaint": True,
            "basic_pipe": ["pipe", 0],
            "refiner_ratio": 0.2,
            "batch_size": 1,
            "cycle": 1,
            "noise_mask_feather": 20,
        },
    },
    "paste": {
        "class_type": "SEGSPaste",
        "inputs": {
            "image": ["img", 0],
            "segs": ["detail", 0],
            "feather": 5,
            "alpha": 255,
        },
    },
    "preview_paste": {
        "class_type": "PreviewImage",
        "inputs": {"images": ["paste", 0]},
    },
    "preview_input": {
        "class_type": "PreviewImage",
        "inputs": {"images": ["img", 0]},
    },
}


def main():
    args = options(__doc__)
    with browser_session(args) as ui:
        # Import the prepared graph through the frontend's file input. The
        # frontend performs node creation, widget callbacks, and connections.
        ui.load_workflow({
            "name": "dd-compat.json",
            "mimeType": "application/json",
            "buffer": json.dumps(PROMPT).encode("utf-8"),
        }, len(PROMPT))
        entry = ui.run()
        outputs = entry["outputs"]
        assert "preview_input" in outputs and "preview_paste" in outputs, outputs

        # Require the frontend to receive and load the output image as well.
        ui.page.wait_for_function("""(app) => {
            const node = app.graph.getNodeById('preview_paste');
            return node?.imgs?.some(img => img.complete && img.naturalWidth > 0);
        }""", arg=ui.app)

        def image_stats(meta):
            response = ui.page.request.get(f"{ui.base_url}/view", params={
                "filename": meta["filename"],
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            })
            assert response.ok, f"Preview image request failed: {response.status}"
            with Image.open(io.BytesIO(response.body())) as image:
                pixels = np.array(image)
                return {"size": image.size, "mean": float(pixels.mean()), "std": float(pixels.std())}

        original = image_stats(outputs["preview_input"]["images"][0])
        detailed = image_stats(outputs["preview_paste"]["images"][0])
        assert detailed["std"] >= 1.0, f"Degenerate output: {detailed}"
        assert original["size"] == detailed["size"], (original, detailed)
        mean_delta = abs(detailed["mean"] - original["mean"])
        std_delta = abs(detailed["std"] - original["std"])
        assert mean_delta >= 0.005 or std_delta >= 0.005, "Detailer output is unchanged"
        print(f"PASS: UI import/Run/preview, detailer mean_delta={mean_delta:.4f}, std_delta={std_delta:.4f}")


if __name__ == "__main__":
    main()
