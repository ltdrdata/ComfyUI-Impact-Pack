"""E2E test for the DifferentialDiffusion cross-version compat shim.

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
- Python Playwright 1.58+ (`pip install playwright && playwright install chromium`)

What it verifies:
1. SEGSDetailer inference runs without AttributeError on DifferentialDiffusion.
2. Output image differs slightly from input (detailer actually edited the face).
3. Execution returns status=success.
"""
from playwright.sync_api import sync_playwright
import json
import time
import sys
import urllib.parse

BASE_URL = "http://127.0.0.1:18188"
TIMEOUT_S = 600

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


def fail(msg: str, code: int = 1) -> None:
    print(f"FAIL: {msg}")
    sys.exit(code)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto(f"{BASE_URL}/", wait_until="domcontentloaded", timeout=30000)

        submit = page.evaluate(
            """async (prompt) => {
                const client_id = (window.api && window.api.clientId) || crypto.randomUUID();
                const resp = await fetch('/prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, client_id })
                });
                const text = await resp.text();
                let parsed = null; try { parsed = JSON.parse(text); } catch {}
                return { status: resp.status, body: parsed || text };
            }""",
            PROMPT,
        )
        if submit.get("status") != 200 or not isinstance(submit.get("body"), dict):
            fail(f"submission failed: {submit}", 2)
        prompt_id = submit["body"].get("prompt_id")
        if not prompt_id:
            fail("no prompt_id returned", 2)
        print(f"prompt_id: {prompt_id}")

        deadline = time.time() + TIMEOUT_S
        last_sig = None
        history_entry = None
        while time.time() < deadline:
            state = page.evaluate(
                f"""async () => {{
                    const h = await fetch('/history/{prompt_id}').then(r => r.json());
                    const q = await fetch('/queue').then(r => r.json());
                    return {{ h, q }};
                }}"""
            )
            q = state["q"]
            sig = f"running={len(q.get('queue_running', []))} pending={len(q.get('queue_pending', []))}"
            if sig != last_sig:
                print(f"[{int(TIMEOUT_S - (deadline - time.time())):>3}s] {sig}")
                last_sig = sig
            if prompt_id in state["h"]:
                history_entry = state["h"][prompt_id]
                break
            time.sleep(3)
        if history_entry is None:
            fail(f"prompt did not complete within {TIMEOUT_S}s", 2)

        status = history_entry.get("status", {})
        messages = status.get("messages", [])
        status_str = status.get("status_str")
        print(f"status_str: {status_str}")

        exec_errors = [m[1] for m in messages if m[0] == "execution_error"]
        for err in exec_errors:
            print(f"[ERROR] node={err.get('node_id')} {err.get('exception_type')}: {err.get('exception_message')}")
            for t in err.get("traceback", [])[-5:]:
                print(f"    {t.strip()}")
        if exec_errors:
            fail("execution_error present (DD compat shim or unrelated)", 1)
        if status_str != "success":
            fail(f"status_str={status_str!r}")

        outputs = history_entry.get("outputs", {})
        if "preview_input" not in outputs or "preview_paste" not in outputs:
            fail(f"expected previews missing from outputs: {list(outputs)}")

        def fetch_png_stats(meta):
            qs = urllib.parse.urlencode(
                {
                    "filename": meta.get("filename", ""),
                    "subfolder": meta.get("subfolder", ""),
                    "type": meta.get("type", "output"),
                }
            )
            raw_list = page.evaluate(
                f"""async () => {{
                    const r = await fetch('/view?{qs}');
                    if (!r.ok) return null;
                    const ab = await r.arrayBuffer();
                    return Array.from(new Uint8Array(ab));
                }}"""
            )
            if not raw_list:
                return None
            import io as _io
            import numpy as np
            from PIL import Image as PILImage

            pim = PILImage.open(_io.BytesIO(bytes(raw_list)))
            arr = np.array(pim)
            return {
                "size": pim.size,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

        in_stats = fetch_png_stats(outputs["preview_input"]["images"][0])
        out_stats = fetch_png_stats(outputs["preview_paste"]["images"][0])
        if in_stats is None or out_stats is None:
            fail("could not fetch one or both preview images")

        print(f"input : {in_stats}")
        print(f"paste : {out_stats}")

        if out_stats["std"] < 1.0:
            fail("paste output is degenerate (flat image)")

        if in_stats["size"] != out_stats["size"]:
            fail(f"size mismatch: {in_stats['size']} vs {out_stats['size']}")

        # SEGSPaste with a detailed face should produce a slightly different mean/std
        # vs the untouched input. Exact equality would indicate the detailer path
        # (and thus the DD compat shim) was bypassed.
        mean_delta = abs(out_stats["mean"] - in_stats["mean"])
        std_delta = abs(out_stats["std"] - in_stats["std"])
        if mean_delta < 0.005 and std_delta < 0.005:
            fail(
                f"output identical to input (mean_delta={mean_delta:.4f}, "
                f"std_delta={std_delta:.4f}) — detailer likely didn't run"
            )

        print(f"PASS: detailer ran, mean_delta={mean_delta:.4f}, std_delta={std_delta:.4f}")
        browser.close()


if __name__ == "__main__":
    main()
