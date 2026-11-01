"""Original PNGs from ComfyUI-extension-tutorials ca31a3b; never rewrite them."""
import hashlib
import json
from pathlib import Path

from PIL import Image


CASES = {
    "lbw": ("ComfyUI-Impact-Pack/workflow/wildcard-lbw.png", 10,
            "50577c44e323720cb4955564ee1c3aba0970093494b10f41df5efdab1205dfe8"),
    "styler": ("ComfyUI-Impact-Pack/workflow/wildcard-prompt-styler.png", 21,
               "be2c571bf7c132fef22b79e33ee2a8d85ad915d5f95402372a1b1a50d8bf07d4"),
    "sdxl6": ("Beginner/kr/sdxl6.png", 109,
              "19152a77bc1b34c74fb9905fa769502e5b8120d050bfbbe8a0f8334a962da75f"),
    "a1111": ("ComfyUI-Inspire-Pack/workflow/reproduce-a1111.png", 14,
              "0a458eac122f57ea40c7df15e1f3e39eda463a3c579c5305c654f52971c8d0fd"),
}


def tutorial(root, case):
    relative, node_id, sha256 = CASES[case]
    path = Path(root) / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, path
    with Image.open(path) as image:
        workflow = json.loads(image.info["workflow"])
    node = next(node for node in workflow["nodes"] if node["id"] == node_id)
    return path, workflow, node
