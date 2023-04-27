import shutil
import folder_paths
import os, sys, subprocess


# ensure .js
print("### Loading: ComfyUI-Impact-Pack")

comfy_path = os.path.dirname(folder_paths.__file__)

def setup_js():
    impact_path = os.path.dirname(__file__)
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "core")
    js_src_path = os.path.join(impact_path, "js", "impact-pack.js")
    shutil.copy(js_src_path, js_dest_path)
    
setup_js()

from .impact_pack import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']