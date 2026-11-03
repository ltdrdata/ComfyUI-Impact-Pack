"""Run a frozen package with separate user/input/output directories."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parent
COMFY = Path('/mnt/teratera/git/ComfyUI')
parser = argparse.ArgumentParser()
parser.add_argument('--profile', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--cache', choices=['classic', 'none', 'lru'], default='classic')
parser.add_argument('--samples', default='samples')
parser.add_argument('--wildcard-limit', type=float, default=50)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--fp32', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--minimal', action='store_true')
args = parser.parse_args()
runtime = ROOT / ('runtime-' + args.profile)
custom = runtime / 'custom_nodes'
custom.mkdir(parents=True, exist_ok=True)
pack = custom / 'comfyui-impact-pack'
source = ROOT / ('baseline-source' if args.baseline else 'source')
if not pack.exists():
    shutil.copytree(source, pack)
(pack / 'impact-pack.ini').write_text('[default]\ncustom_wildcards = ' + str(pack / 'tests/wildcards' / args.samples)
    + '\nwildcard_cache_limit_mb = ' + str(args.wildcard_limit)
    + '\nsam_editor_model = sam_vit_b_01ec64.pth\nsam_editor_cpu = True\n')
plugins = ['comfyui-impact-subpack'] if args.minimal else [
    'comfyui-inspire-pack', 'comfyui-impact-subpack', 'comfyui_essentials',
    'comfyui_controlnet_aux', 'comfyui-animatediff-evolved', 'comfyui-videohelpersuite', 'comfyui_ipadapter_plus']
for name in plugins:
    if not (custom / name).exists():
        (custom / name).symlink_to(COMFY / 'custom_nodes' / name, target_is_directory=True)
if not (runtime / 'user').exists():
    shutil.copytree(ROOT.parent / 'fixed-mode/runtime/user', runtime / 'user')
(runtime / 'input').mkdir(exist_ok=True)
for name in ['ComfyUI_00156_.png', 'ComfyUI_temp_xltgv_00001_.png', 'ComfyUI_00289_.png',
             'ComfyUI_temp_fyzuo_00038_.png', 'example.png', 'chunli.png']:
    if not (runtime / 'input' / name).exists():
        shutil.copy2(COMFY / 'input' / name, runtime / 'input' / name)
with socket.socket() as probe:
    assert probe.connect_ex(('127.0.0.1', args.port)) != 0, 'Occupied test port'
command = [sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', str(args.port),
           '--disable-auto-launch', '--base-directory', str(runtime), '--models-directory', str(COMFY / 'models'),
           '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'comfyui-impact-pack', *plugins]
command += ['--cuda-device', '0', '--deterministic', '--disable-xformers', '--use-pytorch-cross-attention'] if args.gpu else ['--cpu']
if args.fp32:
    command += ['--force-fp32', '--fp32-vae', '--disable-dynamic-vram']
command += ['--cache-' + args.cache]
if args.cache == 'lru':
    command += ['10']
metadata = json.loads((ROOT / 'metadata.json').read_text())
(ROOT / ('server-' + args.profile + '.json')).write_text(json.dumps({
    'pid': os.getpid(), 'command': command, 'runtime': str(runtime), 'source': str(source),
    'revision': metadata['base' if args.baseline else 'head'],
    'source_js_sha256': hashlib.sha256((pack / 'js/impact-pack.js').read_bytes()).hexdigest(),
}, indent=2) + '\n')
env = os.environ | {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0', 'HF_HUB_OFFLINE': '1'}
os.chdir(COMFY)
os.execvpe(command[0], command, env)
