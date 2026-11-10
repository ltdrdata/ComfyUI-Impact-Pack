import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time
from urllib.request import urlopen

ROOT = Path('/mnt/teratera/git/impact-pack-pr1172-followup')
COMFY = Path('/mnt/teratera/git/ComfyUI')
ASSETS = Path(__file__).resolve().parent
PYTHON = '/home/rho/venv/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--ref')
parser.add_argument('--source', type=Path)
parser.add_argument('--port', type=int, default=18172)
parser.add_argument('tests', nargs='+')
args = parser.parse_args()
run = ASSETS / args.label
run.mkdir(exist_ok=True)
runtime = run / 'runtime'
pack = runtime / 'custom_nodes/comfyui-impact-pack'
pack.mkdir(parents=True, exist_ok=True)
if args.ref:
    archive = subprocess.check_output(['git', 'archive', args.ref], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        contents.extractall(pack, filter='data')
elif args.source:
    shutil.copytree(args.source, pack, dirs_exist_ok=True)
else:
    files = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    for name in set(files) - {''}:
        if (ROOT / name).is_file():
            (pack / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, pack / name)
subpack = runtime / 'custom_nodes/comfyui-impact-subpack'
if not subpack.exists():
    subpack.symlink_to(COMFY / 'custom_nodes/comfyui-impact-subpack', target_is_directory=True)
(pack / 'impact-pack.ini').write_text('[default]\ncustom_wildcards = ' + str(pack / 'tests/wildcards/samples') + '\nwildcard_cache_limit_mb = 50\nsam_editor_cpu = True\nsam_editor_model = sam_vit_b_01ec64.pth\n')
command = [PYTHON, 'main.py', '--listen', '127.0.0.1', '--port', str(args.port), '--disable-auto-launch',
           '--base-directory', str(runtime), '--models-directory', str(COMFY / 'models'),
           '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'comfyui-impact-pack', 'comfyui-impact-subpack',
           '--cpu', '--cache-classic']
env = os.environ | {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1',
                    'IMPACT_E2E_BASE_URL': f'http://127.0.0.1:{args.port}', 'IMPACT_E2E_ARTIFACTS': str(run / 'browser'),
                    'IMPACT_E2E_SAM_IMAGE': '/mnt/teratera/git/ComfyUI-extension-tutorials/ComfyUI-Impact-Pack/images/interactive-sam-original.png'}
restart_request = run / 'restart.request'
restart_command = ASSETS / 'request_restart.py'
import shlex
env['IMPACT_E2E_RESTART_COMMAND'] = shlex.join([PYTHON, str(restart_command), str(restart_request)])
with socket.socket() as probe:
    assert probe.connect_ex(('127.0.0.1', args.port)) != 0
with (run / 'server.log').open('w') as log:
    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            assert process.poll() is None, 'Server exited; see server.log'
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/object_info', timeout=1) as response:
                    if 'PreviewBridge' in json.load(response):
                        break
            except OSError:
                time.sleep(.5)
        else:
            raise RuntimeError('Server readiness timeout')
        provenance = {'command': command, 'ref': args.ref, 'source': str(args.source) if args.source else None,
                      'runtime_sha256': {str(path.relative_to(pack)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for folder in ['js', 'modules'] for path in (pack / folder).rglob('*') if path.is_file()},
                      'tests': args.tests,
                      'test_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (ROOT / 'tests').glob('*.py')},
                      'fixture_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (ROOT / 'tests/workflows').glob('all-impact-nodes*')}}
        (run / 'source.json').write_text(json.dumps(provenance, indent=2))
        with (run / 'browser.log').open('w') as output:
            result = subprocess.Popen([PYTHON, '-m', 'pytest', *args.tests, '-v', '--tb=short', '--junitxml=' + str(run / 'browser.xml')],
                                      cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 2700
            while result.poll() is None:
                if time.monotonic() > deadline:
                    result.kill()
                    raise TimeoutError('Browser checks timed out')
                if restart_request.exists():
                    process.terminate()
                    process.wait(timeout=20)
                    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
                    for _ in range(120):
                        assert process.poll() is None, 'Restarted server exited'
                        try:
                            with urlopen(f'http://127.0.0.1:{args.port}/object_info', timeout=1) as response:
                                if 'PreviewBridge' in json.load(response):
                                    break
                        except OSError:
                            time.sleep(.5)
                    else:
                        raise RuntimeError('Restart readiness timeout')
                    restart_request.unlink()
                time.sleep(.1)
        print('Browser checks exit:', result.returncode, flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
raise SystemExit(result.returncode)
