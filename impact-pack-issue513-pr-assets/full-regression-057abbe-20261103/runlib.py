from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / 'metadata.json').read_text())
REPO = Path(META['repo'])
PACK = ROOT / 'runtime/custom_nodes/comfyui-impact-pack'
COMFY = Path('/mnt/teratera/git/ComfyUI')
PY = '/home/rho/venv/bin/python'
ENV = dict(os.environ, IMPACT_COMFY_ROOT=str(COMFY), COMFYUI_ROOT=str(COMFY),
           IMPACT_TUTORIALS_DIR='/mnt/teratera/git/ComfyUI-extension-tutorials',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

def run(label, command, *, cwd=REPO, env=None, timeout=1800):
    artifacts = ROOT / label
    artifacts.mkdir(parents=True, exist_ok=True)
    variables = dict(ENV, IMPACT_E2E_ARTIFACTS=str(artifacts), **(env or {}))
    start = time.monotonic()
    print('START ' + label, flush=True)
    record = {'label': label, 'command': [str(x) for x in command], 'cwd': str(cwd),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'environment_overrides': env or {},
              'head': META['head']}
    with (ROOT / (label + '.log')).open('w') as log:
        try:
            result = subprocess.run(command, cwd=cwd, env=variables, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            record['exit_code'] = result.returncode
        except subprocess.TimeoutExpired:
            record['exit_code'] = 124
            record['timeout_seconds'] = timeout
    record['duration_seconds'] = round(time.monotonic() - start, 2)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print('END', label, record['exit_code'], flush=True)
    return record

def pytest(label, *args, **kwargs):
    return run(label, [PY, '-m', 'pytest', *args, '-v', '--tb=short',
                       '--junitxml=' + str(ROOT / (label + '.xml'))], **kwargs)

from contextlib import contextmanager
from urllib.request import urlopen
import socket

@contextmanager
def server(profile, port, *options):
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0, 'Occupied test port'
    with (ROOT / ('server-' + profile + '.log')).open('w') as log:
        proc = subprocess.Popen([PY, ROOT / 'serve.py', '--profile', profile, '--port', str(port), *options],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert proc.poll() is None, 'Server startup failed: ' + profile
                try:
                    with urlopen(f'http://127.0.0.1:{port}/object_info', timeout=1) as response:
                        if 'ImpactWildcardProcessor' in json.load(response):
                            break
                except OSError:
                    time.sleep(.5)
            else:
                raise RuntimeError('Server did not become ready: ' + profile)
            yield f'http://127.0.0.1:{port}'
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
