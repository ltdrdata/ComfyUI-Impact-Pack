"""Run legacy diagnostics safely; compare every failure with current PR base."""
from runlib import *
import hashlib
import re
import shlex
import shutil
import socket

area = ROOT / 'legacy-isolated'
area.mkdir(exist_ok=True)
pack = area / 'runtime/custom_nodes/comfyui-impact-pack'
shutil.copytree(REPO, pack, dirs_exist_ok=True)
corpus = COMFY / 'custom_nodes/comfyui-impact-pack/custom_wildcards'
shutil.copytree(corpus, pack / 'custom_wildcards', dirs_exist_ok=True)
(ROOT / 'legacy-corpus.json').write_text(json.dumps({
    'source': str(corpus), 'sha256': {str(path.relative_to(corpus)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in corpus.rglob('*') if path.is_file()},
}, indent=2) + '\n')
stop = area / 'stop.py'
stop.write_text('''from pathlib import Path
import os,signal,time
area=Path(__file__).resolve().parent
file=area/'server.pid'
if file.exists():
 pid=int(file.read_text());proc=Path('/proc')/str(pid)/'cmdline'
 if proc.exists() and proc.read_bytes():
  assert str(area/'runtime').encode() in proc.read_bytes()
  os.kill(pid,signal.SIGTERM)
  for _ in range(100):
   if not proc.exists() or not proc.read_bytes():break
   time.sleep(.1)
 file.unlink(missing_ok=True)
''')
(area / 'launch.py').write_text('''from pathlib import Path
import os,sys
area=Path(__file__).resolve().parent
(area/'server.pid').write_text(str(os.getpid()))
command=['/home/rho/venv/bin/python','/mnt/teratera/git/ComfyUI/main.py','--cpu','--disable-auto-launch','--base-directory',str(area/'runtime'),'--disable-all-custom-nodes','--whitelist-custom-nodes','comfyui-impact-pack',*sys.argv[1:]]
os.chdir('/mnt/teratera/git/ComfyUI');os.execv(command[0],command)
''')
(area / 'run.sh').write_text('exec ' + shlex.quote(PY) + ' ' + shlex.quote(str(area / 'launch.py')) + ' "$@"\n')
(area / 'temp').mkdir(exist_ok=True)
with socket.socket() as probe:
    assert probe.connect_ex(('127.0.0.1', 18505)) != 0
adaptations = {}
shells = ['tests/test_error_handling.sh', 'tests/test_edge_cases.sh', 'tests/test_deep_nesting.sh',
          'tests/test_ondemand_loading.sh', 'tests/test_config_quotes.sh',
          'tests/wildcards/test_lazy_load_api.sh', 'tests/wildcards/test_progressive_ondemand.sh',
          'tests/wildcards/test_sequential_loading.sh', 'tests/wildcards/test_versatile_prompts.sh',
          'tests/wildcards/test_wildcard_consistency.sh', 'tests/wildcards/verify_ondemand_mode.sh']
for relative in shells:
    original = (REPO / relative).read_text()
    adapted = re.sub(r'^(IMPACT_PACK_DIR|IMPACT_DIR)=.*$',
        lambda match: match[1] + '=' + shlex.quote(str(pack)), original, flags=re.M)
    adapted = re.sub(r'^COMFYUI_DIR=.*$', 'COMFYUI_DIR=' + shlex.quote(str(area)), adapted, flags=re.M)
    adapted = re.sub(r'pkill(?: -9)? -f "[^"]+"',
        shlex.quote(PY) + ' ' + shlex.quote(str(stop)), adapted)
    adapted = re.sub(r'\b(?:8188|819[0-9])\b', '18505', adapted)
    adapted = adapted.replace('/tmp/', str(area / 'temp') + '/')
    assert 'pkill ' not in adapted and 'killall ' not in adapted
    assert re.search(r'^COMFYUI_DIR=' + re.escape(shlex.quote(str(area))) + '$', adapted, re.M)
    target = pack / relative
    target.write_text(adapted)
    adaptations[relative] = {'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'adapted_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'scope': 'Disposable paths, isolated port/temp/config, and owned PID shutdown only; assertions and waits unchanged.'}
relative = 'tests/wildcards/test_wildcard_final.py'
original = (REPO / relative).read_text()
adapted = re.sub(r'^IMPACT_PACK_DIR = .*$', 'IMPACT_PACK_DIR = Path(' + repr(str(pack)) + ')', original, flags=re.M)
adapted = re.sub(r'^COMFYUI_DIR = .*$', 'COMFYUI_DIR = Path(' + repr(str(area)) + ')', adapted, flags=re.M)
adapted = adapted.replace('8191', '18505')
(pack / relative).write_text(adapted)
adaptations[relative] = {'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'adapted_sha256': hashlib.sha256(adapted.encode()).hexdigest(), 'scope': 'Only package/server paths and port.'}
(ROOT / 'legacy-adaptations.json').write_text(json.dumps(adaptations, indent=2) + '\n')

def config():
    (pack / 'impact-pack.ini').write_text('[default]\ncustom_wildcards = ' + str(pack / 'custom_wildcards')
                                      + '\nwildcard_cache_limit_mb = 50\n')

def clean():
    subprocess.run([PY, stop], check=True)

def markers(label):
    text = (ROOT / (label + '.log')).read_text(errors='replace')
    text = re.sub(r'(?:\x1b|\\033)\[[0-9;]*m', '', text)
    return [line for line in text.splitlines() if re.search(r'❌|✗|\[FAIL\]|\bFAILED\b|FAIL:|Failed:\s*[1-9]', line)]

env = {'PATH': '/home/rho/venv/bin:' + os.environ.get('PATH', '')}
failures = []
records = {}
for relative in ['tests/wildcards/test_progressive_loading.py',
                 'tests/wildcards/test_wildcard_lazy_loading.py',
                 'tests/wildcards/test_lazy_load_verification.py']:
    config()
    label = 'legacy-python-' + Path(relative).stem.removeprefix('test_')
    record = pytest(label, relative, cwd=pack, env=env)
    records[label] = {'record': record, 'failure_markers': markers(label), 'test': relative}
    if record['exit_code']:
        failures.append((label, relative, 'python'))
for relative in shells + ['tests/wildcards/test_wildcard_final.py']:
    config()
    label = 'legacy-' + Path(relative).stem.removeprefix('test_')
    try:
        record = run(label, [PY if relative.endswith('.py') else 'bash', pack / relative], cwd=pack,
                     env=env, timeout=900)
    finally:
        clean()
    found = markers(label)
    records[label] = {'record': record, 'failure_markers': found, 'test': relative}
    if record['exit_code'] or found:
        failures.append((label, relative, 'cli'))

# Keep all test bytes/corpus/config/paths unchanged; replace production only.
source = json.loads((ROOT / 'source-manifest.json').read_text())
production = [name for name in source if name.startswith(('modules/', 'js/')) or name in ('__init__.py', 'install.py')]
saved = {name: (pack / name).read_bytes() for name in production}
differences = []
try:
    clean()
    for name in production:
        data = (ROOT / 'baseline-source' / name).read_bytes()
        if data != saved[name]:
            differences.append(name)
        (pack / name).write_bytes(data)
    for label, relative, kind in failures:
        config()
        baseline = 'baseline-' + label
        try:
            record = pytest(baseline, relative, cwd=pack, env=env) if kind == 'python' else run(
                baseline, [PY if relative.endswith('.py') else 'bash', pack / relative], cwd=pack,
                env=env, timeout=900)
        finally:
            clean()
        records[label]['baseline'] = {'record': record, 'failure_markers': markers(baseline)}
finally:
    clean()
    for name, data in saved.items():
        (pack / name).write_bytes(data)
(ROOT / 'legacy-diagnostics.json').write_text(json.dumps({
    'candidate': META['head'], 'base': META['base'], 'production_files_different': differences,
    'records': records, 'assertions_relaxed': False,
}, indent=2) + '\n')
