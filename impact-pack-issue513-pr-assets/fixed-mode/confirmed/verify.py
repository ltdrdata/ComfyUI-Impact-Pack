"""Frozen frontend RGR and complete validation of fixed-mode seed controls."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/mnt/teratera/git/impact-pack-issue513-followup')
COMFY = Path('/mnt/teratera/git/ComfyUI')
OUT = Path(__file__).resolve().parent
BASE = subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=ROOT, text=True).strip()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
source = ROOT / 'js/impact-pack.js'
fixed = source.read_bytes()
original = subprocess.check_output(['git', 'show', BASE + ':js/impact-pack.js'], cwd=ROOT)
test_files = ['tests/wildcard_seed_control.test.mjs', 'tests/test_e2e_wildcard_cache.py', 'tests/test_e2e_wildcard_tutorials.py']
hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in test_files}
(OUT / 'frozen-tests.json').write_text(json.dumps({'head': head, 'test_sha256': hashes}, indent=2) + '\n')
env = os.environ | {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONHASHSEED': '0',
                   'IMPACT_E2E_BASE_URL': 'http://127.0.0.1:18190',
                   'IMPACT_TUTORIALS_DIR': '/mnt/teratera/git/ComfyUI-extension-tutorials',
                   'IMPACT_E2E_REFERENCE_IMAGE': '/mnt/teratera/git/impact-pack-pr1190-followup-pr-assets/tutorial-audit/baseline-inference/output.png'}
for key in ['IMPACT_E2E_CHECKPOINT', 'IMPACT_E2E_LORA', 'IMPACT_TUTORIAL_SD15_CHECKPOINT', 'IMPACT_TUTORIAL_SDXL_CHECKPOINT']:
    env.pop(key, None)
commands = []


def run(label, command, folder, expected=0, cwd=ROOT):
    folder.mkdir(exist_ok=True)
    print(label, flush=True)
    with (folder / f'{label}.log').open('w') as log:
        result = subprocess.run(command, cwd=cwd, env=env | {'IMPACT_E2E_ARTIFACTS': str(folder / 'browser')},
                                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    commands.append({'label': label, 'command': command, 'cwd': str(cwd), 'exit_code': result.returncode,
                     'log': str(folder / f'{label}.log')})
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    assert result.returncode == expected, (label, result.returncode)


def pytest_report(path):
    suite = ET.parse(path).getroot().find('testsuite')
    counts = {key: int(suite.attrib[key]) for key in ['tests', 'failures', 'errors', 'skipped']}
    assert counts['errors'] == counts['skipped'] == 0
    failures = [{'name': case.attrib['name'], 'message': case.find('failure').attrib['message']}
                for case in suite.findall('testcase') if case.find('failure') is not None]
    assert all(not any(error in item['message'] for error in ['TimeoutError', 'ImportError', 'ModuleNotFoundError']) for item in failures)
    return counts, failures


matrix, _ = pytest_report(OUT / 'check-matrix.xml')
assert matrix == {'tests': 20, 'failures': 0, 'errors': 0, 'skipped': 0}
reports = {}
try:
    for phase, content in [('red', original), ('green', fixed), ('red-again', original)]:
        source.write_bytes(content)
        folder = OUT / phase
        run('unit', ['node', '--experimental-vm-modules', '--test', 'tests/wildcard_seed_control.test.mjs'],
            folder, expected=0 if phase == 'green' else 1)
        unit = (folder / 'unit.log').read_text()
        unit_counts = {name: int(re.search(r'^# ' + name + r' (\d+)$', unit, re.M).group(1))
                       for name in ['tests', 'pass', 'fail', 'cancelled', 'skipped']}
        assert unit_counts == {'tests': 11, 'pass': 11 if phase == 'green' else 3,
                               'fail': 0 if phase == 'green' else 8, 'cancelled': 0, 'skipped': 0}
        if phase != 'green':
            assert unit.count("Expected values to be strictly") == 8
        selection = '(fixed_mode_suppresses and (choice-increment-after or linked-plain-increment-after)) or populate_reuses_cache_only_for_same_expansion'
        run('browser', [sys.executable, '-m', 'pytest', 'tests/test_e2e_wildcard_cache.py', '-k', selection,
                        '-v', '--tb=short', f'--junitxml={folder}/browser.xml'], folder,
            expected=0 if phase == 'green' else 1)
        browser, failures = pytest_report(folder / 'browser.xml')
        assert browser == {'tests': 8, 'failures': 0 if phase == 'green' else 6, 'errors': 0, 'skipped': 0}
        reports[phase] = {'source_sha256': hashlib.sha256(content).hexdigest(), 'unit': unit_counts,
                         'unit_failures': re.findall(r'^not ok \d+ - (.+)$', unit, re.M),
                         'browser': browser, 'browser_failures': failures}
        assert hashes == {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in test_files}
finally:
    source.write_bytes(fixed)
assert reports['red'] == reports['red-again']
(OUT / 'rgr.json').write_text(json.dumps({'successful': True, 'head': head, 'base': BASE,
    'reports': reports, 'test_sha256': hashes, 'fixed_source_restored': True,
    'fixed_source_sha256': hashlib.sha256(fixed).hexdigest()}, indent=2) + '\n')
final = OUT / 'final'
run('node', ['node', '--experimental-vm-modules', '--test', *map(str, sorted((ROOT / 'tests').glob('*.test.mjs')))], final)
run('python', [sys.executable, '-m', 'pytest', '-q', f'--junitxml={final}/python.xml'], final, cwd=ROOT / 'tests')
run('backend', [sys.executable, 'tests/integration_wildcard_cache.py', '--comfy-root', str(COMFY), '--report', str(final / 'backend.json')], final)
run('tensors', [sys.executable, 'tests/integration_wildcard_cache_tensors.py', '--comfy-root', str(COMFY), '--report', str(final / 'tensors.json')], final)
run('tutorial-tensors', [sys.executable, 'tests/integration_wildcard_tutorial_tensors.py', '--comfy-root', str(COMFY),
     '--tutorials-root', env['IMPACT_TUTORIALS_DIR'], '--inspire-root', str(COMFY / 'custom_nodes/comfyui-inspire-pack'),
     '--report', str(final / 'tutorial-tensors.json')], final)
run('browser', [sys.executable, '-m', 'pytest', 'tests/test_e2e_wildcard_cache.py', 'tests/test_e2e_wildcard_tutorials.py',
     '-k', 'not fixed_mode_suppresses', '-v', '--tb=short', f'--junitxml={final}/browser.xml'], final)
python, _ = pytest_report(final / 'python.xml')
browser, _ = pytest_report(final / 'browser.xml')
assert python == {'tests': 204, 'failures': 0, 'errors': 0, 'skipped': 0}
assert browser == {'tests': 27, 'failures': 0, 'errors': 0, 'skipped': 0}
node_log = (final / 'node.log').read_text()
node_counts = {name: int(re.search(r'^# ' + name + r' (\d+)$', node_log, re.M).group(1)) for name in ['tests', 'pass', 'fail', 'cancelled', 'skipped']}
assert node_counts['fail'] == node_counts['cancelled'] == node_counts['skipped'] == 0
for label in ['backend', 'tensors', 'tutorial-tensors']:
    report = json.loads((final / f'{label}.json').read_text())
    assert report['successful'] and not report['errors'] and not report['failures'] and not report['skipped']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
assert head == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert hashes == {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in test_files}
(OUT / 'validation.json').write_text(json.dumps({'successful': True, 'head': head, 'base': BASE,
    'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'unit': node_counts, 'python': python,
    'chrome_fixed_matrix': matrix, 'chrome_remaining_including_render': browser, 'chrome_total': 47,
    'test_sha256': hashes, 'production_sha256': hashlib.sha256(fixed).hexdigest(),
    'commands': commands, 'server_managed_separately': True}, indent=2) + '\n')
print('PASS: frozen RGR, 47 Chrome cases including actual render/cache hit, all CPU/Node checks.', flush=True)
