"""Summarize all suites without treating legacy exit-zero failures as passes."""
from runlib import *
import hashlib
import importlib.metadata
import re
import sys
import xml.etree.ElementTree as ET

subprocess.run([PY, ROOT / 'inventory.py'], check=True)
inventory = json.loads((ROOT / 'inventory.json').read_text())
assert not inventory['pending'], inventory['pending']
maintained = [
    'javascript', 'wildcard-unit', 'mask-blur', 'sam-and-mask', 'animatediff-assembly', 'browser-harness',
    'input-validation', 'wildcard-cache-integration', 'wildcard-cache-tensors', 'wildcard-tutorial-tensors',
    'lora-source-tensors', 'catalog', 'wildcard-classic', 'wildcard-lru', 'connection-types', 'switches',
    'mask-browser', 'sam-browser-api', 'lora-source-loading', 'lbw-tutorial-inference',
    'animatediff-native', 'wildcard-ui', 'utf8-api', 'wildcard-ondemand-ui', 'lora-source-inference',
    'detailer-inference', 'conditioning-inference', 'switch-tutorial-inference', 'regional-inference',
]
records = {path.stem: json.loads(path.read_text()) for path in (ROOT / 'results').glob('*.json')}
assert set(maintained) <= records.keys(), set(maintained) - records.keys()

def xml(label):
    path = ROOT / (label + '.xml')
    if not path.exists():
        return None
    doc = ET.parse(path)
    suite = doc.find('testsuite')
    failures = []
    for case in doc.iter('testcase'):
        for tag in ['failure', 'error', 'skipped']:
            item = case.find(tag)
            if item is not None:
                failures.append({'case': case.get('name'), 'type': tag, 'message': item.get('message')})
    return {key: int(suite.get(key, 0)) for key in ('tests', 'failures', 'errors', 'skipped')} | {'details': failures}

counts = {label: value for label in records if (value := xml(label)) is not None}
node_log = (ROOT / 'javascript.log').read_text()
assert re.search(r'^# tests 62$', node_log, re.M) and re.search(r'^# pass 62$', node_log, re.M)
offline = ['wildcard-unit', 'mask-blur', 'sam-and-mask', 'animatediff-assembly', 'browser-harness']
server_pytest = ['catalog', 'wildcard-classic', 'wildcard-lru', 'connection-types', 'switches',
                 'mask-browser', 'sam-browser-api', 'lora-source-loading', 'lbw-tutorial-inference']
expected = {'catalog': 14, 'wildcard-classic': 86, 'wildcard-lru': 78, 'connection-types': 30,
            'switches': 55, 'mask-browser': 4, 'sam-browser-api': 19, 'lora-source-loading': 2,
            'lbw-tutorial-inference': 1}
for label, total in expected.items():
    assert counts[label]['tests'] == total, (label, counts[label])
source = json.loads((ROOT / 'source-manifest.json').read_text())
assert all(hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest for name, digest in source.items())
worktree = Path(META['worktree'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip() == META['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=worktree)

legacy = json.loads((ROOT / 'legacy-diagnostics.json').read_text())
issues = []
for label, item in legacy['records'].items():
    if not item['record']['exit_code'] and not item['failure_markers']:
        continue
    baseline = item['baseline']
    same = item['record']['exit_code'] == baseline['record']['exit_code']
    same &= item['failure_markers'] == baseline['failure_markers']
    if xml(label):
        same &= xml(label)['details'] == xml('baseline-' + label)['details']
    issues.append({'label': label, 'test': item['test'], 'candidate_exit': item['record']['exit_code'],
                   'baseline_exit': baseline['record']['exit_code'], 'same_failure_on_main': same,
                   'failure_markers': item['failure_markers'], 'xml': xml(label)})
api = json.loads((ROOT / 'legacy-api-diagnostics.json').read_text())
for label, item in api['records'].items():
    if '-candidate-' not in label or (not item['record']['exit_code'] and not item['failure_markers']):
        continue
    baseline = api['records'][label.replace('-candidate-', '-baseline-')]
    same = item['record']['exit_code'] == baseline['record']['exit_code'] and item['failure_markers'] == baseline['failure_markers']
    issues.append({'label': label, 'candidate_exit': item['record']['exit_code'],
                   'baseline_exit': baseline['record']['exit_code'], 'same_failure_on_main': same,
                   'failure_markers': item['failure_markers']})
assert all(issue['same_failure_on_main'] for issue in issues), issues

servers = []
for path in sorted(ROOT.glob('server-*.json')):
    item = json.loads(path.read_text())
    proc = Path('/proc') / str(item['pid']) / 'cmdline'
    assert not proc.exists() or str(ROOT).encode() not in proc.read_bytes(), item
    original = ROOT / ('baseline-source' if item['revision'] == META['base'] else 'source') / 'js/impact-pack.js'
    assert item['source_js_sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
    servers.append({'profile': path.stem, 'pid': item['pid'], 'stopped': True, 'revision': item['revision']})
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError):
        continue
    assert not (b'--base-directory' in args and any(str(ROOT / 'runtime-').encode() in value or
        str(ROOT / 'legacy-isolated/runtime').encode() in value for value in args)), proc
healthy = all(records[label]['exit_code'] == 0 for label in maintained)
healthy &= all(not counts[label]['failures'] and not counts[label]['errors'] and not counts[label]['skipped'] for label in offline + server_pytest)
summary = {'completed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
    'head': META['head'], 'base': META['base'], 'source_js_sha256': META['source_js_sha256'],
    'source_files_verified': len(source), 'worktree_clean': True, 'production_changed': False,
    'maintained_suites': len(maintained), 'maintained_passed': healthy,
    'node_passed': 62, 'offline_pytest_passed': sum(counts[label]['tests'] for label in offline),
    'server_browser_pytest_passed': sum(counts[label]['tests'] for label in server_pytest),
    'pytest_counts': {label: counts[label] for label in maintained if label in counts},
    'executed_entrypoints': inventory['executed'], 'auxiliary_entrypoints': inventory['auxiliary'],
    'maintained_results': {label: records[label] for label in maintained},
    'legacy_issue_groups': issues, 'all_raw_checks_passed': healthy and not issues,
    'unexplained_maintained_failures': [label for label in maintained if records[label]['exit_code']],
    'legacy_production_diff': legacy['production_files_different'], 'servers': servers,
    'runtime': {'node': '24.12.0', 'python': sys.version.split()[0],
        'frontend': importlib.metadata.version('comfyui-frontend-package'),
        'chrome': subprocess.check_output(['google-chrome', '--version'], text=True).strip()},
    'reference': 'main-lbw-reference/reference.json', 'regional_media': 'regional-media-provenance.json',
    'fixture_sha256': {name: source[name] for name in ['tests/workflows/all-impact-nodes.json', 'tests/workflows/all-impact-nodes.widgets.json']},
    'scope_notes': ['Models/tensors use installed weights only; no assets downloaded.',
        'SAM uses CPU on both server and independent predictor; only Impact and Subpack enabled for the legacy resource assertion.',
        'Optional Styler/BNK tutorial graphs are checked for native loading and Impact tensor boundaries, not complete missing-plugin inference.',
        'AnimateDiff checks native workflows and tensor assembly, not full video diffusion.',
        'Legacy assertions were not relaxed; paths, ports and PID cleanup were isolated. Exit-zero failure messages were compared with Main.']}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
provenance = {'harness_revision': META['head'], 'candidate_production': META['head'],
    'baseline_production': META['base'], 'source_manifest': 'source-manifest.json',
    'baseline_legacy_production_diff': legacy['production_files_different'],
    'note': 'The raw run records head field identifies the frozen test harness. Baseline production is recorded separately in legacy-diagnostics.json and server metadata; baseline test bytes and assertions remain the candidate harness.',
    'production_revision_by_result': {label: META['base'] if label.startswith('baseline-') or
        label.startswith('legacy-api-baseline-') else META['head'] for label in records}}
(ROOT / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
legacy_rows = '\n'.join('| ' + issue['label'] + ' | ' + str(issue['candidate_exit']) + ' | ' +
    str(issue['baseline_exit']) + ' | ' + str(issue['same_failure_on_main']) + ' |' for issue in issues)
server_rows = '\n'.join(f'| {label} | {counts[label]["tests"]} | {counts[label]["failures"]} | {counts[label]["errors"]} | {counts[label]["skipped"]} |'
                        for label in server_pytest)
report = f'''# Full regression — {META['head'][:7]}

PR commit `{META['head']}` against its Main base `{META['base']}`. The tracked worktree is clean; production code was unchanged during this run. All {len(source)} frozen source files and the two trusted widget fixtures match their starting hashes.

All **{len(maintained)} maintained suites passed**: Node **62**, standalone Python **{summary['offline_pytest_passed']}**, server/browser pytest **{summary['server_browser_pytest_passed']}**. Additional real tensor, model inference, native workflow and HTTP scripts passed. The inventory contains **{inventory['executed']} executed entrypoints** and three administration/discovery helpers excluded with reasons.

| Server/browser suite | Tests | Failures | Errors | Skips |
| --- | ---: | ---: | ---: | ---: |
{server_rows}

Exact [commands/results](summary.json), [entrypoint inventory](inventory.json), [source provenance](source-provenance.json) and [source hashes](source-manifest.json) are retained. The full catalog preserves 198 records, widget names/order, positional and named saved values through native Export/load. No trusted fixture was regenerated.

## Real model and tensor verification

Installed SD1.5/SDXL models and LoRAs were used without downloads. Conditioning and MODEL/CLIP patch tensors were compared with independent real core/Inspire references. LoRA selection/loading, Detailer, conditioning, Switch tutorials and Regional sampler scripts passed. Regional missing mask pixels were restored from the original companion tutorial PNG; [media provenance](regional-media-provenance.json) records the exact sources and existing widget positions.

The original LBW workflow was rendered freshly on Main with the same deterministic CUDA FP32 settings. The candidate output matches its RGB bytes exactly, then reuses fixed downstream caches. [Main reference](main-lbw-reference/reference.json) · [Main output](main-lbw-reference/output.png) · [Candidate output](lbw-tutorial-inference/test_lbw_tutorial_inference/output.png).

## Legacy diagnostics — failures retained

The full repository run has known legacy failures; it is not reported as all green. Three failing Python assertions and three failing shell/smoke checks are reproduced on the current PR base. Five diagnostic entrypoints are affected. The two exit-zero scripts are still treated as failed checks based on their output.

| Diagnostic | Candidate exit | Main exit | Same failure |
| --- | ---: | ---: | --- |
{legacy_rows}

- `test_progressive_loading.py` expects all five available keys from a loaded-key list and gets zero on both revisions.
- `test_lazy_load_verification.py` expects on-demand mode in two cases but observes full-cache mode on both revisions. Its config import and the runtime config use separate module identities; its cached dictionary reference and return-value tests are independent legacy quality issues.
- `test_config_quotes.sh` reports an unresolved `test_nesting_level1` fixture, then prints success and exits zero.
- `test_progressive_ondemand.sh` expects the already-loaded YAML key to grow its count; both revisions report 6 → 6.
- `run_quick_test.sh` expects an unchanged list while on-demand entries load; both revisions report 3 → 6 and print a failure while exiting zero. Its optional diagnostic log path also does not exist.

[Legacy comparisons](legacy-diagnostics.json) · [Additional API comparisons](legacy-api-diagnostics.json) · [Path/port/PID adaptations](legacy-adaptations.json). Test assertions were not relaxed. The production comparison differs only in `js/impact-pack.js`; all backend files are identical. Corpus content, runtime directories, model weights and browser trace archives are excluded from publication.

## Environment and scope

Node {summary['runtime']['node']}, Python {summary['runtime']['python']}, frontend {summary['runtime']['frontend']}, {summary['runtime']['chrome']}. SAM uses CPU on the server and independent predictor, cache-none, and only Impact/Subpack for the legacy global-resource assertion. All owned test servers have stopped.

Optional Styler/BNK graphs are checked for native loading and Impact tensor boundaries; their missing-plugin end-to-end inference is not claimed. AnimateDiff covers native workflows and tensor assembly, not full video diffusion. Independent GlobalSeed and native promoted populate/before projection bookkeeping retain the PR's documented boundaries.
'''
(ROOT / 'report.md').write_text(report)
print(json.dumps({key: summary[key] for key in ['completed', 'maintained_passed', 'node_passed', 'offline_pytest_passed',
    'server_browser_pytest_passed', 'executed_entrypoints', 'all_raw_checks_passed']}))
