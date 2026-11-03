from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
R = Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
candidate = json.loads((P / 'candidate-source.json').read_text())
head, base = candidate['head'], candidate['base']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip() == head
assert subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=R, text=True).strip() == base
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R)
for name, digest in candidate['source_sha256'].items():
    assert hashlib.sha256((R / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((R / 'js/impact-pack.js').read_bytes()).hexdigest() == 'bf011f3df5c5467a15d71dd5a57864221c1d9b7d5e02c0ea00035f2145494f85'
labels = {'wildcard-unit': 211, 'other-python': 168, 'cache-and-tutorials': 86,
          'cache-lru': 120, 'upstream-interactions': 79, 'bridge-fixed': 8, 'catalog': 14}
results = {}
for label, total in labels.items():
    record = json.loads((P / 'results' / (label + '.json')).read_text())
    suite = ET.parse(P / (label + '.xml')).find('testsuite')
    counts = {key: int(suite.get(key)) for key in ('tests', 'failures', 'errors', 'skipped')}
    expected = {'tests': total, 'failures': 4 if label == 'upstream-interactions' else 0,
                'errors': 0, 'skipped': 2 if label == 'upstream-interactions' else 0}
    assert record['exit_code'] == (1 if label == 'upstream-interactions' else 0) and counts == expected, (label, counts)
    if label == 'upstream-interactions':
        failed = [test.get('name') for test in ET.parse(P / (label + '.xml')).iter('testcase') if test.find('failure') is not None]
        assert len(failed) == 4 and all('control_bridge_stop_with_constant[before_wildcard-' in name for name in failed)
        skipped = [test.find('skipped') for test in ET.parse(P / (label + '.xml')).iter('testcase') if test.find('skipped') is not None]
        assert all(item.get('type') == 'pytest.xfail' and 'Processor Bypass' in item.get('message', '') for item in skipped)
    record['counts'] = counts
    results[label] = record
for label in ['node-all', 'input-validation', 'cache-integration', 'cache-tensors', 'tutorial-tensors',
              'lora-tensors', 'sampling-capture', 'sampling-overlays']:
    record = json.loads((P / 'results' / (label + '.json')).read_text())
    assert record['exit_code'] == 0, label
    results[label] = record
assert '# pass 62\n' in (P / 'node-all.log').read_text()
assert json.loads((P / 'bridge-rgr.json').read_text())['intentional_red']
assert json.loads((P / 'bypass-comparison.json').read_text())['same_two_xfail_on_main_and_candidate']
fixtures = {}
for name in ['tests/workflows/all-impact-nodes.json', 'tests/workflows/all-impact-nodes.widgets.json']:
    data = (R / name).read_bytes()
    assert data == subprocess.check_output(['git', 'show', base + ':' + name], cwd=R)
    fixtures[name] = hashlib.sha256(data).hexdigest()
comparison = json.loads((P / 'sampling/comparison.json').read_text())
assert comparison['successful'] and comparison['base'] == base
assert comparison['candidate_source_js_sha256'] == candidate['source_sha256']['js/impact-pack.js']
servers = []
for path in P.glob('server-*.json'):
    item = json.loads(path.read_text())
    proc = Path('/proc') / str(item['pid']) / 'cmdline'
    assert not proc.exists() or str(P).encode() not in proc.read_bytes(), path
    servers.append({'profile': path.stem, 'pid': item['pid'], 'stopped': True})
summary = {'completed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'head': head,
    'base': base, 'previous_head': candidate['previous_head'], 'commits': 1,
    'source_sha256': candidate['source_sha256'], 'production_js_unchanged': True,
    'test_capture_head': candidate['test_capture_head'],
    'capture_note': 'Production JS and cache/tutorial tests are unchanged. The README and fixed-mode Control Bridge assertions were updated; all eight changed cases rerun, with Main/candidate/Main RGR.',
    'node_passed': 62, 'python_passed': 379, 'browser_passed': 297, 'browser_xfailed': 2,
    'browser_cases_verified': 299,
    'upstream_aggregation': '71 unchanged cases (69 pass, 2 existing Main xfail) plus 8 updated fixed-mode Bridge cases passed. Initial four old-expectation failures retained, not reported as a raw all-green 79-case run.',
    'bypass_comparison': 'bypass-comparison.json', 'bypass_followup': 'bypass-follow-up.md',
    'results': results, 'fixture_sha256': fixtures, 'catalog_records': 198, 'servers': servers,
    'prior_full_regression': '../full-regression-057abbe-20261103/summary.json',
    'prior_full_regression_is_historical': True,
    'scope': 'All affected seed/cache and upstream control cases on classic/LRU, native tutorials/catalog, current loader units, validation and real tensor boundaries, fresh three-render comparison per revision. Unchanged SAM/mask/Switch inference from the previous full run is historical.'}
(P / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
mapping = {'head': head, 'base': base, 'previous_head': candidate['previous_head'], 'commits': 1,
    'source_sha256': candidate['source_sha256'], 'validated_working_tree_matches_commit': True,
    'production_js_unchanged': True, 'rendered_js_matches_commit': True, 'test_capture_head': candidate['test_capture_head'],
    'resolution': 'Combine Main multi-node/named saved-value checks with promoted-widget snapshots and native subgraph link-record ordering. Preserve every upstream test and the original widget layout.'}
(P / 'commit.json').write_text(json.dumps(mapping, indent=2) + '\n')
(P / 'report.md').write_text(f'''# Rebase on latest Main

`{head}` is one commit on `{base}`. Previous PR head: `{candidate['previous_head']}`. The only conflict was the native workflow helper. Main's extra node arguments and type-sensitive positional/named saved-value checks are preserved together with the PR's promoted-widget checks and complete link-record comparison. Production JS is byte-identical to the earlier validated PR. README wording now respects Main's constant-text cache reuse.

## Current validation

Node **62**, Python **379** (211 wildcard/loader + 168 other regressions) passed. Chrome verified **299 scenarios: 297 passed, 2 pre-existing Main xfail** (86 existing cache/tutorial + 120 complete LRU + 79 new Main cache/control interactions + 14 trusted catalog checks). The catalog preserves 198 records and both trusted files are unchanged. Five integration/tensor scripts passed, including all 20 current cache integration groups and original SD1.5/SDXL tensor references. All owned servers stopped.

The first upstream run had four failures because Main's Bridge test expected fixed-mode seed increments. This PR intentionally stops those increments. Its test now requires a stable seed, cache reuse, preserved increment selection, and the existing Stop/reactivate behavior. All eight changed cases passed. The 71 unchanged cases (69 pass, 2 existing Processor Bypass xfail) plus those eight reruns cover all 79 upstream scenarios. This is not presented as one raw all-green 79-case run. [Updated assertions](control-expectations.patch) · [Main/candidate/Main RGR](bridge-rgr.json): two intended seed-increment failures, two passes, same two failures.

The same two Processor Bypass xfail were also rerun directly on both Main and candidate and remained xfail: this seed-control patch does not fix them. [Direct comparison](bypass-comparison.json) · [Follow-up note](bypass-follow-up.md). The follow-up remains a local investigation/draft item, not an externally opened issue or PR.

Fresh actual renders were recorded three times per revision against Main `{base[:7]}`. Main advances wildcard seeds 513/514/515; the patch keeps 513 and caches queues 2–3. Independent sampler seeds 9001/9002/9003 produce distinct outputs with exact corresponding RGB parity. Queue 3 follows native reload. [Comparison](sampling/comparison.json) · [Source mapping](commit.json) · [Exact commands/results](summary.json) · [Rebase range diff](range-diff.txt).

The former full regression of `057abbe` on `56a34b9` is retained as [historical evidence](../full-regression-057abbe-20261103/report.md). Its counts and six legacy failed checks are not claimed as current-base outcomes; Main changed the loader tests. This pass reruns the affected boundaries and does not claim a fresh full SAM/mask/Switch inference run. Independent upstream GlobalSeed and the core promoted populate/before projection issue retain the PR's stated boundaries.
''')
print(json.dumps({'head': head, 'base': base, 'commits': 1, 'node': 62, 'python': 379, 'browser_passed': 297, 'browser_xfail': 2}))
