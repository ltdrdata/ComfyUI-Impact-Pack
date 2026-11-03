"""Prepare PR body and an append-only evidence commit; no network writes."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit
from html.parser import HTMLParser
from markdown_it import MarkdownIt
import hashlib
import json
import re
import shutil
import subprocess

P = Path(__file__).resolve().parent
A, ROOT = P.parent, P.parent.parent
R = ROOT / 'impact-pack-pr-issue513-fixed-seed-controls'
B = ROOT / 'impact-pack-issue513-evidence-publish'
REPO = 'ltdrdata/ComfyUI-Impact-Pack'
mapping = json.loads((P / 'commit.json').read_text())
summary = json.loads((P / 'summary.json').read_text())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip()
assert summary['completed'] and head == mapping['head'] and mapping['validated_working_tree_matches_commit']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=B)
evidence_before = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B, text=True).strip()
assert evidence_before == 'c56c16025652baaae977adce6b0f1a80064c46f0'
for name, digest in mapping['source_sha256'].items():
    assert hashlib.sha256(subprocess.check_output(['git', 'show', head + ':' + name], cwd=R)).hexdigest() == digest
for name, digest in json.loads((P / 'source-bundle.json').read_text()).items():
    assert hashlib.sha256(subprocess.check_output(['git', 'show', head + ':' + name], cwd=R)).hexdigest() == digest
    assert hashlib.sha256((P / 'validated-source' / name).read_bytes()).hexdigest() == digest
prefix = A.name + '/' + P.name
old = (P / 'pr-draft-before.md').read_text()
details = '<details>' + old.split('<details>', 1)[1]
details = details.replace(A.name + '/audit-fixes-20261103', prefix)
body = f'''Issue [#513](https://github.com/{REPO}/issues/513): automatic seed control reruns wildcard nodes even in `mode=fixed`. Pause it for both wildcard nodes, including promoted/nested seeds and active or linked populated text.

- Reuse core hooks, arithmetic and native widget objects; preserve widget names, order, saved values and selected policies.
- Resolve host modes, restore shared policies before serialization/on error, and handle optional binding failures at one adapter boundary while preserving native errors.

**User impact:** fixed + randomize also freezes choices in `populated_text`; use populate for automatic variation. Manual/upstream edits still invalidate caches. Reload the browser after updating.

**Boundaries:** independent Inspire GlobalSeed remains external. Promoted populate/before projection bookkeeping remains a [separate frontend issue draft](impact-pack-issue513-frontend-projection-identity-issue-draft.md). Partially addresses #513.

**Verification:** one commit `{head[:7]}` on `Main` `56a34b9`; Node 24.12.0 **62**, Python **373**; Chrome cache/tutorial **62**, promoted **24 classic + 24 LRU**, catalog **14 / 198 records** passed. The coordinate-free mode transition also passed four focused checks. Same 33 seed checks pass before/after refactoring. [Exact results]({prefix}/summary.json) · [Source mapping]({prefix}/commit.json) · [Review dispositions]({prefix}/review-notes.md).

```bash
# Node runtime: tests/README.md#node-unit-tests
node --experimental-vm-modules --test tests/*.test.mjs
python -m pytest tests/test_wildcard_cache_artifacts.py -v
python -m pytest tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_tutorials.py -k 'not inference' -v
```

[Browser setup](https://github.com/{REPO}/blob/{head}/docs/E2E_TEST_STRATEGY.md).

Fresh SD1.5/foodphoto **512×512** renders, three queues per revision with native reload before queue 3. Earlier PR baseline `b1ee82d` advances promoted seeds **513/514/515**; patched stays **513** and caches queues 2–3. Independent sampler **9001/9002/9003** produces three distinct outputs; corresponding independent outputs and initial fixed outputs match exactly. [Workflow]({prefix}/sampling/promoted-two-branch-workflow.json) · [Cache/pixel proof]({prefix}/sampling/comparison.json).

Green marks the patched fixed branch, red the prior failure, orange the independent sampler. Previews open full-size PNGs; evidence links are outside images.

''' + details
assert body.count('<details>') == 3 and body.count('![') == 18 and '<details open' not in body
assert not re.search(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#513', body, re.I)
draft = '# Keep fixed wildcard nodes cached with automatic seed control\n\n' + body
(ROOT / 'impact-pack-issue513-pr-draft.md').write_text(draft)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in ('src', 'href'))

def local_links(path, content):
    parser = Links()
    parser.feed(MarkdownIt('commonmark').enable('table').render(content) if path.suffix == '.md' else content)
    for target in parser.targets:
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        resolved = (path.parent / unquote(parts.path)).resolve()
        assert resolved.is_relative_to(ROOT) and resolved.is_file(), (path, target)
        yield resolved

selected = set(local_links(ROOT / 'impact-pack-issue513-pr-draft.md', draft))
excluded = {'remote-before.json', 'pr-draft-before.md', 'preparation-before.json',
            'pr-body.md', 'publication.json', 'bundle-files.json'}
for path in P.iterdir():
    if path.is_file() and path.suffix in ('.json', '.xml', '.log', '.md', '.py', '.patch') and path.name not in excluded:
        selected.add(path)
for folder in ('results', 'sampling', 'validated-source', 'refactor-before', 'initial-settings-race', 'diagnostic-readiness-count'):
    selected.update(path for path in (P / folder).rglob('*')
                    if path.is_file() and path.suffix in ('.json', '.xml', '.png', '.html', '.py', '.js', '.mjs', '.md', '.ini', '.log')
                    and '__pycache__' not in path.parts)
for folder in ('mode-browser', 'ordinary-browser', 'promoted-classic', 'promoted-lru', 'catalog-browser'):
    selected.update(path for path in (P / folder).rglob('*.json') if '__pycache__' not in path.parts)
selected.add(P / 'validated-source/.gitattributes')
seen = set()
while True:
    pending = [path for path in selected - seen if path.suffix in ('.md', '.html')]
    if not pending:
        break
    for path in pending:
        seen.add(path)
        selected.update(local_links(path, path.read_text()))
secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
manifest = json.loads((B / 'manifest.json').read_text())
added = {}
for path in sorted(selected):
    relative = path.relative_to(ROOT)
    assert not any(part == 'runtime' or part.startswith('runtime-') or part == '__pycache__' for part in relative.parts), relative
    assert path.stat().st_size < 40 * 1024**2
    if path.suffix != '.png':
        assert not secret.search(path.read_bytes()), path
    target = B / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    record = {'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    added[str(relative)] = record
    if path.is_relative_to(A):
        manifest[str(path.relative_to(A))] = record
(B / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(B / 'README.md').write_text(draft)
metadata = {'code_head': head, 'base': mapping['base'], 'evidence_before': evidence_before,
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'files_copied': len(added),
            'scope': 'Nonblocking review follow-up: one optional API boundary, Escape gesture, shared Node test runtime, refreshed actual render evidence.'}
(B / 'publication.json').write_text(json.dumps(metadata, indent=2) + '\n')
(P / 'bundle-files.json').write_text(json.dumps(added, indent=2) + '\n')
subprocess.run(['git', 'add', '.'], cwd=B, check=True)
subprocess.run(['git', 'commit', '-m', 'Publish final seed-control review verification and refreshed renders'], cwd=B, check=True, stdout=subprocess.DEVNULL)
evidence = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B, text=True).strip()

def public_link(match):
    target = match.group(0)
    if urlsplit(target).scheme:
        return target
    assert (B / target).is_file(), target
    prefix = 'https://raw.githubusercontent.com/' if target.endswith('.png') else 'https://github.com/'
    middle = '/' if target.endswith('.png') else '/blob/'
    return prefix + REPO + middle + evidence + '/' + quote(target, safe='/')

remote = re.sub(r'(?<=\]\()([^)\n]+)(?=\))', public_link, body)
assert remote.count('![') == 18 and remote.count('<details>') == 3
(P / 'pr-body.md').write_text(remote)
metadata |= {'evidence_commit': evidence, 'body_sha256': hashlib.sha256(remote.encode()).hexdigest()}
(P / 'publication.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
