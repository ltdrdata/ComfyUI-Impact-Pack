"""Publish bounded full-regression evidence and update the existing PR body."""
from runlib import *
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlsplit
from markdown_it import MarkdownIt
import hashlib
import re
import shutil

ROOT_GIT = ROOT.parent.parent
B = ROOT_GIT / 'impact-pack-issue513-evidence-publish'
R = Path(META['worktree'])
REPOSITORY = 'ltdrdata/ComfyUI-Impact-Pack'
summary = json.loads((ROOT / 'summary.json').read_text())
assert summary['completed'] and summary['head'] == json.loads((ROOT / 'candidate-source.json').read_text())['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=B)
before = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B, text=True).strip()
assert before == 'a8f91327a0ba07cc1e15b14d44828f7a53684f01'
env = {key: value for key, value in os.environ.items() if key not in ('GH_TOKEN', 'GITHUB_TOKEN')}
pr = json.loads(subprocess.check_output(['gh', 'api', 'repos/' + REPOSITORY + '/pulls/1271'], env=env))
assert pr['head']['sha'] == summary['previous_head'] and pr['state'] == 'open' and pr['commits'] == 1
draft_path = ROOT_GIT / 'impact-pack-issue513-pr-draft.md'
previous = draft_path.read_text()
(ROOT / 'draft-before.md').write_text(previous)
prefix = ROOT.parent.name + '/' + ROOT.name
replacement = f"""**Verification:** one commit `{summary['head'][:7]}` on latest Main `{summary['base'][:7]}`: Node **62**, Python **379**, Chrome **297 passed + 2 pre-existing Main Bypass xfail** (299 scenarios), plus five real validation/tensor scripts. Native workflow checks preserve both Main's multi-node/named values and promoted seed controls. [Rebase report]({prefix}/report.md) · [Exact results]({prefix}/summary.json) · [Source/capture mapping]({prefix}/commit.json). The [prior full regression](impact-pack-issue513-pr-assets/full-regression-057abbe-20261103/report.md) is historical (`057abbe` / `56a34b9`); its six legacy failures are not current-base claims."""
draft, changed = re.subn(r'^\*\*Verification:\*\*.*$', replacement, previous, flags=re.M)
draft = draft.replace('impact-pack-issue513-pr-assets/review3-20261103/sampling', prefix + '/sampling').replace('Previous PR', 'Main baseline')
draft = draft.replace('Earlier PR baseline `b1ee82d`', 'Main baseline `' + summary['base'][:7] + '`')
draft = draft.replace('Manual/upstream edits still invalidate caches.', 'Manual and upstream inputs retain the backend cache semantics; constant text may reuse cached output.')
draft = draft.replace('tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_tutorials.py', 'tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_controls.py tests/test_e2e_wildcard_tutorials.py')
draft = draft.replace('https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/' + summary['previous_head'] + '/docs/E2E_TEST_STRATEGY.md', 'https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/' + summary['head'] + '/docs/E2E_TEST_STRATEGY.md')
assert changed == 1 and draft.count('![') == 18 and draft.count('<details>') == 3 and '<details open' not in draft
draft_path.write_text(draft)
body = draft.split('\n\n', 1)[1]

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in ('src', 'href'))

def linked(path, content):
    parser = Links()
    parser.feed(MarkdownIt('commonmark').enable('table').render(content) if path.suffix == '.md' else content)
    for target in parser.targets:
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        result = (path.parent / unquote(parts.path)).resolve()
        assert result.is_relative_to(ROOT_GIT) and result.is_file(), (path, target)
        yield result

selected = set(linked(draft_path, draft))
selected.add(ROOT / '.gitattributes')
excluded = {'legacy-corpus.json', 'impact-pack-issue513-pr-draft.md', 'impact-pack-issue513-preparation.json', 'draft-before.md', 'pr-body.md', 'publication.json', 'bundle-files.json'}
for path in ROOT.iterdir():
    if path.is_file() and path.suffix in ('.json', '.xml', '.log', '.md', '.py', '.txt', '.patch') and path.name not in excluded:
        selected.add(path)
for name in ('results', 'sampling', 'cache-and-tutorials', 'cache-lru', 'upstream-interactions', 'bridge-fixed', 'bridge-rgr-red', 'bridge-rgr-green', 'bridge-rgr-red-again', 'bypass-main', 'bypass-candidate', 'catalog'):
    selected.update(path for path in (ROOT / name).rglob('*') if path.is_file() and
                    path.suffix in ('.json', '.png', '.html', '.py', '.log', '.md') and '__pycache__' not in path.parts)
seen = set()
while True:
    pending = [path for path in selected - seen if path.suffix in ('.md', '.html')]
    if not pending:
        break
    for path in pending:
        seen.add(path)
        selected.update(linked(path, path.read_text()))
secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
manifest = json.loads((B / 'manifest.json').read_text())
copied = {}
for path in sorted(selected):
    relative = path.relative_to(ROOT_GIT)
    assert not any(part.startswith('runtime-') or part in ('runtime', 'source', 'baseline-source', 'custom_wildcards', '__pycache__') for part in relative.parts)
    assert path.stat().st_size < 40 * 1024**2
    if path.suffix != '.png':
        assert not secret.search(path.read_bytes()), path
    target = B / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    record = {'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    copied[str(relative)] = record
    if path.is_relative_to(ROOT.parent):
        manifest[str(path.relative_to(ROOT.parent))] = record
(B / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(B / 'README.md').write_text(draft)
metadata = {'code_head': summary['head'], 'base': summary['base'], 'evidence_before': before,
            'scope': 'Rebase on Main 46fd549: preserve both native workflow test contracts; current affected regression and fresh Main/patch rendered comparisons. Prior full regression retained as historical.'}
(B / 'publication.json').write_text(json.dumps(metadata, indent=2) + '\n')
subprocess.run(['git', 'add', '.'], cwd=B, check=True)
subprocess.run(['git', 'commit', '-m', 'Publish full regression for wildcard seed controls'], cwd=B, check=True, stdout=subprocess.DEVNULL)
evidence = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B, text=True).strip()
names = list(copied)
raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=B, input=''.join('HEAD:' + name + '\n' for name in names).encode())
offset = 0
for name in names:
    end = raw.index(b'\n', offset)
    size = int(raw[offset:end].split()[2])
    data = raw[end + 1:end + 1 + size]
    offset = end + size + 2
    assert hashlib.sha256(data).hexdigest() == copied[name]['sha256'], name

def public(match):
    target = match.group(0)
    if urlsplit(target).scheme:
        return target
    assert (B / target).is_file(), target
    prefix = 'https://raw.githubusercontent.com/' if target.endswith('.png') else 'https://github.com/'
    middle = '/' if target.endswith('.png') else '/blob/'
    return prefix + REPOSITORY + middle + evidence + '/' + quote(target, safe='/')

remote = re.sub(r'(?<=\]\()([^)\n]+)(?=\))', public, body)
(ROOT / 'pr-body.md').write_text(remote)
metadata |= {'evidence_commit': evidence, 'body_sha256': hashlib.sha256(remote.encode()).hexdigest(),
             'files_verified': len(copied)}
(ROOT / 'publication.json').write_text(json.dumps(metadata, indent=2) + '\n')
(ROOT / 'bundle-files.json').write_text(json.dumps(copied, indent=2) + '\n')
remote_refs = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/Main', 'refs/heads/fix/issue-513-fixed-seed-controls'], cwd=R, text=True)
refs = {line.split()[1]: line.split()[0] for line in remote_refs.splitlines()}
assert refs['refs/heads/Main'] == summary['base']
assert refs['refs/heads/fix/issue-513-fixed-seed-controls'] == summary['previous_head']
subprocess.run(['git', 'push', '--force-with-lease=refs/heads/fix/issue-513-fixed-seed-controls:' + summary['previous_head'],
                'origin', 'HEAD:refs/heads/fix/issue-513-fixed-seed-controls'], cwd=R, check=True)
subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/evidence/issue-513-fixed-seed-controls'], cwd=B, check=True)
request = ROOT / 'body-request.json'
request.write_text(json.dumps({'body': remote}))
subprocess.run(['gh', 'api', '--method', 'PATCH', 'repos/' + REPOSITORY + '/pulls/1271',
                '--input', str(request), '--silent'], env=env, check=True)
print(json.dumps(metadata, indent=2))
