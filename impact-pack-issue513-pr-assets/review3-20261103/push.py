"""Push only the validated branch with an explicit old-head lease, then update PR body."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
R = ROOT / 'impact-pack-pr-issue513-fixed-seed-controls'
B = ROOT / 'impact-pack-issue513-evidence-publish'
BRANCH = 'fix/issue-513-fixed-seed-controls'
EVIDENCE_BRANCH = 'evidence/issue-513-fixed-seed-controls'
REPO = 'ltdrdata/ComfyUI-Impact-Pack'
OLD = '6d1a48803313c0b38193366bcef8f31dd552e70d'
metadata = json.loads((P / 'publication.json').read_text())
mapping = json.loads((P / 'commit.json').read_text())
assert json.loads((P / 'summary.json').read_text())['completed']
assert metadata['code_head'] == mapping['head']
for repo in (R, B):
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=R, text=True).strip() == BRANCH
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip() == metadata['code_head']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=R, text=True).strip() == metadata['base']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B, text=True).strip() == metadata['evidence_commit']

def remote(repo, names):
    output = subprocess.check_output(['git', 'ls-remote', 'origin', *['refs/heads/' + name for name in names]], cwd=repo, text=True)
    return {line.split()[1].removeprefix('refs/heads/'): line.split()[0] for line in output.splitlines()}

refs = remote(R, [BRANCH, 'Main'])
assert refs['Main'] == metadata['base']
assert refs[BRANCH] in (OLD, metadata['code_head'])
evidence_refs = remote(B, [EVIDENCE_BRANCH])
assert evidence_refs[EVIDENCE_BRANCH] in (metadata['evidence_before'], metadata['evidence_commit'])
actions = {'started_utc': datetime.now(timezone.utc).isoformat(), 'expected_old_head': OLD,
           'new_head': metadata['code_head'], 'code_push_complete': False,
           'evidence_push_complete': False, 'body_updated': False}

def record():
    (P / 'publication-actions.json').write_text(json.dumps(actions, indent=2) + '\n')

record()
if refs[BRANCH] != metadata['code_head']:
    subprocess.run(['git', 'push', '--force-with-lease=refs/heads/' + BRANCH + ':' + OLD,
                    'origin', 'HEAD:refs/heads/' + BRANCH], cwd=R, check=True)
actions['code_push_complete'] = True
record()
if evidence_refs[EVIDENCE_BRANCH] != metadata['evidence_commit']:
    subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/' + EVIDENCE_BRANCH], cwd=B, check=True)
actions['evidence_push_complete'] = True
record()
env = {key: value for key, value in os.environ.items() if key not in ('GITHUB_TOKEN', 'GH_TOKEN')}
pr = json.loads(subprocess.check_output(['gh', 'api', 'repos/' + REPO + '/pulls/1271'], env=env))
assert pr['head']['sha'] == metadata['code_head'] and pr['commits'] == 1 and pr['state'] == 'open'
body = (P / 'pr-body.md').read_text()
assert hashlib.sha256(body.encode()).hexdigest() == metadata['body_sha256']
request = P / 'body-request.json'
request.write_text(json.dumps({'body': body}))
subprocess.run(['gh', 'api', '--method', 'PATCH', 'repos/' + REPO + '/pulls/1271',
                '--input', str(request), '--silent'], env=env, check=True)
actions['body_updated'] = True
actions['completed_utc'] = datetime.now(timezone.utc).isoformat()
record()
print('Validated single code commit force-pushed with exact lease; evidence branch and PR body updated.')
