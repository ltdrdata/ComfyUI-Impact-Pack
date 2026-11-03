from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,subprocess

ROOT=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
OUT=Path(__file__).resolve().parent
ASSETS=OUT.parent
DRAFT=ASSETS.parent/'impact-pack-issue513-pr-draft.md'
PREP=ASSETS.parent/'impact-pack-issue513-preparation.json'
BASE='244e93ef9883aaf241d5d0e54ebb57e3b2e840d3'
PRIOR='aaec21a4fbb76e14fa16388ce61d87c2226e5bf0'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
browser=json.loads((OUT/'browser-review.json').read_text())
assert browser['exit_code']==0 and browser['server_stopped']
assert browser['counts']=={'tests':50,'failures':0,'errors':0,'skipped':0}
unit=json.loads((OUT/'unit-review.json').read_text())
node_log=(OUT/'node-suite.log').read_text()
node_counts={key:int(re.search(r'^# '+key+r' (\d+)$',node_log,re.M)[1]) for key in ['tests','pass','fail','skipped']}
assert node_counts=={'tests':20,'pass':20,'fail':0,'skipped':0}
assert unit['green']['exit_code']==0 and unit['red']['counts']['fail']==8
assert unit['red']['failures']==unit['red-again']['failures']
for name,expected in browser['test_sha256'].items():assert digest(ROOT/name)==expected
assert digest(ROOT/'tests/wildcard_seed_control.test.mjs')==unit['green']['test_sha256']
assert digest(ROOT/'js/impact-pack.js')==browser['source_sha256']==unit['green']['source_sha256']
assert not git('diff',PRIOR,'--','js','modules','__init__.py','tests/workflows')
assert git('branch','--show-current')=='fix/issue-513-fixed-seed-controls'
assert git('rev-list','--count',BASE+'..HEAD')=='1'
assert git('rev-parse','HEAD')==PRIOR
changed=['docs/E2E_TEST_STRATEGY.md','tests/README.md','tests/test_e2e_wildcard_cache.py','tests/wildcard_seed_control.test.mjs']
assert sorted(git('diff','--name-only').splitlines())==sorted(changed)
assert git('diff','--cached','--name-only')==''
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
subprocess.run(['git','add',*changed],cwd=ROOT,check=True)
subprocess.run(['git','commit','--amend','--no-edit'],cwd=ROOT,check=True)
head=git('rev-parse','HEAD')
assert git('status','--porcelain')==''
assert git('rev-list','--count',BASE+'..HEAD')=='1'
assert not git('diff',PRIOR,head,'--','js','modules','__init__.py','tests/workflows')
report={
 'head':head,'base':BASE,'prior_validated_head':PRIOR,'worktree':str(ROOT),
 'branch':git('branch','--show-current'),'completed_at_utc':datetime.now(timezone.utc).isoformat(),
 'production_unchanged_from_prior_validation':True,'production_sha256':browser['source_sha256'],
 'trusted_workflow_fixtures_unchanged':True,'commits':1,'worktree_clean':True,
 'latest_node_suite':node_counts,
 'latest_chrome_suite':browser['counts'],
 'node_rgr':{phase:unit[phase]['counts'] for phase in ['red','green','red-again']},
 'mutations_detected':5,
 'newly_detected_mutations':['drop-fixed-arguments','drop-fixed-return','lose-selected-policy'],
 'findings':[
  {'area':'production','result':'Keep existing 22-line core-policy wrapper. No custom seed arithmetic, RNG, persistent state, widget layout change or duplicate cache implementation is needed.'},
  {'area':'unit tests','result':'Fixed-mode forwarding and selected-policy checks missed three deliberately broken variants. Strengthened both hooks, all control policies, legacy false, call count, receiver/arguments/return and exception restoration.'},
  {'area':'unit test scope','result':'Clarified that spies verify the delegation boundary; actual seed updates and queue bookkeeping are checked by Chrome, not a copied core algorithm.'},
  {'area':'Chrome coverage','result':'Added after-queue decrement/randomize for both node types; all three policies now exercise both timings. 24 fixed cases plus 18 other cache/native cases and 8 original PNG restorations passed.'},
  {'area':'documentation','result':'Documented Node test command and boundary; repaired stale pytest selection; removed abandoned-patch history from the PR draft.'}
 ],
 'retained_limits':['Subgraph-promoted seed remains outside the node-local guard.','Inspire GlobalSeed rewrites remain external; prior review is source-only.','This review did not rerun full catalog or rendering; prior original-image and three-queue independent-sampler evidence has identical production code.'],
 'validation_paths':{'unit':'unit-review.json','node':'node-suite.log','browser':'browser-review.json'},
 'test_sha256':browser['test_sha256']|{'tests/wildcard_seed_control.test.mjs':unit['green']['test_sha256']},
 'temporary_server_stopped':True
}
(OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n')
(ASSETS/'combined.diff').write_text(git('diff',BASE,head)+'\n')
body=DRAFT.read_text()
(OUT/'pr-draft-before.md').write_text(body)
body=body.replace('Head: `aaec21a`; one local commit.','Head: `'+head[:7]+'`; one local commit.')
body=body.replace('There is no backend change relative to Main. The earlier comment-only cache draft\nis superseded and archived; comment filtering is not the implementation of this fix.\n','The backend remains unchanged relative to the base commit.\n')
body=body.replace('Node tests load the actual extension with mocked imports and core hooks. They check\npolicy forwarding, legacy false, both callback timings, exception cleanup and\nunrelated nodes.', 'Node tests load the actual extension with core-hook spies. They check policy\nselection, one-call delegation, receiver/argument/return forwarding, legacy false,\nand both callbacks restoring the selected policy after an exception. They do not\nmodel seed arithmetic; real Chrome queues check seed updates and core bookkeeping.')
start=body.index('Validated at `aaec21a` with the same frozen tests in both Red runs:')
end=body.index('| Unchanged fixed input |',start)
results=f'''Latest checks at `{head[:7]}`:

- **20 Node tests passed**; **50 Chrome tests passed**: 24 fixed-mode cases,
  18 other cache/native cases and 8 original PNG restoration cases.
- **0 failures, 0 errors, 0 skips**. Ruff and whitespace checks passed.
- Node RGR with frozen current tests: **8 failed / 3 passed → 11 passed → the
  same 8 failed / 3 passed**. All five deliberately broken variants were detected.
  The previous tests missed three variants: lost fixed-mode arguments, lost return
  values and restoring `increment` instead of the user's selected policy.
- Added after-queue decrement/randomize cases for both wildcard nodes. All three
  automatic policies now exercise both before/after timing.

[Review, scope and source hashes](impact-pack-issue513-pr-assets/draft-review-20261103/review.json)
· [Unit RGR and mutation results](impact-pack-issue513-pr-assets/draft-review-20261103/unit-review.json)
· [Chrome commands and results](impact-pack-issue513-pr-assets/draft-review-20261103/browser-review.json).

Production JavaScript and the trusted workflow fixtures are unchanged from `aaec21a`.
The earlier validation at that commit also passed **204 Python tests**, **8 CPU
integration groups**, **5 real-model tensor groups**, **3 tutorial tensor groups**
and **one original LBW inference/cache case**. Earlier Chrome RGR was **6 failed /
2 passed → 8 passed → the same 6 failed / 2 passed**. Those checks and the image
captures below were retained, not rerun for this test/documentation review.

'''
body=body[:start]+results+body[end:]
body=body.replace('Previously completed side-effect checks at the same head (archived; not rerun for this image):',
 'Previously completed side-effect checks at `aaec21a`, with identical production code:')
body=body.replace('[Full validation and commands]', '[Earlier full validation and commands]')
body=body.replace('execution. Issue #513 remains open until the change is reviewed and published.',
 'execution. This partially addresses #513; it does not close the issue.')
DRAFT.write_text(body)
prep=json.loads(PREP.read_text())
prep.update({'head':head,'draft_sha256':digest(DRAFT),'updated_at_utc':report['completed_at_utc'],
 'review':str(OUT/'review.json'),'prior_validated_head':PRIOR,'render_evidence_head':PRIOR,
 'validation_head':PRIOR,'latest_review_node_tests':20,'latest_review_chrome_tests':50})
PREP.write_text(json.dumps(prep,indent=2)+'\n')
for link in re.findall(r'\]\(([^)]+)\)',body):
    if not link.startswith(('http:','https:','#')):assert (DRAFT.parent/link).exists(),link
print(json.dumps({'head':head,'commits':1,'worktree_clean':True,'draft':str(DRAFT),'review':str(OUT/'review.json')},indent=2))
