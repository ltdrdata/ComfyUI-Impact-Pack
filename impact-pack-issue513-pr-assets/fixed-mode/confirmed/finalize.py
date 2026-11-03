"""Finalize the local fixed-mode draft from successful, matching evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
from urllib.parse import unquote

import numpy as np
from PIL import Image

ROOT=Path('/mnt/teratera/git/impact-pack-issue513-followup')
OUT=Path(__file__).resolve().parent
ASSETS=OUT.parents[1]
DRAFT=ASSETS.parent/'impact-pack-issue513-pr-draft.md'
PREP=ASSETS.parent/'impact-pack-issue513-preparation.json'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
base=subprocess.check_output(['git','rev-parse','HEAD^'],cwd=ROOT,text=True).strip()
validation=json.loads((OUT/'validation.json').read_text())
rgr=json.loads((OUT/'rgr.json').read_text())
assert validation['successful'] and rgr['successful'] and validation['head']==rgr['head']==head
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
for path,digest in validation['test_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
assert hashlib.sha256((ROOT/'js/impact-pack.js').read_bytes()).hexdigest()==validation['production_sha256']
with socket.socket() as connection:
    assert connection.connect_ex(('127.0.0.1',18190))!=0, 'Stop the isolated server before finalizing'

paths=['modules/impact/impact_server.py','modules/impact/impact_pack.py','modules/impact/wildcards.py',
       '__init__.py','tests/workflows/all-impact-nodes.json','tests/workflows/all-impact-nodes.widgets.json']
unchanged={}
for path in paths:
    data=(ROOT/path).read_bytes()
    assert data==subprocess.check_output(['git','show',base+':'+path],cwd=ROOT)
    unchanged[path]=hashlib.sha256(data).hexdigest()
(OUT/'source-provenance.json').write_text(json.dumps({'head':head,'base':base,'unchanged_from_base':unchanged,
    'changed_files':subprocess.check_output(['git','diff','--name-only',base,head],cwd=ROOT,text=True).splitlines(),
    'new_registered_nodes':0,'new_widgets':0,'commits':int(subprocess.check_output(['git','rev-list','--count',base+'..'+head],cwd=ROOT,text=True))},indent=2)+'\n')

cases=[]
for folder in sorted((OUT/'check-matrix').iterdir()):
    if not folder.is_dir(): continue
    steps=[]
    for phase in ['first','second','changed','manual-seed','resume-populate','next-populate']:
        path=folder/f'{phase}.json'
        if not path.is_file(): continue
        entry=json.loads(path.read_text()); prompt=entry['prompt'][2]
        node=next(n for n,v in prompt.items() if v['class_type'] in ['ImpactWildcardProcessor','ImpactWildcardEncode'])
        preview=next(n for n,v in prompt.items() if v['class_type']=='PreviewAny')
        cached={str(n) for event,data in entry['status']['messages'] if event=='execution_cached' for n in data['nodes']}
        saved=next(n for n in entry['prompt'][3]['extra_pnginfo']['workflow']['nodes'] if str(n['id'])==node)
        seed_index=3 if prompt[node]['class_type']=='ImpactWildcardProcessor' else 5
        steps.append({'phase':phase,'output':entry['outputs'][preview]['text'][0],'execution_seed':prompt[node]['inputs']['seed'],
            'native_seed':saved['widgets_values'][seed_index],'selected_control':saved['widgets_values'][seed_index+1],
            'wildcard_cached':node in cached,'preview_cached':preview in cached,
            'history':str(path.relative_to(ASSETS)),'history_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert not steps[0]['wildcard_cached'] and not steps[0]['preview_cached']
    assert steps[1]['wildcard_cached'] and steps[1]['preview_cached']
    assert steps[0]['native_seed']==steps[1]['native_seed']==5
    assert steps[0]['selected_control']==steps[1]['selected_control']==re.search(r'-(increment|decrement|randomize)-',folder.name).group(1)
    assert not steps[2]['wildcard_cached'] and not steps[2]['preview_cached']
    cases.append({'case':folder.name,'steps':steps})
assert len(cases)==20
(OUT/'cache-evidence.json').write_text(json.dumps({'head':head,'cases':cases},indent=2)+'\n')

reference=Path('/mnt/teratera/git/impact-pack-pr1190-followup-pr-assets/tutorial-audit/baseline-inference/output.png')
render=OUT/'final/browser/test_lbw_tutorial_inference/output.png'
with Image.open(reference) as ref, Image.open(render) as actual:
    rgb=np.asarray(actual.convert('RGB')); expected=np.asarray(ref.convert('RGB'))
    assert rgb.shape==expected.shape
    different=int(np.any(rgb!=expected,axis=2).sum()); assert different==0
    rgb_digest=hashlib.sha256(rgb.tobytes()).hexdigest()
second=json.loads((render.parent/'cached-history.json').read_text())
cached={str(n) for event,data in second['status']['messages'] if event=='execution_cached' for n in data['nodes']}
assert {'10','3','8','9'}<=cached
render_report={'head':head,'sampling_rerun':True,'width':rgb.shape[1],'height':rgb.shape[0],
    'different_rgb_pixels':different,'rgb_sha256':rgb_digest,'reference':str(reference),'candidate':str(render),
    'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(render.read_bytes()).hexdigest(),
    'second_queue_cached':{'ImpactWildcardEncode':'10' in cached,'KSampler':'3' in cached,'VAEDecode':'8' in cached,'SaveImage':'9' in cached}}
(OUT/'render-comparison.json').write_text(json.dumps(render_report,indent=2)+'\n')

node_count=validation['unit']['pass']
backend=json.loads((OUT/'final/backend.json').read_text())['test_groups']
tensors=json.loads((OUT/'final/tensors.json').read_text())['tests_run']
tutorial=json.loads((OUT/'final/tutorial-tensors.json').read_text())['test_groups']
body=DRAFT.read_text()
body=body[:body.index('## Results')]+f'''## Results

Validated at `{head[:7]}` with the same frozen tests in both Red runs:

- Node RGR: **8 failed / 3 passed → 11 passed → the same 8 failed / 3 passed**.
- Chrome RGR: **6 failed / 2 passed → 8 passed → the same 6 failed / 2 passed**.
- **{node_count} Node tests**, **204 Python tests**, **{backend} CPU integration groups**,
  **{tensors} real-model tensor groups** and **{tutorial} tutorial tensor groups** passed.
- **47 Chrome tests passed**: 20 fixed-mode cases, 18 other cache/native cases,
  8 original PNG restoration cases and 1 original LBW inference/cache case.
- Final checks: **0 failures, 0 errors, 0 skips**. Ruff and whitespace checks passed.

| Unchanged fixed input | First queue | Repeat queue | Text/manual seed edit |
| --- | --- | --- | --- |
| Direct plain/comment/empty text | miss | hit | text edit: miss |
| Direct active choices | miss | hit | miss |
| Linked plain text | miss | hit | text edit: miss |
| Linked active choices | miss | hit | miss |

These observations apply to both wildcard nodes and their downstream PreviewAny.
The native seed remains `5` in the first two queues, while the selected increment,
decrement or randomize policy is preserved. Before/after transitions back to populate
resume the existing core behavior.
[Queue histories and hashes](impact-pack-issue513-pr-assets/fixed-mode/confirmed/cache-evidence.json).

Actual LBW render: **512 × 768, zero differing RGB pixels** against the trusted
reference. Its second queue cached **ImpactWildcardEncode, KSampler, VAEDecode and
SaveImage**, and retained the original fixed-mode seed and randomize selection.
[Pixel and cache comparison](impact-pack-issue513-pr-assets/fixed-mode/confirmed/render-comparison.json)
· [Rendered image](impact-pack-issue513-pr-assets/fixed-mode/confirmed/final/browser/test_lbw_tutorial_inference/output.png).

![Fixed LBW workflow after a cached second queue](impact-pack-issue513-pr-assets/fixed-mode/confirmed/final/browser/test_lbw_tutorial_inference/final.png)

[Full validation and commands](impact-pack-issue513-pr-assets/fixed-mode/confirmed/validation.json)
· [RGR and identical failure identities](impact-pack-issue513-pr-assets/fixed-mode/confirmed/rgr.json)
· [Frozen tests](impact-pack-issue513-pr-assets/fixed-mode/confirmed/frozen-tests.json)
· [Source and trusted fixture hashes](impact-pack-issue513-pr-assets/fixed-mode/confirmed/source-provenance.json)
· [Complete diff](impact-pack-issue513-pr-assets/combined.diff).

The full node catalog was not rerun; its unchanged fixtures are used by the affected
native checks. Original PNGs other than LBW cover widget restoration, not complete
inference. API callers and upstream nodes explicitly changing inputs can still cause
execution. Issue #513 remains open until the change is reviewed and published.
'''
assert not re.search(r'\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#513',body,re.I)
DRAFT.write_text(body)
(ASSETS/'combined.diff').write_bytes(subprocess.check_output(['git','diff',base,head],cwd=ROOT))
for target in re.findall(r'\]\(([^)]+)\)',body):
    if not target.startswith('http'):
        assert (DRAFT.parent/unquote(target)).is_file(),target
validation['server_stopped']=True
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
prep=json.loads(PREP.read_text()); prep.update({'status':'validated_local_draft','updated_at_utc':datetime.now(timezone.utc).isoformat(),
    'head':head,'base':base,'partial_resolution':False,'resolution_scope':'Reported fixed-mode automatic seed control behavior; explicit input changes remain supported',
    'draft_sha256':hashlib.sha256(DRAFT.read_bytes()).hexdigest(),'rgr':str(OUT/'rgr.json'),'validation':str(OUT/'validation.json'),
    'source_provenance':str(OUT/'source-provenance.json'),'render_comparison':str(OUT/'render-comparison.json'),
    'cache_evidence':str(OUT/'cache-evidence.json'),'temporary_server':'stopped'})
PREP.write_text(json.dumps(prep,indent=2)+'\n')
print(json.dumps({'head':head,'Node':node_count,'Python':204,'Chrome':47,'render_pixels_different':different,'native_seed_and_selected_policy_preserved':True}))
