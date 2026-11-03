"""Prepare a reviewed PR body and append bounded evidence; does not push."""
from pathlib import Path
from urllib.parse import quote,urlsplit,unquote
from html.parser import HTMLParser
from datetime import datetime,timezone
from markdown_it import MarkdownIt
import hashlib,json,re,shutil,subprocess
P=Path(__file__).resolve().parent;A=P.parent;ROOT=A.parent;R=ROOT/'impact-pack-pr-issue513-fixed-seed-controls';B=ROOT/'impact-pack-issue513-evidence-publish';REPO='ltdrdata/ComfyUI-Impact-Pack'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();mapping=json.loads((P/'commit.json').read_text());summary=json.loads((P/'summary.json').read_text());assert mapping['head']==head and mapping['validated_working_tree_matches_commit'] and summary['completed']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R);assert not subprocess.check_output(['git','status','--porcelain'],cwd=B)
for name,sha in mapping['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=R)).hexdigest()==sha
previous=(ROOT/'impact-pack-issue513-pr-draft.md').read_text();(P/'pre-publication-draft.md').write_text(previous);details='<details>'+previous.split('<details>',1)[1];prefix=A.name+'/'+P.name
body=f'''Issue [#513](https://github.com/{REPO}/issues/513): `mode=fixed` still reruns wildcard nodes when automatic seed control changes their input. Pause those changes for both wildcard nodes, including promoted seeds in native nested subgraphs and active/linked populated text.

- Reuse core hooks, arithmetic and native widget objects. Preserve widget names, order, saved values and selected policies.
- Resolve effective host modes; restore shared policies before declining an unavailable binding. Restore before serialization/on error, and redraw restored after-queue controls before paint.

**User impact:** fixed + randomize also freezes choices in `populated_text`. Use populate for automatic variation. Manual/upstream edits still invalidate caches. Reload the browser after updating.

**Boundaries:** independent Inspire GlobalSeed remains external. Native promoted populate/before projection bookkeeping remains a [separate frontend issue draft](impact-pack-issue513-frontend-projection-identity-issue-draft.md). Partially addresses #513.

**Verification:** one commit `{head[:7]}` on `Main` `56a34b9`; [validated source/capture mapping]({prefix}/commit.json). Node 24.12.0 **60 tests × 10 runs**, Python **373 passed**; Chrome cache/tutorial **62**, catalog **14 / 198 records**, promoted **24 classic + 24 LRU** scenarios verified. Promoted cases contrast host=fixed/interior=populate, with three fixed queues, native reload and manual invalidation. Focused mode cases were rerun after replacing popup interaction; original failures remain in the [exact results]({prefix}/summary.json) and [review notes]({prefix}/review-notes.md).

```bash
# Node 24
node --experimental-vm-modules --test tests/*.test.mjs
python -m pytest tests/test_wildcard_cache_artifacts.py -v
python -m pytest tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_tutorials.py -k 'not inference' -v
```

[Browser setup](https://github.com/{REPO}/blob/{head}/docs/E2E_TEST_STRATEGY.md). RGR: unit **4 failures → 31 passes → same 4 failures**; artifact **2 failures → 3 passes → same 2 failures**; ignoring host mode in Chrome after timing **2 failures → 2 passes → same 2 failures**.

Fresh SD1.5/foodphoto renders, **512×512**, three queues per revision; native reload before queue 3. Previous-PR baseline `b1ee82d` advances promoted seeds **513/514/515**; patched stays **513** and caches queues 2–3. The independent sampler advances **9001/9002/9003**, producing three distinct images. Corresponding independent outputs and initial fixed outputs match exactly. [Workflow]({prefix}/sampling/promoted-two-branch-workflow.json) · [Cache/pixel proof]({prefix}/sampling/comparison.json).

Green marks the patched fixed branch, red the prior failure, orange the independent sampler. Previews open full-size PNGs; evidence links are outside images.

'''+details
assert body.count('<details>')==3 and body.count('![')==18 and '<details open' not in body
assert not re.search(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#513',body,re.I)
draft='# Keep fixed wildcard nodes cached with automatic seed control\n\n'+body;(ROOT/'impact-pack-issue513-pr-draft.md').write_text(draft)
# Keep precise current source plus the pre-audit source needed by the RGR scripts.
for name in mapping['source_sha256']:
 if Path(name).suffix=='.md':continue
 target=P/'validated-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(subprocess.check_output(['git','show',head+':'+name],cwd=R))
for name in ('pytest.ini','test_e2e_widget_catalog.py'):
 target=P/'validated-source/tests'/name;target.write_bytes(subprocess.check_output(['git','show',head+':tests/'+name],cwd=R))
(P/'validated-source/README.md').write_text('''# Source-pinned focused tests\n\nThese files match the published code commit in ../commit.json. From this directory, Node 24 with --experimental-vm-modules --test tests/wildcard_seed_control.test.mjs runs the 31 seed tests; from tests/, Python pytest test_wildcard_cache_artifacts.py runs the 3 artifact tests. The full project Node suite has 60 tests. Complete commands/results and their boundaries are in ../summary.json. No installed models or runtime copies are included.\n''')
(P/'sampling/README.md').write_text(f'''# Three real rendered queues per revision\n\nCandidate source is mapped to code commit {head} in ../commit.json. Captures were recorded on the matching working tree before commit; original metadata remains unchanged. Host mode is fixed while the interior mode is populate. Both baseline and candidate use native UI promotion and Export/reload. Queue 3 follows reload.\n\ncomparison.json records actual POST/history IDs, seed values, cache events and RGB hashes. Original captures, histories, visible widget states and all twelve output PNGs are preserved. The six overlays add separate SVG/HTML annotations without changing original captures. Links remain outside raster images in the PR.\n\nSD1.5 epicrealism + installed foodphoto LoRA, 512x512, deterministic CUDA/PyTorch attention; no model downloads. Historical evidence remains in earlier directories and is not described as newly rerun.\n''')
# Include direct body links plus bounded reports/artifacts from this validation only.
class Links(HTMLParser):
 def __init__(self):super().__init__();self.targets=[]
 def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ('src','href'))
def local_links(path,text):
 parser=Links();parser.feed(MarkdownIt('commonmark').enable('table').render(text) if path.suffix=='.md' else text)
 for target in parser.targets:
  u=urlsplit(target)
  if u.scheme or not u.path:continue
  resolved=(path.parent/unquote(u.path)).resolve();assert resolved.is_relative_to(ROOT) and resolved.is_file(),(path,target)
  yield resolved
selected=set(local_links(ROOT/'impact-pack-issue513-pr-draft.md',draft))
excluded={'pre-publication-draft.md','pr-draft-before.md','preparation-before.json','pr-body.md','publication.json','publication-driver.log','bundle-files.json'}
for f in P.iterdir():
 if f.is_file() and f.suffix in ('.json','.xml','.log','.md','.py','.patch') and f.name not in excluded:selected.add(f)
for folder in ('results','sampling','validated-source','before','mode-browser','mode-lru-browser','browser-mode-rgr-red','browser-mode-rgr-green','browser-mode-rgr-red-again'):
 selected.update(f for f in (P/folder).rglob('*') if f.is_file() and f.suffix in ('.json','.png','.html','.py','.js','.mjs','.md','.ini') and '__pycache__' not in f.parts and not (folder=='before' and f.suffix=='.md'))
# Original non-mode matrix cases were already green; retain their request/history data.
for folder in ('popup-promoted-classic','popup-promoted-lru'):
 selected.update(f for f in (P/folder).rglob('*.json') if '__pycache__' not in f.parts)
selected.update((A/'review2-20261103/core-probe-verified').rglob('*.json'))
# Supporting upstream draft is historical and clearly marked as an unsubmitted draft.
seen=set()
while True:
 pending=[f for f in selected-seen if f.suffix in ('.md','.html')]
 if not pending:break
 for f in pending:seen.add(f);selected.update(local_links(f,f.read_text()))
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
manifest=json.loads((B/'manifest.json').read_text());added={}
for f in sorted(selected):
 rel=f.relative_to(ROOT);assert not any(x=='runtime' or x.startswith('runtime-') or x=='__pycache__' for x in rel.parts),rel
 assert f.suffix not in ('.zip','.pyc','.safetensors','.pth') and f.stat().st_size<40*1024**2
 if f.suffix!='.png':assert not secret.search(f.read_bytes()),f
 target=B/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);record={'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()};added[str(rel)]=record
 if f.is_relative_to(A):manifest[str(f.relative_to(A))]=record
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(B/'README.md').write_text(draft)
meta={'code_head':head,'base':mapping['base'],'prepared_utc':datetime.now(timezone.utc).isoformat(),'new_files':len(added),'scope':'Current audit fixes, bounded regression results and fresh rendered output evidence; earlier evidence retained as historical.'};(B/'publication.json').write_text(json.dumps(meta,indent=2)+'\n')
(P/'bundle-files.json').write_text(json.dumps(added,indent=2)+'\n')
subprocess.run(['git','add','.'],cwd=B,check=True);subprocess.run(['git','commit','-m','Publish validated wildcard seed review fixes and fresh evidence'],cwd=B,check=True,stdout=subprocess.DEVNULL)
evidence=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip()
def link(match):
 target=match.group(0)
 if urlsplit(target).scheme:return target
 assert (B/target).is_file(),target
 return ('https://raw.githubusercontent.com/'+REPO+'/'+evidence+'/' if target.endswith('.png') else 'https://github.com/'+REPO+'/blob/'+evidence+'/')+quote(target,safe='/')
remote=re.sub(r'(?<=\]\()([^)\n]+)(?=\))',link,body);assert remote.count('![')==18 and remote.count('<details>')==3
(P/'pr-body.md').write_text(remote)
meta|={'evidence_commit':evidence,'body_sha256':hashlib.sha256(remote.encode()).hexdigest()};(P/'publication.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
