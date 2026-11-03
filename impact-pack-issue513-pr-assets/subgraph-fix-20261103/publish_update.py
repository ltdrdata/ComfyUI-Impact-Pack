"""Publish bounded, source-pinned review evidence and generate the concise PR body."""
from pathlib import Path
from urllib.parse import quote,urlsplit
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess
P=Path(__file__).resolve().parent;A=P.parent;ROOT=A.parent
R=ROOT/'impact-pack-pr-issue513-fixed-seed-controls';B=ROOT/'impact-pack-issue513-evidence-publish'
REMOTE='ltdrdata/ComfyUI-Impact-Pack'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
report=json.loads((P/'summary.json').read_text());assert report['head']==head and report['completed']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=B)
body=f'''Issue [#513](https://github.com/{REMOTE}/issues/513): fixed wildcard nodes still rerun when automatic seed control changes their input. This pauses automatic seed changes for both wildcard nodes, including promoted seeds in nested native subgraphs and linked/active populated text.

- Delegate to core hooks and seed arithmetic, preserving queue timing and the selected policy. Restore temporary policies before serialization and on error.
- Resolve promoted seed/mode bindings per instance; preserve core first-queue bookkeeping when native widget projections are recreated.
- Preserve widget names, order and saved values. Manual edits and upstream changes still invalidate the cache.

**User impact:** fixed + randomize now also freezes choices in `populated_text`. Use populate with the expression in `wildcard_text` for automatic variation. Tooltips explain this change. Reload the browser after updating.

**Remaining limitation:** Inspire GlobalSeed is an independent upstream controller and can still change fixed inputs. This partially addresses #513.

**Validation:** rebased onto `Main` `56a34b9`; one PR commit `{head[:7]}`. Python **370 passed**, Node **54 passed**; Chrome cache/tutorial **62**, catalog **14** (198 records), connection-type **30**, promoted-seed **24 classic + 24 LRU** scenarios. Promoted cases run ten unchanged fixed queues, then check manual edits, native restoration and submitted seeds on populate resumption. [Exact runs and boundaries]({A.name}/{P.name}/summary.json) · [Review notes]({A.name}/{P.name}/review-notes.md).

```bash
node --experimental-vm-modules --test tests/*.test.mjs
python -m pytest tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_tutorials.py -k 'not inference' -v
python -m pytest tests/test_e2e_widget_catalog.py tests/test_e2e_connection_types.py -v
```

Browser commands require isolated configured servers; [setup](https://github.com/{REMOTE}/blob/{head}/docs/E2E_TEST_STRATEGY.md). RGR: unit **8 fail → 25 pass → same 8 fail**; browser **4 fail → 4 pass → same 4 fail**. Earlier broad model/SAM checks remain [historical evidence]({A.name}/full-regression-20261103/summary.json).

Three fresh rendered queues per revision, SD1.5/foodphoto, 512×512. Baseline is the previous PR rebased onto current Main: promoted seeds advance **513/514/515**. Patched seeds stay **513**, with cache hits on queues 2–3. The independent sampler advances **9001/9002/9003** and produces three distinct images in both revisions. Corresponding independent outputs and initial fixed outputs match exactly. Queue 3 follows native reload. [Workflow and provenance]({A.name}/{P.name}/sampling/README.md) · [Cache/pixel comparison]({A.name}/{P.name}/sampling/comparison.json).

Green = patched fixed branch; red = previous-PR failure; orange = independent sampler. Click any preview for full size; evidence links below are clickable.
'''
for q in range(1,4):
 label='initial render' if q==1 else 'cached fixed branch' if q==2 else 'after native reload'
 body+=f'\n<details>\n<summary>Queue {q}/3 — {label}</summary>\n\n'
 for phase,title in [('baseline','Previous PR'),('candidate','Patched')]:
  base=f'{A.name}/{P.name}/sampling/{phase}/queue-{q}'
  overlay=f'{A.name}/{P.name}/sampling/overlays/{phase}-queue-{q}.png'
  body+=f'**{title}**\n\n[![{title} queue {q} overlay]({overlay})]({overlay})\n\n'
  body+=f'[Original capture]({base}-workflow.png) · [History]({base}-history.json) · [Widget values]({base}-widget-state.json)\n\n'
  body+='| Fixed output | Independent output |\n| --- | --- |\n'
  body+=f'| [![{title} fixed output {q}]({base}-fixed-output.png)]({base}-fixed-output.png) | [![{title} independent output {q}]({base}-independent-output.png)]({base}-independent-output.png) |\n\n'
 body+='</details>\n'
assert body.count('<details>')==3 and body.count('![')==18 and '<details open' not in body
assert not re.search(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#513',body,re.I)
draft='# Keep fixed wildcard nodes cached with automatic seed control\n\n'+body
(ROOT/'impact-pack-issue513-pr-draft.md').write_text(draft)
# Keep existing immutable evidence files, adding the new source-pinned capture.
selected=set()
for f in P.iterdir():
 if f.is_file() and f.suffix in ('.json','.log','.xml','.py','.md') and (f.name.startswith(('latest-','final-','arrow-','settled-','occupied-port-')) or f.name in ('summary.json','review-notes.md','commit.json','checks.py','serve.py','sampling_capture.py','unit_rgr_final.py','publish_update.py','summarize_final.py','verify_publication.py')):
  selected.add(f)
selected.update((P/'results').glob('*.json'))
for folder in ('sampling','final-promoted-classic','latest-promoted-lru','final-promoted-lru-retry','final-mode-classic','final-mode-lru','arrow-mode-classic','arrow-mode-lru','settled-mode-classic','settled-mode-lru','final-browser-rgr-red','final-browser-rgr-green','final-browser-rgr-red-again'):
 selected.update(f for f in (P/folder).rglob('*') if f.is_file() and f.suffix in ('.json','.png','.html','.py','.js','.md') and '__pycache__' not in f.parts)
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
manifest=json.loads((B/'manifest.json').read_text())
for source in sorted(selected):
 rel=source.relative_to(A);assert not any(x.startswith('runtime') for x in rel.parts)
 assert source.stat().st_size<40*1024**2
 if source.suffix!='.png':assert not secret.search(source.read_bytes()),source
 target=B/A.name/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest[str(rel)]={'size':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(B/'README.md').write_text(draft)
metadata={'code_head':head,'base':report['base'],'published_utc':datetime.now(timezone.utc).isoformat(),'new_files':len(selected),'files':len(manifest),'scope':'Current promoted-seed validation and three queue-pair render sequences; previous broad evidence preserved as historical. No runtimes, models or trace ZIPs.'}
(B/'publication.json').write_text(json.dumps(metadata,indent=2)+'\n')
subprocess.run(['git','add','.'],cwd=B,check=True)
subprocess.run(['git','commit','-m','Add promoted fixed-seed regression and rendered queue evidence'],cwd=B,check=True)
evidence=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip()
def remote_link(match):
 target=match.group(0)
 if urlsplit(target).scheme:return target
 assert (B/target).is_file(),target
 return ('https://raw.githubusercontent.com/'+REMOTE+'/'+evidence+'/' if target.endswith('.png') else 'https://github.com/'+REMOTE+'/blob/'+evidence+'/')+quote(target,safe='/')
remote=re.sub(r'(?<=\]\()([^)\n]+)(?=\))',remote_link,body)
assert remote.count('<details>')==3 and remote.count('![')==18
(P/'pr-body.md').write_text(remote);(ROOT/'impact-pack-issue513-pr-published.md').write_text(remote)
metadata.update(evidence_commit=evidence,body_sha256=hashlib.sha256(remote.encode()).hexdigest());(P/'publication.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
