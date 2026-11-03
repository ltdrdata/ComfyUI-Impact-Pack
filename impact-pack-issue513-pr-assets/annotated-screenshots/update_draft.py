from datetime import datetime,timezone
import hashlib,json,re,subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
ASSETS=OUT.parent
ROOT=ASSETS.parent
DRAFT=ROOT/'impact-pack-issue513-pr-draft.md'
PREP=ROOT/'impact-pack-issue513-preparation.json'
REPO=ROOT/'impact-pack-pr-issue513-fixed-seed-controls'
manifest=json.loads((OUT/'manifest.json').read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
production_diff=subprocess.check_output(['git','diff','--name-only',manifest['recorded_candidate_head'],head,'--','js','modules','__init__.py'],cwd=REPO,text=True).splitlines()
assert not production_diff,production_diff
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
prefix='impact-pack-issue513-pr-assets/annotated-screenshots'
source_prefix='impact-pack-issue513-pr-assets/fixed-mode/seed-refresh/independent-sampler'

def sequence(prefix,raw_prefix):
 lines=[]
 for q in (1,2,3):
  lines += [f"**Queue {q}{' — after native Export/reload' if q==3 else ''}**",'']
  for phase,label in [('baseline','Baseline'),('candidate','Patched')]:
   r=next(row for row in manifest['images'] if row['phase']==phase and row['queue']==q)
   lines += [f"{label} · Prompt `{r['prompt_id'][:8]}` · [Original capture]({raw_prefix}/{phase}/queue-{q}-workflow.png) · [Queue history]({raw_prefix}/{phase}/queue-{q}-history.json) · [Toggle annotations / evidence links]({prefix}/{phase}-queue-{q}.html)",'',
    f"![{r['title']}: numbered seed, fixed render and independent sampler annotations]({prefix}/{phase}-queue-{q}.png)",'']
 return '\n'.join(lines)

body=DRAFT.read_text();before=hashlib.sha256(body.encode()).hexdigest()
body=body.replace('Native Export/reload precedes queue 3. **Green boxes** identify the wildcard seed,\nits fixed sampler seed and pancake output; **orange boxes** identify the independent\nsampler seed and teapot output. On the patched revision the green values/image stay\nthe same while the orange values/image change.',
 'Native Export/reload precedes queue 3. Numbered **green overlays** identify the\npatched wildcard controls, fixed sampler seed and pancake output. **Red overlays**\nidentify the same branch on the baseline, where fixed mode still allows automatic\nseed changes. **Orange overlays** identify the independent sampler seed and teapot\noutput on both revisions. The patched green values/image stay the same while the\norange values/image change. English notes distinguish submitted seeds from the\nvisible post-queue values and link each observation to its execution records.')
start=body.index('| Queue | Baseline | Patched |')
end=body.index('These are **six distinct browser captures',start)
body=body[:start]+sequence(prefix,source_prefix)+'\n\n'+body[end:]
old='''Boxes are transparent outlines added to the live browser before capture. Original
captures, rectangle coordinates, image hashes and full queue IDs are retained.
[Sequence index and hashes]'''
new='''The presentation follows the #1190 constant-text draft: unchanged original Chrome
screenshots are embedded as images, with separate SVG borders/numbers and English
HTML note layers rendered in Chrome. The existing captured coordinates locate the
highlighted controls and outputs. Each HTML view can toggle the annotations and
links to its original capture, history, visible widget state and rendered outputs.
The source hashes, six unique prompt IDs, visible/submitted seeds, cache events and
RGB output hashes were checked again while rendering these overlays. This updates
the presentation of the recorded `aaec21a` run; it does not add new inference runs.
Production code is identical to `4819ada`.
[Overlay sources, hashes and renderer](impact-pack-issue513-pr-assets/annotated-screenshots/README.md)
· [Overlay verification](impact-pack-issue513-pr-assets/annotated-screenshots/review.json)
· [Sequence index and hashes]'''
assert old in body
body=body.replace(old,new).replace('[Full-size sequential evidence]('+source_prefix+'/sequence.md)', '[Full-size annotated sequence]('+prefix+'/README.md)')
for target in re.findall(r'\]\(([^)]+)\)',body):
 if not target.startswith(('http://','https://','#')) and not target.endswith(('annotated-screenshots/README.md','annotated-screenshots/review.json')):
  assert (DRAFT.parent/target).is_file(),target
DRAFT.write_text(body)
readme='''# Annotated three-queue screenshots

Six original Chrome captures from the recorded baseline/candidate sampling run.
The presentation follows the #1190 constant-text draft: separate SVG borders and
numbers plus English HTML notes over the unchanged source image. Each HTML view
has a Show annotation layer checkbox and links to its original capture, history,
visible widgets and actual 512 × 512 outputs.

- Green: patched fixed-mode controls, unchanged sampler seed and pancake output.
- Red: baseline fixed-mode branch; wildcard seed advances and sampling runs again.
- Orange: independent sampler, which still increments and renders a different teapot
  on each of the three queues. Its visible seed is already the next submission seed.

[Renderer](render.py) · [Source hashes and annotations](manifest.json) · [Verification](review.json).

The recorded candidate is `aaec21a`; its production code is identical to `4819ada`.
These are presentation updates of existing captures, not additional inference runs.
Baseline queues 2 and 3 can render identical pancake pixels despite executing again;
cache events, submitted inputs and distinct prompt IDs establish that behavior.

'''+sequence('.', '../fixed-mode/seed-refresh/independent-sampler')
(OUT/'README.md').write_text(readme)
review={'updated_utc':datetime.now(timezone.utc).isoformat(),'reference_draft':str(ROOT/'impact-pack-pr1190-constant-text-pr-draft.md'),
 'current_head':head,'recorded_candidate_head':manifest['recorded_candidate_head'],'production_source_diff':production_diff,
 'overlay_count':len(manifest['images']),'unique_original_captures':len({r['source_sha256'] for r in manifest['images']}),
 'unique_prompt_ids':len({r['prompt_id'] for r in manifest['images']}),'all_original_hashes_unchanged':True,
 'checks':['Source screenshot hashes match original comparison report','History hashes and prompt IDs match original comparison report','Visible seed/policy values match captured widget state','Submitted seed inputs and cache events match annotations','Actual output RGB hashes match comparison report','Notes stay above graph content; image loads at original 2400x1800','Annotation toggle reveals original screenshot','All overlays visually checked for layout and corresponding states'],
 'draft_sha256_before':before,'draft_sha256_after':hashlib.sha256(DRAFT.read_bytes()).hexdigest(),'code_worktree_clean':True,'new_inference_runs':0}
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
prep=json.loads(PREP.read_text())
prep.update(updated_at_utc=review['updated_utc'],draft_sha256=review['draft_sha256_after'],screenshot_overlay_manifest=str(OUT/'manifest.json'),screenshot_overlay_review=str(OUT/'review.json'),render_original_screenshot=str(ASSETS/'fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-workflow.png'),render_screenshot=str(OUT/'candidate-queue-3.png'),render_screenshot_sequence=str(OUT/'README.md'))
PREP.write_text(json.dumps(prep,indent=2)+'\n')
print('Draft and preparation updated; six full-size annotated frames, original captures preserved, production source equivalent.')
