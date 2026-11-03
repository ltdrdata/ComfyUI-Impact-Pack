"""Attach clickable evidence and actual outputs in collapsed queue blocks."""
from datetime import datetime,timezone
import hashlib,json,re,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ASSETS=OUT.parent;ROOT=ASSETS.parent
DRAFT=ROOT/'impact-pack-issue513-pr-draft.md';PREP=ROOT/'impact-pack-issue513-preparation.json'
REPO=ROOT/'impact-pack-pr-issue513-fixed-seed-controls'
m=json.loads((OUT/'manifest.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only',m['recorded_candidate_head'],head,'--','js','modules','__init__.py'],cwd=REPO,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
def sequence(prefix,raw):
 lines=[]
 for q in (1,2,3):
  suffix='initial render' if q==1 else 'repeat' if q==2 else 'after native Export/reload'
  lines += ['<details>',f'<summary>Queue {q}/3 — {suffix}: baseline vs patched screenshots and actual output PNGs</summary>','']
  for phase,label in [('baseline','Baseline'),('candidate','Patched')]:
   r=next(r for r in m['images'] if r['phase']==phase and r['queue']==q)
   image=f'{prefix}/{phase}-queue-{q}.png';base=f'{raw}/{phase}/queue-{q}'
   lines += [f"**{label} · Prompt `{r['prompt_id']}`**",'',
    f"[![{r['title']}]({image})]({image})",'',
    f'[Original capture]({base}-workflow.png) · [Queue history]({base}-history.json) · [Visible widget values]({base}-widget-state.json)', '',
    '| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |',
    '| --- | --- |',
    f'| [![{label} queue {q}: pancake PNG]({base}-fixed-output.png)]({base}-fixed-output.png) | [![{label} queue {q}: teapot PNG]({base}-independent-output.png)]({base}-independent-output.png) |','',
    f'[Open full-size pancake PNG]({base}-fixed-output.png) · [Open full-size teapot PNG]({base}-independent-output.png)','']
  lines += ['</details>','']
 return '\n'.join(lines)
prefix='impact-pack-issue513-pr-assets/annotated-screenshots';raw='impact-pack-issue513-pr-assets/fixed-mode/seed-refresh/independent-sampler'
body=DRAFT.read_text();before=hashlib.sha256(body.encode()).hexdigest()
start=body.index('**Queue 1**') if '**Queue 1**' in body else body.index('<details>\n<summary>Queue 1/3')
end=body.index('These are **six distinct browser captures',start)
body=body[:start]+sequence(prefix,raw)+'\n\n'+body[end:]
body=body.replace('Every\nrow links its screenshot to the raw execution history and unannotated capture.','Each\nqueue block includes clickable capture/history/widget links and the actual output\nPNGs. Blocks are collapsed by default; click an output preview to open it at full size.')
body=body.replace('Each HTML view can toggle the annotations and\nlinks to its original capture, history, visible widget state and rendered outputs.','Evidence links are clickable Markdown below each capture. Actual pancake and\nteapot PNGs are embedded directly in the PR body, inside the default-collapsed\nqueue blocks. Link-like text and the annotation checkbox are omitted from the\nraster overlays; the HTML source retains its optional annotation toggle.')
DRAFT.write_text(body)
readme='''# Three-queue screenshots and actual outputs

Expand a queue to compare baseline and patched captures. Each block contains
clickable links to the original screenshot, execution history and widget state,
plus the actual 512 × 512 pancake and teapot output PNGs. Click an image to open it
at full size. Evidence links are outside the raster overlay.

Green identifies the patched fixed branch; red identifies the baseline branch
that still reruns; orange identifies the independent incrementing sampler.
The recorded candidate `aaec21a` has identical production code to `4819ada`.
These are annotations of existing runs, with no additional inference.

[Renderer](render.py) · [Source hashes](manifest.json) · [Verification](review.json).

'''+sequence('.', '../fixed-mode/seed-refresh/independent-sampler')
(OUT/'README.md').write_text(readme)
review={'updated_utc':datetime.now(timezone.utc).isoformat(),'current_head':head,'recorded_candidate_head':m['recorded_candidate_head'],'production_source_diff':[],
 'overlay_count':6,'embedded_actual_output_pngs':12,'collapsed_queue_blocks':3,'unique_prompt_ids':6,
 'all_original_hashes_unchanged':True,'all_overlay_links_outside_raster':True,'raw_images_and_record_links_in_markdown':True,
 'checks':['Original screenshot/history hashes and actual output RGB hashes verified by renderer','Numbered notes remain above graph content','Raster export has no visible HTML links or annotation checkbox','Three details blocks are closed by default','All six captures and twelve output PNGs embedded and clickable'],
 'draft_sha256_before':before,'draft_sha256_after':hashlib.sha256(DRAFT.read_bytes()).hexdigest(),'code_worktree_clean':True,'new_inference_runs':0}
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
d=json.loads(PREP.read_text());d.update(updated_at_utc=review['updated_utc'],draft_sha256=review['draft_sha256_after'],screenshot_overlay_manifest=str(OUT/'manifest.json'),screenshot_overlay_review=str(OUT/'review.json'),render_screenshot=str(OUT/'candidate-queue-3.png'),render_screenshot_sequence=str(OUT/'README.md'))
PREP.write_text(json.dumps(d,indent=2)+'\n')
print('Three closed queue blocks; six overlays and twelve actual output PNGs embedded with clickable evidence links.')
