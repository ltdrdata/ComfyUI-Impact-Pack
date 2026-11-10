import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess

from PIL import Image
from playwright.sync_api import sync_playwright

ASSETS = Path(__file__).resolve().parent
ROOT = Path('/mnt/teratera/git/impact-pack-pr1172-followup')
OUT = ASSETS / 'red-green'
OUT.mkdir(exist_ok=True)
HEAD = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
    HEAD += ' + local fixes'
CASES = [
    ('restart-image', 'Retained tab → server restart → image Bridge', 'test_saved_mask_survives_server_restart_without_reloading_tab[always-PreviewBridge]', 2,
     'MaskEditor Save → Run → restart the isolated server without reloading the tab → first Run. restore_mask = always.',
     'The input file survives, but its cached registration no longer exists. The saved mask is lost.',
     'Reconnection registers the saved file again. The output matches the mask; all 3 post-restart queues pass.'),
    ('restart-latent', 'Retained tab → server restart → latent Bridge', 'test_saved_mask_survives_server_restart_without_reloading_tab[always-PreviewBridgeLatent]', 2,
     'MaskEditor Save → Run → restart the isolated server without reloading the tab → first Run. restore_mask = always.',
     'The input file survives, but the stale registration loses the mask. The latent returns its no-mask output.',
     'Reconnection restores the saved mask. Both restoration modes and all 3 post-restart queues pass.'),
    ('sam-image', 'SAM save → image Bridge', 'test_legacy_sam_temp_mask_reaches_bridge_and_survives_reload[never-PreviewBridge]', 1,
     'First Detect → Save to node → Run. Aspect-preserving resize: width = 0, height = 512; restore_mask = never.', "The saved SAM mask is lost. Black is the image Bridge's no-mask output.", 'The MASK matches the SAM detection.'),
    ('sam-latent', 'SAM save → latent Bridge', 'test_legacy_sam_temp_mask_reaches_bridge_and_survives_reload[never-PreviewBridgeLatent]', 1,
     'First Detect → Save to node → Run. Same VAE; aspect-preserving resize: width = 0, height = 512; restore_mask = never.', "The saved SAM mask is lost. White is the latent Bridge's no-mask output.", 'The MASK matches SAM and reaches the next latent Bridge.'),
    ('clear-image', 'Saved clear → change image → restore mask', 'test_clear_mask_then_refresh_keeps_it_empty[always-PreviewBridge]', 2,
     'MaskEditor Clear / Save → change input color to #c08040 → first Run. restore_mask = always.', 'Clear saved successfully, but restoration reuses the older mask: 1,678 painted pixels.', 'Restoration uses the saved empty mask: 0 painted pixels.'),
    ('clear-latent', 'Saved clear → change latent → restore mask', 'test_clear_mask_then_refresh_keeps_it_empty[always-PreviewBridgeLatent]', 2,
     'MaskEditor Clear / Save → change latent batch_size to 2 → first Run. restore_mask = always.', 'Clear saved successfully, but restoration reuses the older mask: 1,678 painted pixels.', 'Restoration uses the saved empty mask: 0 painted pixels.'),
    ('shared-image', 'Shared saved file → image Bridge', 'test_queued_bridge_results_preserve_the_latest_saved_edit[shared_file-PreviewBridge]', 1,
     'Same saved 256×256 mask: first input 512×512 / always, then 256×256 / if_same_size. Both results held until the second execution ends.',
     'The first execution redirects the shared ID. The second loses the saved mask: 64×64, all black.',
     'The second execution keeps the saved 256×256 mask; maximum pixel error 1.'),
    ('shared-latent', 'Shared saved file → latent Bridge', 'test_queued_bridge_results_preserve_the_latest_saved_edit[shared_file-PreviewBridgeLatent]', 1,
     'Same saved 256×256 mask: first input 512×512 / always, then 256×256 / if_same_size. Both results held until the second execution ends.',
     'The first execution redirects the shared ID. The second loses the saved mask: 32×32, all white.',
     'The second execution keeps the saved 256×256 mask; maximum pixel error 1.'),
]

CSS = '''*{box-sizing:border-box}body{margin:0;background:#e2e8f0;font-family:Arial,"Noto Sans CJK KR",sans-serif;color:#e2e8f0}
article{width:2320px;background:#0f172a;margin:0 auto 32px;padding:26px}h1{margin:0 0 9px;font-size:34px}header p{font-size:23px;margin:0 0 18px;color:#cbd5e1}
.legend{font-size:20px;display:flex;gap:34px;padding-bottom:18px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:22px}.panel{border:3px solid var(--tone);border-radius:12px;overflow:hidden;background:#172033}
.panel h2{margin:0;padding:17px 20px;font-size:27px;color:var(--tone)}.scene{position:relative}.scene img{display:block;width:100%;height:auto}.scene svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.outcome{padding:18px 20px;font-size:24px;font-weight:bold;min-height:96px}.observed{display:flex;gap:22px;padding:0 20px 20px;align-items:center}.masks{display:flex;gap:12px}.masks figure{margin:0;text-align:center}.masks img{width:164px;height:auto;background:#000;image-rendering:pixelated;border:1px solid #64748b}.masks figcaption{font-size:18px;margin-top:7px}.numbers{font-size:21px;line-height:1.55}.numbers strong{color:var(--tone)}.prompt{padding:12px 20px;border-top:1px solid #334155;font-size:17px;font-family:monospace;overflow-wrap:anywhere}.footer{padding-top:16px;font-size:19px;color:#cbd5e1}.sources{max-width:2320px;margin:0 auto 30px;padding:20px;background:white;color:#0f172a;font-size:20px}.sources a{color:#005f99}
'''


def embedded(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outlines(record, tone):
    target = record['node_id']
    source = next(n for n in record['graph'] if n['type'] in ['LoadImage', 'EmptyImage', 'EmptyLatentImage'])
    bridge = next(n for n in record['graph'] if n['type'] in ['PreviewBridge', 'PreviewBridgeLatent'])
    output = next(n for n in record['graph'] if n['id'] == target)
    boxes = []
    for number, node, color in [(1, source, '#f8c15c'), (2, bridge, '#71e4ff'), (3, output, tone)]:
        x, y, w, h = node['bounds']
        x -= 9; y -= 36; w += 18; h += 44
        boxes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" stroke="{color}" stroke-width="6"/>'
                     f'<circle cx="{x-20}" cy="{y+24}" r="18" fill="{color}"/><text x="{x-20}" y="{y+32}" fill="#0f172a" font-size="25" font-weight="bold" text-anchor="middle">{number}</text>')
    v = record['viewport']
    return f'<svg viewBox="0 0 {v["width"]} {v["height"]}">{"".join(boxes)}</svg>'


manifest = []
cards = []
for slug, title, case, step, stage, red_note, green_note in CASES:
    panels = []
    sources = []
    pair = {'id': slug, 'case': case, 'step': step, 'stage': stage, 'sides': {}}
    shared = slug.startswith('shared-')
    restart = slug.startswith('restart-')
    for side, run, sha, tone, note in [
        ('Red', 'feedback-fixes/red-shared' if shared else 'feedback-fixes/red-chrome' if restart else 'feedback-fixes/red-pr1172',
         '4e7b9c0 + cleanup · before queue fix' if shared else 'Candidate before review fixes' if restart else '6572207 · Main + #1172', '#fb7185', red_note),
        ('Green', 'feedback-fixes/green-chrome', HEAD, '#4ade80', green_note),
    ]:
        root = ASSETS / run / 'browser' / case / 'comparisons'
        records = json.loads((root / 'records.json').read_text())
        record = next(r for r in records if r['step'] == step)
        assert record['assertion_passed'] == (side == 'Green')
        raw = root / record['screen']
        with Image.open(raw) as image:
            assert image.size == (record['viewport']['width'], record['viewport']['height'])
        record['screen_sha256'] = digest(raw)
        pair['sides'][side.lower()] = {
            'source_manifest': f'{run}/source.json',
            'records': str((root / 'records.json').relative_to(ASSETS)), 'screen': str(raw.relative_to(ASSETS)),
            'source_sha256': digest(raw), 'expected_sha256': digest(root / record['expected']),
            **{k: record[k] for k in ['prompt_id', 'actual_size', 'expected_size', 'actual_nonzero', 'expected_nonzero', 'max_error']},
        }
        expected = ' × '.join(map(str, record['expected_size']))
        actual = ' × '.join(map(str, record['actual_size']))
        error = record['max_error'] if record['max_error'] is not None else 'size mismatch'
        panels.append(f'''<section class="panel" style="--tone:{tone}"><h2>{side} · {sha}</h2>
          <div class="scene"><img src="{embedded(raw)}" alt="Unmodified {side} screenshot">{outlines(record,tone)}</div>
          <div class="outcome">{html.escape(note)}</div><div class="observed">
          <div class="masks"><figure><img src="{embedded(root/record['actual'])}" alt="Actual output mask"><figcaption>Actual MASK · {actual}</figcaption></figure>
          <figure><img src="{embedded(root/record['expected'])}" alt="Expected mask"><figcaption>Expected · {expected}</figcaption></figure></div>
          <div class="numbers"><strong>{'ASSERTION PASS' if side == 'Green' else 'ASSERTION FAIL'}</strong><br>Nonzero pixels: {record['actual_nonzero']:,}<br>Expected: {record['expected_nonzero']:,}<br>Max pixel error: {error}</div></div>
          <div class="prompt">Prompt ID: {record['prompt_id']}</div></section>''')
        sources.append(f'<a href="../{html.escape(str(raw.relative_to(ASSETS)),quote=True)}">{side} original screenshot</a>')
    assert pair['sides']['red']['expected_sha256'] == pair['sides']['green']['expected_sha256']
    card = f'''<article id="{slug}"><header><h1>{html.escape(title)}</h1><p>{html.escape(stage)}</p></header>
      <div class="legend"><span style="color:#f8c15c">1 · Same input / source settings</span><span style="color:#71e4ff">2 · Same restoration mode</span><span>3 · Observed MASK output</span></div>
      <div class="columns">{''.join(panels)}</div><div class="footer">Same assertion stage. Expected mask SHA-256 matches across Red and Green. Original captures and output files are unchanged; boxes and labels are separate HTML/SVG layers.</div></article>'''
    (OUT / f'{slug}.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style>{card}<nav class="sources">{" · ".join(sources)} · <a href="manifest.json">Source manifest</a></nav></html>')
    cards.append(card)
    manifest.append(pair)

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    for pair in manifest:
        page = browser.new_page(viewport={'width': 2320, 'height': 1800}, device_scale_factor=1)
        page.goto((OUT / f'{pair["id"]}.html').as_uri())
        page.evaluate('document.fonts.ready')
        assert page.locator('img').evaluate_all('(imgs) => imgs.every(i => i.complete && i.naturalWidth > 0)')
        assert page.locator('article a').count() == 0
        output = OUT / f'{pair["id"]}.png'
        page.locator('article').screenshot(path=str(output))
        pair['comparison'] = str(output.relative_to(ASSETS))
        pair['comparison_sha256'] = digest(output)
        for side in pair['sides'].values():
            assert digest(ASSETS / side['screen']) == side['source_sha256']
        page.close()
    browser.close()
(OUT / 'manifest.json').write_text(json.dumps({'candidate': HEAD, 'feedback_fixes': True, 'source_manifest': 'feedback-fixes/green-chrome/source.json', 'method': 'Original PNGs embedded unchanged; separate HTML/SVG annotation layers. Both sides use the same assertion stage and identical expected mask bytes.', 'comparisons': manifest}, indent=2) + '\n')
(OUT / 'evidence.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Preview Bridge Red / Green evidence</title><style>{CSS}</style>{"".join(cards)}</html>')
print(f'Rendered {len(manifest)} paired Red/Green comparisons; source hashes unchanged.')
