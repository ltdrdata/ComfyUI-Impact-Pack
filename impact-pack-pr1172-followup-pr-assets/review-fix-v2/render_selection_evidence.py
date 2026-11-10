"""Render separate annotation layers over unchanged browser captures."""
import base64
import hashlib
import html
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'overlays'
OUT.mkdir(exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def embedded(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()

css = '''*{box-sizing:border-box}body{margin:0;background:#0f172a;color:#e2e8f0;font-family:Arial,sans-serif}
article{width:2320px;padding:28px}h1{font-size:36px;margin:0 0 14px}header p{font-size:24px;margin:0 0 20px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{border:3px solid var(--tone);border-radius:12px;overflow:hidden;background:#172033}
h2{padding:16px 20px;margin:0;font-size:28px;color:var(--tone)}.scene{position:relative;aspect-ratio:1500/1100;overflow:hidden}
.scene img{display:block;width:146.666667%;height:auto;max-width:none}.scene svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.result{padding:16px 20px;font-size:26px;min-height:116px;line-height:1.35}.result strong{color:var(--tone)}
.selection{display:flex;gap:20px;align-items:center;padding:12px 20px;font-size:23px}.selection img{width:100px;height:100px;border:1px solid #94a3b8}
.prompt{padding:14px 20px;border-top:1px solid #475569;font:18px monospace;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;margin-top:20px;font-size:22px}th,td{text-align:left;border:1px solid #475569;padding:12px}td:nth-child(2){font:19px monospace}
footer{margin-top:18px;font-size:21px;line-height:1.4;color:#cbd5e1}'''

def boxes(tone):
    rectangles = [(88, 172, 264, 142, '#f8c15c', '1'),
                  (588, 172, 263, 370, '#71e4ff', '2'),
                  (1088, 172, 264, 100, tone, '3'),
                  (646, 738, 148, 264, tone, '3')]
    return '<svg viewBox="0 0 1500 1100">' + ''.join(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="5" rx="6"/>'
        f'<text x="{x+5}" y="{y-8}" font-size="26" font-weight="bold" fill="{color}">{label}</text>'
        for x,y,w,h,color,label in rectangles) + '</svg>'

manifest = []
for selection in ['clear', 'full']:
    panels = []
    item = {'selection': selection, 'stage': 'same-input', 'sides': {}}
    for side, run, tone in [('Red', 'chrome-red', '#fb7185'), ('Green', 'chrome-green-core', '#4ade80')]:
        case = ROOT / run / 'browser' / f'test_saved_mask_selection_controls_blocking[{selection}-PreviewBridgeLatent]'
        records = json.loads((case / 'queues.json').read_text())
        first = records[0]
        assert first['stage'] == 'same-input'
        assert first['consumer_ran'] == first['mask_ran']
        assert first['consumer_ran'] == (selection == 'full') if side == 'Green' else first['consumer_ran'] != (selection == 'full')
        if side == 'Green':
            assert len(records) == len({r['prompt_id'] for r in records}) == 3
            assert all(r['consumer_ran'] == r['mask_ran'] == (selection == 'full') for r in records)
            green_records = records
        screen = case / 'same-input.png'
        saved = case / 'saved-selection.png'
        item['sides'][side.lower()] = {
            'screen': str(screen.relative_to(ROOT)), 'screen_sha256': digest(screen),
            'saved_selection': str(saved.relative_to(ROOT)), 'saved_selection_sha256': digest(saved),
            'queue_record': str((case / 'queues.json').relative_to(ROOT)), 'queues': records,
            'source_manifest': run + '/source.json',
        }
        if first['consumer_ran']:
            outcome = 'Both downstream consumers ran.'
            detail = 'The current MASK is all zero.' if selection == 'clear' else 'The current MASK is all white.'
        else:
            outcome = 'Both downstream consumers were skipped.'
            detail = 'The visible painted preview is retained from the earlier queue; it is not a current output.'
        title = 'Published a5350a6' if side == 'Red' else 'Corrected empty-mask check'
        panels.append(f'''<section class="panel" style="--tone:{tone}"><h2>{side} · {title}</h2>
<div class="scene"><img src="{embedded(screen)}">{boxes(tone)}</div>
<div class="result"><strong>{outcome}</strong><br>{detail}</div>
<div class="selection"><img src="{embedded(saved)}"><span>Same saved selection: {'all zero (Clear)' if selection == 'clear' else 'all one (full mask)'}<br>block = if_empty_mask · restore_mask = always</span></div>
<div class="prompt">Prompt: {first['prompt_id']}</div></section>''')
    assert item['sides']['red']['saved_selection_sha256'] == item['sides']['green']['saved_selection_sha256']
    rows = ''.join(f'<tr><td>{html.escape(r["stage"])}</td><td>{r["prompt_id"]}</td><td>{"Run" if r["consumer_ran"] else "Skipped"}</td></tr>' for r in green_records)
    title = 'Saved Clear stops downstream execution' if selection == 'clear' else 'A full selected mask permits downstream execution'
    content = f'''<!doctype html><meta charset="utf-8"><style>{css}</style><article>
<header><h1>{title}</h1><p>Same first queue after native MaskEditor Save. 1: unchanged input · 2: unchanged controls · 3: downstream consumers.</p></header>
<div class="columns">{''.join(panels)}</div><table><thead><tr><th>Green stage</th><th>Distinct prompt ID</th><th>Both consumers</th></tr></thead><tbody>{rows}</tbody></table>
<footer>Original captures are unchanged. This view crops empty right/bottom canvas space while preserving aspect ratio.<br>Execution history determines whether consumers ran; a retained preview does not prove a new output.</footer></article>'''
    (OUT / (selection + '.html')).write_text(content)
    manifest.append(item)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel='chrome', headless=True)
    try:
        page = browser.new_page(viewport={'width':2320,'height':1800}, device_scale_factor=1)
        for item in manifest:
            name = item['selection']
            page.goto((OUT / (name + '.html')).as_uri())
            page.wait_for_function('Array.from(document.images).every(image => image.complete && image.naturalWidth > 0)')
            page.locator('article').screenshot(path=str(OUT / (name + '.png')))
            item['overlay'] = name + '.png'
            item['overlay_sha256'] = digest(OUT / (name + '.png'))
    finally:
        browser.close()
(OUT / 'manifest.json').write_text(json.dumps({'method':'Unchanged captures with separate HTML/SVG annotations; matching saved-mask hashes and execution stages.','comparisons':manifest},indent=2)+'\n')
print('Rendered two comparisons with unchanged original captures and queue records.')
