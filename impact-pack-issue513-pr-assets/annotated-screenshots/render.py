"""Render separate SVG/HTML annotations over the six recorded Chrome captures.

Uses the presentation pattern in the PR #1190 constant-text draft. Screenshots,
histories, outputs and the existing live-browser highlights remain unchanged.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from playwright.sync_api import sync_playwright


OUT = Path(__file__).resolve().parent
ASSETS = OUT.parent
SOURCE = ASSETS / "fixed-mode/seed-refresh/independent-sampler"
GREEN, RED, ORANGE = "#a3e635", "#fb7185", "#ffbf69"
FIXED_BRANCH = {"10", "3", "8", "9"}
CONTROL_BRANCH = {"17", "18", "19"}
CSS = """
*{box-sizing:border-box}body{margin:0;background:#e2e8f0;font-family:"Noto Sans CJK KR",sans-serif}
article{margin:0 auto;background:#0f172a;color:#f8fafc;overflow:hidden}
header{padding:24px 36px}h1{font-size:34px;margin:0 0 8px;font-weight:700}
header p{font-size:21px;margin:0;color:#cbd5e1}header label{display:block;margin-top:10px;font-size:18px;color:#cbd5e1}
.scene{position:relative}.scene>img{display:block;width:100%;height:auto}
.scene>svg{position:absolute;inset:0;pointer-events:none}
.note{position:absolute;top:145px;width:700px;background:rgba(15,23,42,.97);color:#e2e8f0;
padding:16px 20px 20px;border:2px solid;border-radius:10px;font-size:22px;line-height:1.45}
.note strong{display:block;font-size:25px;margin-bottom:8px}.note p{margin:0;overflow-wrap:break-word}
.footer{padding:20px 36px;font-size:19px;line-height:1.6;color:#cbd5e1}.footer a{color:#67e8f9}
.original-only .annotation{visibility:hidden}
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label):
    relative = path.relative_to(ASSETS).as_posix()
    return f'<a href="../{quote(relative)}">{html.escape(label)}</a>'


def rect(region, color, number=None):
    x, y, w, h = (region[k] for k in ("x", "y", "width", "height"))
    result = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="none" stroke="{color}" stroke-width="4"/>'
    if number:
        bx, by = x - 27, y + min(20, h / 2)
        result += f'<circle cx="{bx}" cy="{by}" r="20" fill="{color}" stroke="#0f172a" stroke-width="2"/><text x="{bx}" y="{by + 8}" text-anchor="middle" fill="#0f172a" font-size="26" font-weight="700">{number}</text>'
    return result


report = json.loads((SOURCE / "comparison.json").read_text())
assert report["successful"]
assert all(row["independent_baseline_candidate_different_rgb_pixels"] == 0
           for row in report["independent_branch_pixel_parity"])
assert report["first_fixed_baseline_candidate_different_rgb_pixels"] == 0
records = []
for phase in ("candidate", "baseline"):
    rows = report["reports"][phase]["queues"]
    assert len(rows) == 3
    assert len({row["independent_rgb_sha256"] for row in rows}) == 3
    if phase == "candidate":
        assert len({row["fixed_rgb_sha256"] for row in rows}) == 1
    for row in rows:
        q = row["queue"]
        folder = SOURCE / phase
        source = folder / f"queue-{q}-workflow.png"
        history_path = folder / f"queue-{q}-history.json"
        state_path = folder / f"queue-{q}-widget-state.json"
        regions_path = folder / f"queue-{q}-highlight-regions.json"
        assert digest(source) == row["screenshot_sha256"]
        assert digest(history_path) == row["history_sha256"]
        history = json.loads(history_path.read_text())
        state = json.loads(state_path.read_text())
        regions = {r["node_id"]: r for r in json.loads(regions_path.read_text())}
        values = {node: {w["name"]: w["value"] for w in state[node]["widgets"]}
                  for node in ("10", "3", "17")}
        cached = {str(n) for event, data in history["status"]["messages"]
                  if event == "execution_cached" for n in data["nodes"]}
        prompt = history["prompt"][2]
        assert history["status"]["status_str"] == "success"
        assert history["prompt"][1] == row["prompt_id"]
        assert values["10"]["mode"] == "fixed"
        assert values["10"]["control_after_generate"] == "increment"
        assert values["10"]["seed"] == row["fixed_ui_seed_after_queue"]
        assert values["17"]["seed"] == row["independent_ui_seed_after_queue"]
        assert prompt["10"]["inputs"]["seed"] == row["fixed_execution_seed"]
        assert prompt["17"]["inputs"]["seed"] == row["independent_execution_seed"] == 9000 + q
        assert prompt["3"]["inputs"]["seed"] == values["3"]["seed"] == 20261103
        assert CONTROL_BRANCH.isdisjoint(cached)
        assert set(row["independent_sampler_ancestors"]) == {"4", "5", "7", "16"}
        reuse = phase == "candidate" and q > 1
        assert row["fixed_branch_cached"] == reuse
        assert FIXED_BRANCH <= cached if reuse else FIXED_BRANCH.isdisjoint(cached)
        for branch in ("fixed", "independent"):
            with Image.open(folder / f"queue-{q}-{branch}-output.png") as image:
                assert image.size == (512, 512)
                assert hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() == row[f"{branch}_rgb_sha256"]

        color = GREEN if phase == "candidate" else RED
        # Extend the captured seed rectangle to include mode and the policy row.
        # The two intervening rows are the unchanged LoRA/wildcard selectors.
        control = dict(regions["10"])
        control["y"] -= 3 * control["height"]
        control["height"] *= 5
        groups = [(color, [control]), (color, [regions["3"], regions["9"]]),
                  (ORANGE, [regions["17"], regions["19"]])]
        if phase == "candidate":
            title = f"Patched · Queue {q}/3 — " + ("initial render" if q == 1 else "fixed branch cached")
            note1 = ("Fixed mode keeps seed 513", "mode=fixed and increment stays selected. Submitted seed: 513; visible seed: 513. All three patched queues use 513.")
            note2 = ("Same rendered branch", ("Initial sampling establishes the reference image. " if q == 1 else "Wildcard, sampler, decode and SaveImage are cached. ") + "Sampler seed: 20261103. Pancake RGB bytes match across all three queues.")
        else:
            title = f"Baseline · Queue {q}/3 — " + ("seed advances after the first render" if q == 1 else "fixed branch runs again")
            note1 = ("Fixed mode still advances", f"mode=fixed with increment. Submitted wildcard seed: {row['fixed_execution_seed']}; visible seed: {row['fixed_ui_seed_after_queue']}. Baseline submissions use 513 → 514 → 515.")
            note2 = ("Sampling seed is still fixed", ("Initial render matches the patched first output. " if q == 1 else "Wildcard, sampler, decode and SaveImage run again. ") + "Sampler seed stays 20261103. Resampling can still produce identical pixels.")
        if q == 3:
            title += " after native reload"
        note3 = ("Independent sampler increments", f"Submitted: {row['independent_execution_seed']}; visible: {row['independent_ui_seed_after_queue']} (the next seed). Sampler, decode and SaveImage execute each time; three teapot images have distinct RGB hashes.")
        notes = [note1, note2, note3]
        svg = "".join(rect(box, col, i + 1 if j == 0 else None)
                      for i, (col, boxes) in enumerate(groups) for j, box in enumerate(boxes))
        cards = "".join(f'<div class="note annotation" style="left:{100 + i * 750}px;border-color:{groups[i][0]}"><strong style="color:{groups[i][0]}">{i + 1} · {html.escape(t)}</strong><p>{html.escape(body)}</p></div>'
                        for i, (t, body) in enumerate(notes))
        evidence = [(source, "Original capture"), (history_path, "Queue history"),
                    (state_path, "Visible widget values"), (folder / f"queue-{q}-fixed-output.png", "Pancake output"),
                    (folder / f"queue-{q}-independent-output.png", "Teapot output"),
                    (SOURCE / "comparison.json", "Three-queue comparison")]
        stamp = datetime.fromtimestamp(row["execution_start_ms"] / 1000, timezone.utc).isoformat()
        subtitle = f"Recorded capture {report['reports'][phase]['commit'][:7]} · Prompt {row['prompt_id']} · {stamp}"
        raw = "../" + quote(source.relative_to(ASSETS).as_posix())
        identifier = f"{phase}-queue-{q}"
        page_html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style>
<article id="{identifier}" style="width:2400px"><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>
<label><input type="checkbox" checked onchange="document.querySelector('.scene').classList.toggle('original-only',!this.checked)"> Show annotation layer</label></header>
<div class="scene"><img src="{raw}" width="2400" height="1800" alt="Unchanged original Chrome screenshot"><svg class="annotation" viewBox="0 0 2400 1800" width="2400" height="1800">{svg}</svg>{cards}</div>
<div class="footer">Original capture unchanged. Numbered borders and English notes are separate SVG/HTML layers. Visible sampler values are after-queue values.<br>{' · '.join(link(path, label) for path, label in evidence)}</div></article></html>'''
        (OUT / f"{identifier}.html").write_text(page_html)
        records.append({"id": identifier, "phase": phase, "queue": q, "title": title,
                        "prompt_id": row["prompt_id"], "execution_start_utc": stamp,
                        "source": str(source.relative_to(ASSETS)), "source_sha256": digest(source),
                        "history_sha256": digest(history_path), "highlights": groups, "notes": notes,
                        "evidence": [str(path.relative_to(ASSETS)) for path, _ in evidence],
                        "fixed_rgb_sha256": row["fixed_rgb_sha256"],
                        "independent_rgb_sha256": row["independent_rgb_sha256"]})

assert len({r["source_sha256"] for r in records}) == len({r["prompt_id"] for r in records}) == 6
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    for r in records:
        page = browser.new_page(viewport={"width": 2400, "height": 2200}, device_scale_factor=1)
        page.goto((OUT / f"{r['id']}.html").as_uri())
        page.evaluate("document.fonts.ready")
        assert page.locator("img").evaluate("img => img.complete && img.naturalWidth === 2400 && img.naturalHeight === 1800")
        boxes = page.locator(".note").evaluate_all("ns => ns.map(n => ({bottom:n.offsetTop+n.offsetHeight,right:n.offsetLeft+n.offsetWidth}))")
        assert all(b["bottom"] < 365 and b["right"] <= 2300 for b in boxes), (r["id"], boxes)
        page.get_by_role("checkbox").uncheck()
        assert page.locator(".scene").evaluate("e => e.classList.contains('original-only')")
        page.get_by_role("checkbox").check()
        page.locator("article").screenshot(path=str(OUT / f"{r['id']}.png"))
        assert digest(ASSETS / r["source"]) == r["source_sha256"]
        r["annotated_sha256"] = digest(OUT / f"{r['id']}.png")
        page.close()
    browser.close()
(OUT / "manifest.json").write_text(json.dumps({
    "method": "Unchanged original Chrome screenshots embedded as img; separate SVG borders/numbers and English HTML notes, rendered in Chrome. Annotation toggle exposes the original. No crop, retouching, new inference or generated screenshot content.",
    "reference_style": "/mnt/teratera/git/impact-pack-pr1190-constant-text-pr-assets/annotated-screenshots/render.py",
    "recorded_candidate_head": report["head"], "recorded_base": report["base"],
    "images": records,
}, indent=2) + "\n")
print("Rendered six overlays; source hashes, histories, widget values, cache events and RGB output hashes verified.")
