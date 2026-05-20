# E2E Test Strategy

End-to-end test strategy for ComfyUI Impact Pack, covering isolated server launch, baseline smoke verification, real workflow execution, and operational notes for reliable test runs.

---

## Table of Contents

1. [Purpose & Scope](#purpose--scope)
2. [Prerequisites](#prerequisites)
3. [Server Launch](#server-launch)
4. [Baseline Smoke Test](#baseline-smoke-test)
5. [Workflow Execution E2E](#workflow-execution-e2e)
6. [Verification Criteria](#verification-criteria)
7. [Operational Notes](#operational-notes)
8. [Extension Points](#extension-points)

---

## Purpose & Scope

End-to-end (E2E) testing validates ComfyUI Impact Pack as it behaves in a real ComfyUI runtime, not only through unit-level assertions. The goal is to catch regressions that surface only when the full stack is loaded: server startup, node registration, REST API responses, frontend page loading, and actual workflow execution through the `/prompt` API.

**Why isolated E2E**
- A developer's ComfyUI installation typically has many custom nodes. Any one of them can fail to import, log warnings, bind frontend routes, or shadow Impact Pack behavior.
- Running Impact Pack against a clean environment removes noise and produces reproducible results.
- `--disable-all-custom-nodes` combined with `--whitelist-custom-nodes` isolates Impact Pack (and its companion subpack) from every other installed custom node while still letting them load normally.

**What this strategy validates**
- ComfyUI server starts successfully with Impact Pack and Impact Subpack loaded.
- The frontend page is served and renders the ComfyUI application shell.
- The `/object_info` REST endpoint returns the full node catalog for both packs.
- A baseline count and a sample of expected node names are present, guarding against silent node-registration regressions.
- Detailer workflows execute through the `/prompt` API and produce non-degenerate outputs, proving the full inference path (detector → SEGSDetailer → SEGSPaste) runs end to end.

**Out of scope** (covered by [Extension Points](#extension-points))
- UI-driven node creation and connection
- Visual regression testing beyond pixel-delta sanity checks
- Performance benchmarking

---

## Prerequisites

### Environment

| Component | Requirement |
|-----------|-------------|
| Python | >= 3.12 |
| Playwright (Python) | >= 1.58.0 |
| Chromium runtime | Installed via `playwright install chromium` |
| ComfyUI repository | Checked out at the parent directory of `custom_nodes/comfyui-impact-pack` |
| Impact Pack | Installed as a custom node (this repository) |
| Impact Subpack | Installed as a custom node alongside Impact Pack (delivers Ultralytics / SAM node types) |

### Directory Layout

E2E tests assume the standard ComfyUI custom-node layout:

```
ComfyUI/
├── main.py
├── custom_nodes/
│   ├── comfyui-impact-pack/      ← this repository
│   └── comfyui-impact-subpack/   ← detector / SAM node provider
└── ...
```

### Install Commands

```bash
# From the ComfyUI repository root
pip install playwright
playwright install chromium
```

Impact Pack's own Python dependencies are expected to be installed already (see `pyproject.toml` / `requirements.txt`). Impact Subpack contributes its own dependency list — install it with its documented procedure.

### Model Assets

Detailer-dependent workflows require model files on disk. Place them at the ComfyUI-relative paths below:

| Path | Purpose |
|------|---------|
| `models/checkpoints/SD1.5/realcartoonPixar_v8.safetensors` | SD1.5 checkpoint used by `CheckpointLoaderSimple` (any compatible SD1.5 checkpoint works; adjust the workflow's `ckpt_name` accordingly) |
| `models/ultralytics/bbox/face_yolov8m.pt` | Face bbox detector used by `UltralyticsDetectorProvider` |
| `models/sams/sam_vit_b_01ec64.pth` | SAM weights used when a workflow includes `SAMLoader` |
| `input/ComfyUI_00156_.png` | Portrait with a clearly visible face, used by `LoadImage` in the reference workflow |

The Ultralytics and SAM node classes themselves ship with Impact Subpack — installing the subpack is the delivery vehicle for those node types, separate from the model weights above.

Pure [baseline smoke](#baseline-smoke-test) testing does not need any of these assets; they are required only once a workflow hits a detector or loader node.

---

## Server Launch

### Default Launch

Launch an isolated ComfyUI instance with Impact Pack and Impact Subpack as the only active custom nodes. This is the default for every test beyond the pure smoke layer:

```bash
# Working directory: the ComfyUI repository root (parent of custom_nodes/)
python main.py \
    --disable-all-custom-nodes \
    --whitelist-custom-nodes comfyui-impact-pack comfyui-impact-subpack \
    --port 18188
```

Most detailer workflows depend on `UltralyticsDetectorProvider`, `SAMLoader`, and related detector/segmenter node types shipped by Impact Subpack. Running without the subpack leaves those node classes unregistered, and any workflow referencing them will fail at prompt validation.

### Minimal Launch (smoke only)

For the pure API smoke path — page load + `/object_info` inspection, no workflow execution — Impact Pack alone is sufficient:

```bash
python main.py \
    --disable-all-custom-nodes \
    --whitelist-custom-nodes comfyui-impact-pack \
    --port 18188
```

This minimal launch is **insufficient for detection-dependent tests**. Use it only when the test consists of startup + `/object_info` inspection.

### Flag Explanation

| Flag | Purpose |
|------|---------|
| `--disable-all-custom-nodes` | Skip import of every custom node in `custom_nodes/`. Eliminates side effects from unrelated packs. |
| `--whitelist-custom-nodes comfyui-impact-pack comfyui-impact-subpack` | Re-enable only the listed folder names. Accepts multiple values separated by spaces. |
| `--port 18188` | Bind to a non-default port so the test server does not collide with a developer's regular ComfyUI instance on 8188. |

### Expected Startup Log Markers

After launch, the server log should include lines similar to:

```
### Loading: ComfyUI-Impact-Pack (V8.28.2)
### Loading: ComfyUI-Impact-Subpack (V<version>)
Skipping <other-custom-node> due to disable_all_custom_nodes and whitelist_custom_nodes
...
To see the GUI go to: http://127.0.0.1:18188
```

The `Skipping ...` lines confirm isolation: other custom nodes are present on disk but were not loaded. The final `To see the GUI ...` line confirms the server is ready to accept connections.

---

## Baseline Smoke Test

The smoke test confirms that an isolated server serves both the frontend page and the node catalog. It is intentionally small and has no external dependencies beyond Playwright.

### Script

```python
# e2e_smoke.py
# Usage: python e2e_smoke.py
# Preconditions:
#   - ComfyUI is running at http://127.0.0.1:18188 with Impact Pack AND
#     Impact Subpack whitelisted (default launch).
#   - Playwright Python + chromium runtime are installed.

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:18188"
REQUIRED_SUBPACK_NODES = {
    "UltralyticsDetectorProvider",
    "SAMLoader",
    "SAMDetectorCombined",
    "SAMDetectorSegmented",
}


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context().new_page()

        # 1. Frontend page loads
        page.goto(f"{BASE_URL}/", wait_until="networkidle")
        title = page.title()
        assert "ComfyUI" in title, f"Unexpected page title: {title!r}"

        # 2. /object_info returns the node catalog
        resp = page.request.get(f"{BASE_URL}/object_info")
        assert resp.status == 200, f"/object_info HTTP {resp.status}"
        object_info = resp.json()

        # 3. Impact Pack nodes are registered
        impact_nodes = [name for name in object_info if name.startswith("Impact")]
        assert len(impact_nodes) >= 85, (
            f"Impact node count regression: got {len(impact_nodes)}, expected >= 85"
        )

        # 4. Impact Subpack nodes are registered
        missing = REQUIRED_SUBPACK_NODES - set(object_info.keys())
        assert not missing, f"Subpack nodes missing: {sorted(missing)}"

        print(f"Title:        {title}")
        print(f"Total nodes:  {len(object_info)}")
        print(f"Impact nodes: {len(impact_nodes)}")
        print(f"Sample:       {impact_nodes[:3]}")

        browser.close()


if __name__ == "__main__":
    main()
```

### Expected Output

Observed baseline against Impact Pack v8.28.2 + Impact Subpack on an isolated server (default launch with both packs whitelisted):

```
Title:        *Unsaved Workflow - ComfyUI
Total nodes:  858
Impact nodes: 87
Sample:       ['ImpactSegsAndMask', 'ImpactSegsAndMaskForEach', 'ImpactFlattenMask']
```

Additional Subpack-contributed node names verified present: `UltralyticsDetectorProvider`, `SAMLoader`, `SAMDetectorCombined`, `SAMDetectorSegmented`.

Exact values will drift as the codebase evolves; the assertions above validate the minimum contract, not literal equality.

---

## Workflow Execution E2E

The baseline smoke test confirms node registration, but it does not exercise execution logic. Workflow execution E2E posts a concrete graph to `/prompt`, polls `/history/{prompt_id}` until completion, and downloads output artifacts from `/view` for inspection. This catches regressions in the inference path that are invisible to catalog-level checks.

### API Pattern

```
POST /prompt              — submit {prompt, client_id}, receive {prompt_id}
GET  /queue               — running + pending prompts (progress monitoring)
GET  /history/{prompt_id} — status + outputs once execution completes
GET  /view?filename=...   — download an individual output artifact (PNG, etc.)
```

Polling cadence of ~3s on `/history` is sufficient; the endpoint returns an empty object while the prompt is still in flight and populates fully on completion.

### Flat Prompt Format

The `/prompt` endpoint expects a **flat** graph: a dict keyed by node ID, where each value is `{class_type, inputs}`. Socket connections are expressed as two-element lists `[upstream_node_id, output_slot_index]`.

```python
PROMPT = {
    "ckpt": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "SD1.5/realcartoonPixar_v8.safetensors"},
    },
    "pos": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["ckpt", 1], "text": "a detailed face, sharp focus"},
    },
    "img": {
        "class_type": "LoadImage",
        "inputs": {"image": "ComfyUI_00156_.png"},
    },
    "detector": {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": "bbox/face_yolov8m.pt"},
    },
    "detail": {
        "class_type": "SEGSDetailer",
        "inputs": {
            "image": ["img", 0],
            "segs": ["bbox_segs", 0],
            # ... sampler knobs elided ...
            "noise_mask_feather": 20,   # non-zero triggers DifferentialDiffusion path
        },
    },
}
```

### Pitfall: Subgraph Blueprints

Workflow JSON exported from the ComfyUI UI may contain high-level blueprint nodes such as `workflow/Impact::MAKE_BASIC_PIPE`. These are template/subgraph references that the UI expands client-side; they are not valid `class_type` values for direct `/prompt` submission.

For programmatic E2E, **flatten blueprints into their concrete constituent nodes** before posting. For example, a `MAKE_BASIC_PIPE` blueprint flattens into:

```
CheckpointLoaderSimple → (model, clip, vae)
CLIPTextEncode          (positive prompt)
CLIPTextEncode          (negative prompt)
ToBasicPipe             (model, clip, vae, positive, negative)
```

The reference implementation below demonstrates this flattening.

### Reference Implementation

`tests/e2e_dd_compat.py` is a validated reference workflow covering the critical detailer path:

```
LoadImage
    → UltralyticsDetectorProvider
    → BboxDetectorSEGS
    → SEGSDetailer(noise_mask_feather=20)   # activates DifferentialDiffusion compat
    → SEGSPaste
    → PreviewImage (paste output)
    → PreviewImage (untouched input)
```

Pass criteria:
- Submission returns HTTP 200 with a `prompt_id`.
- `/history/{prompt_id}` eventually reports `status.status_str == "success"` with no `execution_error` messages.
- Both expected `PreviewImage` outputs are present in `history[prompt_id].outputs`.
- Input preview and paste preview have matching dimensions.
- Paste preview has non-degenerate statistics (`std >= 1.0`).
- Paste preview differs from input (`abs(mean_delta) >= 0.005` or `abs(std_delta) >= 0.005`) — equality would indicate the detailer path, and therefore the DifferentialDiffusion compat shim, was bypassed.

Typical observed deltas for the reference image: `mean_delta ≈ 0.01`, `std_delta ≈ 0.03`. These are small because `SEGSPaste` only rewrites the cropped face region; the majority of the frame is untouched and subtracts out.

---

## Verification Criteria

The smoke test is considered to PASS when **all** of the following hold:

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | Server startup log contains `Loading: ComfyUI-Impact-Pack` | Inspect server stdout / log tail |
| 2 | Server startup log contains `Loading: ComfyUI-Impact-Subpack` (default launch) | Inspect server stdout / log tail |
| 3 | Log contains `Skipping ... due to disable_all_custom_nodes and whitelist_custom_nodes` for at least one other custom node (when other nodes are installed) | Inspect server log |
| 4 | `http://127.0.0.1:18188/` returns HTTP 200 and a page title containing `ComfyUI` | `page.goto(...)` + `page.title()` |
| 5 | `GET /object_info` returns HTTP 200 with a JSON body | `page.request.get(...).status` and `.json()` |
| 6 | `/object_info` contains at least 85 Impact-prefixed node names | Count keys where `name.startswith("Impact")` |
| 7 | Required Subpack nodes present: `UltralyticsDetectorProvider`, `SAMLoader`, `SAMDetectorCombined`, `SAMDetectorSegmented` | Set membership against `object_info` keys |

The baseline threshold of 85 was chosen below the observed value of 87 to tolerate minor refactors that rename or remove a handful of nodes without triggering a false failure. Raise the threshold deliberately when new nodes ship; lower it only with a reviewed explanation.

For workflow execution tests, pass criteria are scenario-specific; see the per-test criteria listed alongside each reference implementation.

---

## Operational Notes

### Cache Busting

ComfyUI caches sampler outputs across runs when node inputs are identical. This can mask regressions — a workflow may appear to "pass" because it is replaying a cached success.

| Strategy | When to use |
|----------|-------------|
| Per-run seed randomization (`seed = int(time.time()) & 0xFFFFFFFF`) | Default; cheapest invalidation for sampler-bearing nodes |
| Full server restart | After changing Python source inside `modules/impact/` or loaded packages |
| Clear `modules/impact/__pycache__/` | When `.pyc` files may be stale relative to edited `.py` files |

When restarting the server, kill any prior instance first and confirm the port is free before relaunching:

```bash
pkill -9 -f 'python main.py'
# wait a moment, then verify nothing is still listening on the test port
curl -fsS http://127.0.0.1:18188/system_stats && echo "still up" || echo "port free"
```

Only relaunch once the probe reports the port is free. Launching while a dying process still holds the socket produces confusing `address already in use` errors downstream.

### Verifying Internal Code Paths Executed

Workflow-level pass criteria (`status_str == "success"`, no exception, non-zero pixel delta) prove the graph ran end to end. They do **not** prove that a specific internal function was reached. A bug that silently bypasses a compat shim can still return `success`.

To confirm a specific branch executed, temporarily instrument the target:

```python
# modules/impact/utils.py  (temporary)
import logging
def apply_differential_diffusion(...):
    logging.warning("[E2E-MARKER] apply_differential_diffusion:execute")
    ...
```

Run the workflow, then grep the server log for the marker:

```bash
grep 'E2E-MARKER' /tmp/server.log
```

Absence of the marker despite `status_str == "success"` is a signal that the code path was skipped — typically because an upstream dispatch picked a different branch. Remove the marker before committing.

### Log File Decoding

Progress bars emitted by `tqdm` (used by samplers, detectors, SAM) write carriage returns (`\r`) rather than newlines, collapsing a long run onto a single physical line in the log file. Naive `grep` on that file may report only the final progress state.

Normalize before grepping:

```bash
tr '\r' '\n' < /tmp/server.log | grep -F '[E2E-MARKER]'
```

Or in Python:

```python
with open("/tmp/server.log", "r", errors="replace") as f:
    text = f.read().replace("\r", "\n")
```

---

## Extension Points

The smoke and reference workflow tests are the baseline verification layers. Future E2E scenarios should build on the same isolated-launch foundation:

### Reference Test Implementation

`tests/e2e_dd_compat.py` — validated workflow covering LoadImage → UltralyticsDetectorProvider → BboxDetectorSEGS → SEGSDetailer → SEGSPaste → PreviewImage. Demonstrates the full `/prompt` + `/history` + `/view` lifecycle and the pixel-delta assertion pattern. Treat it as the canonical template for new workflow-execution tests.

### Additional Workflow Scenarios

Beyond the detailer compat path, useful workflow-level tests include: FaceDetailer end-to-end (KSampler-generated face pipe), SEGSDetailer with `cycle > 1` (iterative refinement), MASK_TO_SEGS + SEGSPaste (mask-driven editing), and Impact Switch / Pipe nodes (control-flow regressions).

### UI-Driven Node Creation

Use Playwright to open the frontend, drag an Impact node from the node library onto the canvas, and connect inputs/outputs. Validates that frontend metadata (category, display name, input schema) stays synchronized with backend definitions.

### Node Signature Regression Detection

Snapshot the full `/object_info` payload for a known-good release, then diff against the current response. Flag any change in input type, input name, output type, or output count. Useful as a pre-release guard against accidental public API breakage.

### Headed Mode for Debugging

For interactive debugging, launch Playwright with `headless=False` and optionally `slow_mo=500`. Pair with `page.pause()` at the point of failure to inspect the live browser state.

```python
browser = p.chromium.launch(headless=False, slow_mo=500)
# ... later ...
page.pause()  # opens Playwright Inspector
```

### Cross-Browser Coverage

Extend beyond chromium by parameterizing the browser launcher over `p.chromium`, `p.firefox`, and `p.webkit`. Impact Pack's frontend surface is thin, but cross-browser validation guards against regressions introduced by future frontend-facing features.
