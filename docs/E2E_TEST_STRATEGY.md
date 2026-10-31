# E2E Test Strategy

E2E tests use the real ComfyUI frontend in Google Chrome. Execution tests submit
work by clicking **Run**, then verify both backend results and frontend updates.
The supported workflow paths are native JSON/PNG loading and direct UI editing.
API-format import compatibility is outside the scope of these fixes. Some tests
use API-shaped fixtures for setup, then check the native workflow and UI behavior;
that setup does not extend the supported import scope.

## Test coverage

| Script | Setup through the UI | Assertions |
| --- | --- | --- |
| `tests/e2e_wildcards.py` | Create a blank workflow, search for and place ImpactWildcardProcessor and Preview as Text, drag a connection, enter text, select modes, click Run | UTF-8 file expansion, populate/fixed/reproduce behavior, editable/disabled text fields, websocket feedback, preview text |
| `tests/e2e_dd_compat.py` | Load the prepared Detailer graph through the frontend file input, click Run | Successful SEGSDetailer inference with DifferentialDiffusion, output image loaded in the UI, non-degenerate image, changed pixels by mean/std comparison |
| `tests/test_e2e_switches.py` | Create and wire nodes; import prepared graphs and UI-created JSON, reload Chrome, delete connected nodes, reconnect and click Run | Slot identities and typed links, list order, non-default selection and routing, fresh output after reconnecting, output preservation when a Reroute disconnects during loading |
| `tests/test_e2e_widget_catalog.py` | Load the UI-created catalog and pan to each node at 100% zoom | Every registered node exactly once; live widget names, order, and values match the fixed baseline under positional and named restoration |
| `tests/test_e2e_lora_wildcard_source.py` | Search for the node, select non-default values, native Export and file import | Delayed creation responses cannot overwrite saved widget names/order/values or category options in either restoration mode |
| `tests/e2e_lora_wildcard_inference.py` | Import a prepared graph, choose modes, Run, native Export and reload | Real SD1.5 LoRA loading and BREAK conditioning match core-node reference images pixel-for-pixel; four modes preserve output count/order/text and native saved widgets |
| `tests/e2e_tutorials.py` | Import the original Switch tutorial PNGs, change selection widgets, save and reimport JSON | Every selected branch and final output matches direct core-node image results; both selection modes retain the final preview |
| `tests/e2e_regional.py` | Import the regional img2img tutorial and prepared variants, click Run | Exact color masks, output differs from a VAE roundtrip, each regional prompt affects its own region more than distant pixels, browser previews match server images |
| `tests/e2e_conditionings.py` | Import conditioning and regional-prompt graphs, click Run, save and reimport through the file input | Combine/Concat results match core ComfyUI nodes; regional prompts retain both items in order; dynamic slots and links survive loading |

Widget behavior requires direct interaction. Inference tests may load a
prepared workflow because manually assembling that graph is not their subject.
The Detailer graph remains defined in `e2e_dd_compat.py` and is supplied as
a JSON file payload to the frontend file input.

`tests/e2e_ui.py` provides the shared browser interactions. Workflow construction
and editing use the UI; JavaScript reads graph state and locates canvas controls.
Negative tests deliberately inject frontend faults to prove that checks detect
broken behavior. Tests do not call `app.graph.add`, invoke `app.queuePrompt`, or
POST workflows directly. Read-only HTTP requests inspect execution history and
retrieve output images after the UI submits the workflow.
Image assertions also wait for each node's preview to load the current output
filename, then compare its decoded browser pixels with the server image. Reading
those pixels uses a detached canvas without changing the workflow or its UI.

The shell tests using `curl` are API integration tests, not UI E2E tests.
`bash tests/test_encoding.sh http://127.0.0.1:18188` uses the same isolated
server and samples without modifying its configuration or managing processes.

## Prerequisites

- A working ComfyUI installation with dependencies matching its requirements.
- Impact Pack; also Impact Subpack for the Detailer test.
- Python Playwright and Google Chrome. The scripts default to
  `chromium.launch(channel="chrome")`.
- For Detailer: `SD1.5/realcartoonPixar_v8.safetensors` in checkpoints,
  `bbox/face_yolov8m.pt` in Ultralytics models, and the portrait
  `ComfyUI_00156_.png` in the input directory. Adjust `PROMPT` in the script
  for different local assets.
- Conditioning tests use the same checkpoint and do not require Impact Subpack.

Wildcard UI tests require no models. Configure the **test copy** of Impact
Pack to use its tracked samples in `impact-pack.ini`:

```ini
[default]
custom_wildcards = /absolute/path/to/test-copy/tests/wildcards/samples
wildcard_cache_limit_mb = 50
```

To repeat the wildcard UI test in on-demand mode, set the limit to `0.0001`
and restart the test server. Both modes use the same UI assertions.

For `tests/test_e2e_lora_wildcard_source.py`, instead set `custom_wildcards` to
the test copy's `tests/wildcards/lora_samples` directory and restart. The two
categories have no model dependencies: this suite checks native workflow
restoration without inference. It delays real `/impact/get_items` responses so
the default category arrives after the restored category. Python tests cover
file containment, configured paths, parsing, loading/encoding calls, selection
modes and preservation of existing wildcard results separately.

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
IMPACT_E2E_ARTIFACTS=/tmp/impact-lora-source-e2e \
python -m pytest tests/test_e2e_lora_wildcard_source.py -v
node --test tests/lora_wildcard_source.test.mjs
```

For RGR, run the same tests against the incoming implementation, the fix, and
the incoming implementation again. The final Red must fail on the same
assertions; keep the tests and trusted fixtures unchanged. Restore the fix
before final catalog verification. Catalog recording uses the ordinary Full
Cache setup below, not the LoRA sample directory.

For real LoRA inference, configure `custom_wildcards` as
`tests/wildcards/model_samples` and restart the isolated server. Reuse local
models with `--models-directory`: the script requires
`v1-5-pruned-emaonly-fp16.safetensors` under checkpoints and
`SD1.5/etc/add_detail.safetensors` plus `SD1.5/etc/detailmaker.safetensors` under
loras. It downloads no assets. The reference uses core LoRA loaders, two CLIP
encodings joined by ConditioningConcat, and identical sampler settings. Cases
include a repeated LoRA, signed/scientific weights, literal prompt syntax,
trailing text, an extra LoRA, all four selection modes, and native reload.
Unpatched and distinct-LoRA images must differ so parity cannot pass with
silently ignored patches or duplicated batch entries.
For exact pixel comparisons, start the server with `--deterministic
--force-fp32 --fp32-vae --disable-dynamic-vram --use-pytorch-cross-attention`.
Default FP16/dynamic-VRAM output is not a bitwise pixel oracle for this test.
Reference text enters CLIP through a linked string to avoid the frontend's
dynamic-prompt widget changing literal braces before submission.

```bash
python tests/e2e_lora_wildcard_inference.py --base-url http://127.0.0.1:18188 \
  --artifacts /tmp/impact-lora-inference
```

The separate model integration check compares every conditioning tensor and
all MODEL/CLIP patch tensors and strengths with core nodes, and verifies that
the original inputs remain unpatched. It uses default ComfyUI model settings
and the same local assets, without a running server. Only unused HTTP route
registration is substituted. It performs no sampling or browser actions:

```bash
python tests/integration_lora_wildcard_tensors.py --comfy-root /path/to/ComfyUI
```

The model samples also include a negative-weight row before two original rows
with the same LoRA name. The inference suite selects the saved duplicate label,
then exports and reloads it under both widget restoration modes; text, pixels
and widget names/order/values must still identify the original second row.
The tensor check separately verifies that empty categories preserve ordinary
prompt punctuation and BREAK sections exactly as the original node did.

## Server setup

Use an isolated checkout/copy and a separate ComfyUI user/output directory.
The scripts create workflows and enqueue jobs; do not point them at a busy
production server. They do not start/stop the server or replace its config.

From the ComfyUI directory containing the test copy:

```bash
python main.py --listen 127.0.0.1 --port 18188 \
  --disable-all-custom-nodes \
  --whitelist-custom-nodes comfyui-impact-pack comfyui-impact-subpack
```

Only Impact Pack is needed for the wildcard UI test. When testing inside a
separate base directory, use ComfyUI's `--base-directory`, `--models-directory`,
and `--input-directory` options to keep test data separate and reuse local models.

## Run

From the Impact Pack root:

```bash
python tests/e2e_wildcards.py --base-url http://127.0.0.1:18188
python tests/e2e_dd_compat.py --base-url http://127.0.0.1:18188
python tests/e2e_conditionings.py --base-url http://127.0.0.1:18188
```

Switch regression tests use pytest and need no models:

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
IMPACT_E2E_ARTIFACTS=/tmp/impact-switch-e2e \
python -m pytest tests/test_e2e_switches.py -v
```

Each case receives a fresh Chrome context and saves its own screenshot and
trace. `-k workflow_import` selects loading/reloading; `-k disconnect` selects
disconnect-related cases. Node creation, link dragging, file input, browser reload,
Delete, and Run use browser actions. JavaScript only reads graph state.
The frontend may restore wildcard slots to their declared `*` type on load;
the tests preserve link types and verify execution rather than transient labels.

The Switch cases use a blank workflow before importing, and the loader
waits for replacement nodes after configuration. Prepared workflows have three
distinct inputs or outputs and `select=2`; assertions compare against the file,
not just the graph observed before loading. Deleting an unselected connection
must preserve selection and output. The Reroute case verifies that rejecting an
old Reroute connection during loading does not shift the remaining output slots.
Interactive reconnect cases change the source text to detect stale results.
Wildcard-output cases create ValueReceiver and Switch/List nodes through the UI,
connect them in both orders, run, and save/reload a native JSON workflow. They
check the widget names and order plus positional `widgets_values`, following
the repository's workflow compatibility rule in `AGENTS.md`.

The all-node catalog requires **Full Cache mode** because the trusted snapshot
includes wildcard picker values displaying `Select Wildcard 🟢 Full Cache`.
In the test copy's `impact-pack.ini`, set `wildcard_cache_limit_mb` above the
total size of bundled and custom wildcard files, then restart the server.
The example value of `50` is sufficient only when that total is below 50 MB.
Both pytest and the recorder check `/impact/wildcards/list/loaded` before loading
the catalog and fail with setup instructions if `on_demand_mode` is true.
Keep the trusted fixtures unchanged; the separate wildcard UI suite covers
On-Demand behavior.

The all-node catalog is a native workflow exported by ComfyUI, not a graph
assembled in Python or JavaScript. Before starting a node addition, run the
catalog pytest command below against the trusted fixtures and require a pass.
Keep those fixtures unchanged while implementing the new node. Then extend the
catalog on a test server:

```bash
python tests/test_e2e_widget_catalog.py --base-url http://127.0.0.1:18188 \
  --artifacts /tmp/impact-catalog-recording
```

The recorder first loads the existing workflow and compares it with the fixed
snapshot in both restoration modes. It then searches for and places only the
new registered node types through Chrome, including aliases and deprecated
nodes, and clicks native **Export** into the artifact directory. Existing nodes
and their expected records are preserved. It reloads this candidate in both
modes, then completes browser error checks and artifact saving before replacing
either fixture. Failure during these steps leaves both original files unchanged;
when no nodes are new, the recorder only verifies them.
The baseline records frontend widget name/value pairs in their original order,
including frontend-added controls and empty lists for nodes without widgets.
Text widgets are also checked against their DOM input values. No image matching
or backend input schema is used as the widget oracle.

Run the regression test separately:

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
IMPACT_E2E_ARTIFACTS=/tmp/impact-catalog-e2e \
python -m pytest tests/test_e2e_widget_catalog.py -v
```

Pytest never regenerates the baseline. Review fixture changes against the widget
compatibility rule instead of accepting new names or order by rerecording them.
Loading a workflow through broken code must never redefine its expected state.
Fix existing failures before extending the catalog, and restore missing fixtures
from a trusted revision. Include the new node, workflow addition, and its widget
record in the same change, then rerun the full catalog E2E.
Recorder regression cases deliberately corrupt widget names, order, values, and
value types. Separate cases inject an asynchronous browser error or a trace-saving
failure after exporting a candidate. Each checks that failure leaves both trusted
fixture files byte-for-byte unchanged.
The catalog suite includes a static fixture-coverage check, ordinary Chrome UI
checks, and recorder-protection tests with injected faults. The three recorder
publication cases use two real node types and a scoped registration list to test
successful saving and session failures; they do not establish full catalog coverage.
The source registration list detects missing or newly added nodes. This catalog
tests native workflow loading and rendering; its unconnected nodes are not an
inference workflow. Runtime behavior remains covered by the other E2E suites.
Model and input filenames are compared as saved widget values. Their files are
not required for this loading test; ComfyUI may report missing-model setup
warnings. Node registration and widget comparisons are still required.
Two direct-creation cases also check that label-filter presets start at `all`,
append a selected label to the text field, and retain both values after export
and reload.

Branching tests connect Inversed Switch through three distinct image operations
back into Switch, including both legacy names. Selection changes through the UI
must produce the expected pixels and final output, including after reimport.
Their contract follows the tutorial's `switch.md`: prompt mode disconnects links
before execution and uses ImpactInt/Primitive selectors; execution mode accepts
computed integer outputs and requests only the selected input. The execution-mode
cases derive selection from a core GetImageSize node's batch-count output.
An input that raises on execution must be harmless while unselected, must raise
when selected, and must become harmless again after switching back.
The prompt-selection cases also check multiple consumers per output and removal
of every unselected link. Chained switches with an unconnected selected output
must disable that branch and recover when the selection becomes valid again.
Extra PreviewImage outputs on intermediate branches are limited to prompt mode.
The tutorial warns that output nodes (including PreviewBridge) inside execution-mode
branches can force evaluation and propagate ExecutionBlocker to the rejoining Switch;
its supported example removes or mutes those endpoints.

The suite also covers both selection modes, the legacy Switch names, SEGS
concatenation, and image/mask list and batch nodes. Image checks compare count,
order, dimensions, and pixel values before and after importing the saved graph.
Mask batch cases use asymmetric grayscale patterns and their inverses, assembled
into batches by core ComfyUI nodes. Every output pixel is compared with a PIL
reference using explicit crop boxes and Lanczos resizing. Cases cover a single
input, equal sizes, enlargement, reduction, center cropping on both axes, three
inputs, multiple masks per input, and one-pixel width or height. Disabling center
cropping, reversing output order, or dropping batch items must fail these checks
even when workflow execution succeeds.

Regression checks must fail when file importing is bypassed, selection is
unconditionally reset to 1, or the Inversed Switch configuration guard is
removed. Also check the original loading failure, obsolete self-connection,
and missing `select=0` recovery against their corresponding cases.
Replacing dynamic lazy inputs with a plain dictionary or discarding an Inversed
Switch's routing information after its first consumer must fail the branching tests.
Preview checks include a red-to-blue output update, blocked image requests, and
serving the previous red image at the new output URL. Both broken display cases
must be rejected even though backend execution succeeds.

To test the original PNG examples, clone `ComfyUI-extension-tutorials` next to
ComfyUI or set its location explicitly:

```bash
IMPACT_TUTORIALS_DIR=/path/to/ComfyUI-extension-tutorials \
python tests/e2e_tutorials.py --base-url http://127.0.0.1:18188 \
  --artifacts /tmp/impact-tutorial-e2e
```

This checks `switch-simple.png`, `inversed-switch-simple.png`,
`select_on_execution.png`, and `select_on_prompt.png`. It requires ControlNet
Auxiliary's OpenPose node and its models, plus the samples' input images:
`ComfyUI_00289_.png`, `ComfyUI_temp_fyzuo_00038_.png`, `example.png`, and `chunli.png`.
Inputs and models are not downloaded or replaced by the test.

For `regional-sampler-img2img.png`, restore the original embedded source image
and color map first: the published example contains `[IMAGE DATA]` placeholders.
The same assets are embedded in Inspire Pack's `regional-conditioning-img2img.png`.
This creates a separate JSON and checks that both nodes refer to the same files:

```bash
python - /path/to/ComfyUI-extension-tutorials /tmp/regional-sampler-img2img.json <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
with Image.open(root / "ComfyUI-Impact-Pack/workflow/regional-sampler-img2img.png") as image:
    workflow = json.loads(image.info["workflow"])
with Image.open(root / "ComfyUI-Inspire-Pack/workflow/regional-conditioning-img2img.png") as image:
    source = {node["id"]: node for node in json.loads(image.info["workflow"])["nodes"]}
nodes = {node["id"]: node for node in workflow["nodes"]}
for node, slot in ((35, 3), (49, 1)):
    assert nodes[node]["type"] == source[node]["type"]
    assert nodes[node]["widgets_values"][0] == source[node]["widgets_values"][0]
    data = source[node]["widgets_values"][slot]
    assert data.startswith("data:image/")
    nodes[node]["widgets_values"][slot] = data
Path(sys.argv[2]).write_text(json.dumps(workflow))
PY
```

The test accepts a restored PNG or JSON and normalizes the obsolete image
filenames to `#DATA`, keeping the embedded pixels. It requires Inspire Pack,
Essentials, and the checkpoint named in the sample:

```bash
IMPACT_REGIONAL_WORKFLOW=/path/to/restored/regional-sampler-img2img.json \
python tests/e2e_regional.py --base-url http://127.0.0.1:18188 \
  --artifacts /tmp/impact-regional-e2e
```

Mask expectations come directly from RGB equality on the original color map,
independently of the mask nodes. With the original seed and sampling settings,
the test changes one regional prompt at a time. The average pixel change inside
that region must exceed one 8-bit level and twice the change far outside it.
Boundary margins exclude mask overlap and VAE decoding effects. This detects
ignored or misplaced regions; it does not judge the artistic quality or exact
semantic content of a generated image. A plain VAE roundtrip must also differ
from the sampled output. The test records the measured regional differences.

Headless Chrome is the default and still receives real mouse, keyboard, and
file-input events. To watch the same interactions in a visible Chrome window,
use `--headed` on a machine with a display:

```bash
python tests/e2e_wildcards.py --headed --artifacts /tmp/impact-wildcard-ui
```

Common options:

- `--base-url`: test server URL (default `http://127.0.0.1:18188`).
- `--browser-channel`: Playwright browser channel (default `chrome`).
- `--headed`: show the browser window.
- `--timeout`: execution timeout in seconds (default `600`).
- `--artifacts`: directory for the final screenshot and Playwright trace.
  Defaults to a new temporary directory printed at startup.

Any failed assertion, workflow failure, timeout, or uncaught browser JavaScript
error makes the script exit nonzero. On completion or failure, inspect
`final.png` and `trace.zip` in the artifact directory:

```bash
playwright show-trace /tmp/impact-wildcard-ui/trace.zip
```

Screenshot, trace, and browser cleanup are attempted independently. If the
page has closed, the original test error is preserved and trace saving is
still attempted. Cleanup failures are logged and fail an otherwise successful run.
Browser JavaScript errors are checked after cleanup, including errors raised while
the final screenshot or trace is being saved. An earlier test failure is preserved.

The trace records UI actions and DOM snapshots, so it is possible to distinguish
frontend interaction failures from server or model failures. These scripts are
run separately from the standalone unit tests in `tests/pytest.ini`.
