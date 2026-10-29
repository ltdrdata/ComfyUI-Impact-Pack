# E2E Test Strategy

E2E tests use the real ComfyUI frontend in Google Chrome. They submit work
by clicking **Run**, then verify both backend results and frontend updates.

## Test coverage

| Script | Setup through the UI | Assertions |
| --- | --- | --- |
| `tests/e2e_wildcards.py` | Create a blank workflow, search for and place ImpactWildcardProcessor and Preview as Text, drag a connection, enter text, select modes, click Run | UTF-8 file expansion, populate/fixed/reproduce behavior, editable/disabled text fields, websocket feedback, preview text |
| `tests/e2e_dd_compat.py` | Load the prepared Detailer graph through the frontend file input, click Run | Successful SEGSDetailer inference with DifferentialDiffusion, output image loaded in the UI, non-degenerate image, changed pixels by mean/std comparison |
| `tests/test_e2e_switches.py` | Create and wire nodes; import prepared graphs and UI-created JSON, reload Chrome, delete connected nodes, reconnect and click Run | Slot identities and typed links, list order, non-default selection and routing, fresh output after reconnecting, output preservation when a Reroute disconnects during loading |
| `tests/e2e_conditionings.py` | Import conditioning and regional-prompt graphs, click Run, save and reimport through the file input | Combine/Concat results match core ComfyUI nodes; regional prompts retain both items in order; dynamic slots and links survive loading |

Widget behavior requires direct interaction. Inference tests may load a
prepared workflow because manually assembling that graph is not their subject.
The Detailer graph remains defined in `e2e_dd_compat.py` and is supplied as
a JSON file payload to the frontend file input.

`tests/e2e_ui.py` provides the shared browser interactions. JavaScript evaluation
is limited to reading graph state and locating canvas controls. Tests do not
call `app.graph.add`, assign widget values, invoke `app.queuePrompt`, or POST
workflows directly. Read-only HTTP requests inspect execution history and
retrieve output images after the UI submits the workflow.

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

The suite also covers both selection modes, the legacy Switch names, SEGS
concatenation, and image/mask list and batch nodes. Image checks compare count,
order, dimensions, and pixel values before and after importing the saved graph.
The different-size mask batch case currently exposes an existing error on
ComfyUI 0.43.0: `MakeMaskBatch` passes a three-dimensional mask to
`common_upscale`, which expects a channel dimension. This also fails on the
unmodified Main baseline and remains a failing regression test, without a skip
or expected-failure marker.

Regression checks must fail when file importing is bypassed, selection is
unconditionally reset to 1, or the Inversed Switch configuration guard is
removed. Also check the original loading failure, obsolete self-connection,
and missing `select=0` recovery against their corresponding cases.

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

The trace records UI actions and DOM snapshots, so it is possible to distinguish
frontend interaction failures from server or model failures. These scripts are
run separately from the standalone unit tests in `tests/pytest.ini`.
