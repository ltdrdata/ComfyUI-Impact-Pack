# E2E Test Strategy

E2E tests use the real ComfyUI frontend in Google Chrome. They submit work
by clicking **Run**, then verify both backend results and frontend updates.

## Test coverage

| Script | Setup through the UI | Assertions |
| --- | --- | --- |
| `tests/e2e_wildcards.py` | Create a blank workflow, search for and place ImpactWildcardProcessor and Preview as Text, drag a connection, enter text, select modes, click Run | UTF-8 file expansion, populate/fixed/reproduce behavior, editable/disabled text fields, websocket feedback, preview text |
| `tests/e2e_dd_compat.py` | Load the prepared Detailer graph through the frontend file input, click Run | Successful SEGSDetailer inference with DifferentialDiffusion, output image loaded in the UI, non-degenerate image, changed pixels by mean/std comparison |

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
```

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
