# Wildcard System Test Suite

Comprehensive test suite for ComfyUI Impact Pack wildcard system.

## Standalone regression tests

From the `tests/` directory, run `python3 -m pytest`.
Install dependencies with `python3 -m pip install -r tests/requirements.txt`
from the repository root. This runs configuration, populate, and strict
encoding and LoRA wildcard source checks without starting ComfyUI. Seed regressions cover negative seeds,
existing outputs, and shared RNG isolation during success, failure, and concurrent
execution. The configuration keeps pytest from
importing the extension's server-initializing `__init__.py`.
Circular-reference tests use `pytest-timeout` to stop after five seconds.

SAM mask regressions use real ComfyUI/Impact code with controlled predictions.
Use ComfyUI's Python dependencies and set `IMPACT_COMFY_ROOT` explicitly; no
weights, GPU or server are needed. Run separately from stubbed wildcard tests:

```bash
IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_sam_masks.py tests/test_mask_to_segs.py -v
```

They check detection grouping, threshold/fallback rules, combined pixels,
empty predictions, hint layouts, downstream masks and device cleanup.

From the Impact Pack root, select the files explicitly:
`python3 -m pytest tests/wildcards/test_config.py tests/wildcards/test_populate_regression.py tests/wildcards/test_seed_regression.py tests/wildcards/test_wildcard_encoding_strict.py tests/wildcards/test_lora_wildcard_source.py`.

Both suites also run directly with `python3 tests/wildcards/test_populate_regression.py`
and `python3 tests/wildcards/test_wildcard_encoding_strict.py`; both delegate
to pytest with the same configuration.
The three TXT encoding fixtures in `tests/wildcards/samples/` are included
in the repository. The older server-dependent suites below are run separately.

## Mask-to-SEGS regression tests

Run separately from the stubbed wildcard tests, using a real ComfyUI installation
on CPU without models. Use its Python environment with the Impact Pack dependencies
and `tests/requirements.txt` installed; `IMPACT_COMFY_ROOT` points to the directory
containing ComfyUI's `nodes.py` and `comfy/`:

```bash
IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_mask_to_segs.py -v
```

Checks cover finite out-of-range values before contour quantization, bicubic
overshoot, adjacent values at the quantization threshold, mixed ranges in every
batch, input immutability, and existing combined behavior.
Tutorial-derived cases also cover holes and nested regions, bbox/contour fill,
expanded edge crops, `drop_size` boundaries, disconnected combined regions and
the mask-list consumer's image coordinates.
Only unused server registration is substituted, using pytest's scoped
monkeypatch cleanup. `tests/test_e2e_mask_to_segs.py` separately executes the bicubic
workflow in Chrome, compares preview pixels with an explicit core-node clamp,
and checks native saved widget names/order/values in both restoration modes.
After reload it changes the resize factor through the UI, verifies the changed
prompt and uncached execution path, and checks new filenames, dimensions and pixels.
See [E2E setup and commands](../docs/E2E_TEST_STRATEGY.md) for the isolated CPU
server, Playwright/Chrome requirements, and catalog sample configuration.

`tests/test_e2e_mask_tutorials.py` separately loads the original tutorial PNG/JSON
files, checks their saved controls, exports and reloads them under both restoration
modes. Set `IMPACT_TUTORIALS_DIR` to the `ComfyUI-extension-tutorials` checkout.
This loading test needs no referenced media or models and performs no inference.
Missing required environment variables fail with setup guidance; they do not
silently skip an explicitly requested regression suite.

## AnimateDiff tensor assembly

Run separately from the wildcard suite, whose dependency stubs conflict with
the real ComfyUI imports used here:

```bash
COMFYUI_ROOT=/path/to/ComfyUI python -m pytest tests/test_animatediff_assembly.py -v
```

Checks preserve tensors, pixels, dtype, frame order, hooks and tutorial node
paths while bounding mask/latent concatenation allocations before sampling.
Learned-model operations and HTTP registration are controlled; no model weights,
GPU or server are required, and full AnimateDiff inference is not tested.

Against an isolated server, run
`IMPACT_TUTORIALS_DIR=/path/to/ComfyUI-extension-tutorials python tests/e2e_animatediff_tutorials.py --base-url http://127.0.0.1:18188 --artifacts /tmp/impact-animatediff-tutorials`
for four original PNGs and native JSON export/reload in both widget restoration
modes. Also run the catalog suite with its trusted fixtures and Full Cache setup
from [E2E setup and artifacts](../docs/E2E_TEST_STRATEGY.md); these browser checks
validate widget names/order/values, not inference.

## Chrome UI E2E

`python3 tests/e2e_wildcards.py` creates and connects nodes, edits widgets,
clicks Run, and checks the displayed results. `python3 tests/e2e_dd_compat.py`
imports a prepared Detailer workflow through the UI and clicks Run to verify
inference and image previews. Both default to headless Google Chrome; add
`--headed` to watch the browser on a machine with a display.

See [E2E setup and artifacts](../docs/E2E_TEST_STRATEGY.md). These scripts need
a running test server and are separate from the curl-based API integration tests.

Wildcard cache and original-PNG cases check seeded selection, cache reuse, linked
providers and native widget restoration in both modes:

```bash
python3 -m pytest tests/test_e2e_wildcard_cache.py -v
python3 -m pytest tests/test_e2e_wildcard_tutorials.py -k native -v
```

Run CPU hook/cache checks and real SD1.5 conditioning/LoRA comparisons separately:

```bash
python3 tests/integration_wildcard_cache.py --comfy-root /path/to/ComfyUI
python3 tests/integration_wildcard_cache_tensors.py --comfy-root /path/to/ComfyUI
```

The [E2E guide](../docs/E2E_TEST_STRATEGY.md) lists model overrides, pinned tutorial
assets, the Inspire/SDXL tensor script and the original LBW sampling comparison.

SAM execution and original tutorial suites check real ViT-B masks, downstream
consumers and native widget restoration in both modes. The detector PNG executes;
the Mask Pointer JSON covers loading only because its painted mask is absent.
See the linked E2E guide for assets, environment variables and commands.

`python3 -m pytest tests/test_e2e_legacy_sam.py -v` checks the legacy SAM menu
and Clipspace button without importing the removed `ClipspaceDialog` module.
Set `IMPACT_E2E_SAM_IMAGE` and install the configured SAM model to include real
detection, repeated saving/pasting, RGB preservation, and native workflow reload
in both widget restoration modes, starting from input and nested output images.
The output case edits the second image of a batch and preserves the other image
and the Clipspace selection after saving.
Menu and keyboard/cancel/resize checks use a controlled preparation response;
they do not load SAM. Detection/save cases use real preparation and inference.
Run `python3 -m pytest tests/test_sam_api.py -v` against the current backend
for repeated invalid-path rejection between valid preparation requests. It uses
HTTP without launching Chrome and requires the configured SAM model. Run
`node --experimental-vm-modules --test tests/legacy_sam.test.mjs` for current-image
resizing and missing image metadata with controlled browser objects, including
global and fallback UI exports.
Older frontends without named widget restoration skip the two named detection
cases; their positional save/reload cases still run.
With Playwright and Chrome installed, `python3 -m pytest tests/test_e2e_ui.py`
also checks failure cleanup and JavaScript errors during artifact saving using
intercepted browser requests, without a server.
With a running test server, `IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 python3 -m pytest tests/test_e2e_switches.py`
checks Switch/Combine import, browser reload, routing, and editing through Chrome,
including non-default selection, legacy names, image/mask lists and batches, and
SEGS concatenation. Mask batch checks compare every output pixel with expected
patterns, including resizing, center cropping, batch order, and one-pixel dimensions.
`python3 tests/e2e_conditionings.py` compares conditioning results with core
ComfyUI and checks regional-prompt combination, using the Detailer checkpoint.
`python3 tests/e2e_tutorials.py` imports the original Switch tutorial PNGs,
changes every selection, and compares final output pixels with direct core-node
results. Set `IMPACT_TUTORIALS_DIR` to the tutorial checkout; see the E2E guide
for its input images and OpenPose requirements.
Image checks verify that browser previews contain the current server output,
including rejecting blocked requests and stale pixels served at a new URL.
`python3 -m pytest tests/test_e2e_preview_bridge.py -v` exercises Preview Bridge
through Chrome's Vue MaskEditor: paint/mask editing and native reload, Clipspace feedback, downstream
compositing, latent mask propagation, blocking and restoration. Its default
cases need no models. Set `IMPACT_E2E_SAM_IMAGE` to include Interactive SAM with
the server's configured model. See the E2E guide for setup and coverage.
`node --experimental-vm-modules --test tests/preview_bridge.test.mjs` checks
Vue store-only updates, awaited registration, retry after failure, and legacy
workflow references without starting a server.
`python3 tests/e2e_regional.py` checks exact color masks and the localized effect
of all three regional prompts in the restored regional img2img tutorial.
Set `IMPACT_REGIONAL_WORKFLOW` to that PNG or JSON; see the E2E guide for assets
and dependencies. This checks regional processing, not artistic quality.

`tests/test_e2e_widget_catalog.py` loads `workflows/all-impact-nodes.json` in
Chrome and compares each node's live widget names, order, and values with
`workflows/all-impact-nodes.widgets.json`. It checks both positional and named
restoration, including nodes without widgets, aliases, and deprecated nodes.
The catalog is created by UI node search and native Export; see the E2E guide
for recording instructions. It tests loading, not execution of every node.
For each new registered node, first verify the existing trusted catalog, then
extend both fixtures through the recorder and rerun the E2E. Existing snapshot
records must remain unchanged; see [the repository rules](../AGENTS.md).

`node --test tests/lora_wildcard_source.test.mjs` checks dependent dropdown
request ordering, saved selections, query encoding and isolation from existing
nodes, including returning to a category while its older response is pending.
`tests/test_e2e_lora_wildcard_source.py` verifies the same saved-value
contract through Chrome search, editing, native Export and file import in both
restoration modes. Configure its isolated server with
`tests/wildcards/lora_samples`; see the E2E guide.

`python3 tests/e2e_lora_wildcard_inference.py --artifacts /tmp/impact-lora-inference`
checks all four modes and native workflow reload against core LoRA/CLIP/sampler
nodes using real SD1.5 models. Configure the isolated server with
`tests/wildcards/model_samples` and the models listed in the script docstring.
It compares every output pixel, output text, filename, list length and entry order;
browser previews must also match server images. A saved duplicate label must
still select its original row after syntax expansion and native reload in both
restoration modes. Mixed-syntax random/sequential/batch workflows also retain
their original candidates, and leading punctuation is compared with core-node
references. A legacy `pose` category resolves `poses.txt` and survives native
reload even when the alias is absent from current dropdown options. This is
separate from the model-free catalog and delayed-response tests.
Use the deterministic FP32 server options in the E2E guide for pixel equality.
`python3 tests/integration_lora_wildcard_tensors.py --comfy-root /path/to/ComfyUI`
separately compares every conditioning tensor and MODEL/CLIP LoRA patch tensor
and weight with core nodes under ComfyUI's default model-loading settings.
It also checks populated/empty-category punctuation and BREAK conditioning and verifies
that the input models retain no patches. Only unused HTTP route registration
is substituted; model operations are real.

`python3 tests/integration_lora_wildcard_validation.py --comfy-root /path/to/ComfyUI`
uses the real ComfyUI input validator on CPU without loading models. Dynamic
category aliases and saved choices remain accepted; invalid literal modes and numbers,
invalid numeric strings and incompatible links are rejected with the expected
input-specific error. Only the MODEL/CLIP input provider and unused HTTP route
registration are substituted. Run separately from the stubbed unit tests.

## Test Suites

### test_encoding.sh (15 tests)
**Purpose**: UTF-8 multi-language encoding validation
**Server**: URL argument; defaults to `http://127.0.0.1:8198`.
Configure an isolated server with `tests/wildcards/samples` before running.
The script never changes configuration or starts/stops server processes.
**Coverage**:
- Korean Hangul characters
- Emoji support
- Chinese characters
- Arabic RTL text
- Mathematical and currency symbols
- Mixed multi-language content
- UTF-8 in dynamic prompts, quantifiers, multi-select

### test_error_handling.sh (10 tests)
**Purpose**: Graceful error handling verification
**Port**: 8197
**Coverage**:
- Non-existent wildcards
- Circular reference detection (max 100 iterations)
- Malformed syntax
- Deep nesting without crashes
- Multiple circular references

### test_edge_cases.sh (20 tests)
**Purpose**: Edge case and boundary condition validation
**Port**: 8196
**Coverage**:
- Empty lines and whitespace filtering
- Very long lines (>1000 characters)
- Special characters preservation
- Case-insensitive matching
- Comment line filtering
- Pattern matching (__*/name__)
- Quantifiers (N#__wildcard__)
- Complex syntax combinations

### test_deep_nesting.sh (17 tests)
**Purpose**: Transitive wildcard expansion and depth-agnostic pattern matching
**Port**: 8194
**Coverage**:
- 7-level transitive expansion (directory depth + file references)
- All depth levels (1-7) individually
- Mixed depth combinations
- Nesting with quantifiers and multi-select
- Nesting with weighted selection
- Depth-agnostic pattern matching (`__*/name__`)
- Complex multi-wildcard prompts

### test_ondemand_loading.sh (8 tests)
**Purpose**: Progressive on-demand wildcard loading
**Port**: 8193
**Coverage**:
- Small cache (1MB) - on-demand enabled
- Moderate cache (10MB) - progressive loading
- Large cache (100MB) - eager loading
- Aggressive lazy loading (0.5MB)
- Balanced mode (50MB default)
- On-demand with deep nesting
- On-demand with multiple wildcards
- Cache boundary testing

### test_config_quotes.sh (5 tests)
**Purpose**: Configuration path handling validation
**Port**: 8192
**Coverage**:
- Unquoted paths
- Double-quoted paths
- Single-quoted paths
- Paths with spaces
- Mixed quote scenarios

### test_dynamic_prompts_full.sh (11 tests)
**Purpose**: Comprehensive dynamic prompt feature validation with statistical analysis
**Port**: 8188
**Coverage**:
- **Multiselect** (4 tests): 2-item, 3-item, single-item, max-item with separator validation
- **Weighted Selection** (5 tests): 10:1 ratio, equal weights, extreme bias, multi-level weights, default mixing
- **Basic Selection** (2 tests): Simple random, nested selection
- Statistical distribution verification (100+ iterations per test)
- Duplicate detection and item count validation
- Separator correctness validation

## Quick Start

```bash
# Run against the isolated server described in docs/E2E_TEST_STRATEGY.md
bash test_encoding.sh http://127.0.0.1:18188

# Older scripts below require a disposable test copy
bash test_error_handling.sh
bash test_edge_cases.sh
bash test_deep_nesting.sh
bash test_ondemand_loading.sh
bash test_config_quotes.sh
bash test_dynamic_prompts_full.sh
```

## Test Infrastructure

- **Encoding suite**: Uses an already-running isolated server; no configuration or process cleanup
- **Other legacy API suites**: Manage configuration and server lifecycle themselves; use a disposable test copy
- **Logging**: Detailed logs in `/tmp/*_test.log`

## Test Samples

Located in `wildcards/samples/`:
- `아름다운색.txt` - Korean UTF-8 test with 12 symbolic colors
- `test_encoding_*.txt` - UTF-8 encoding test files
- `test_edge_*.txt` - Edge case test files
- `test_error_*.txt` - Error handling test files
- `test_nesting_*.txt` - Nesting test files (7 levels)
- `patterns/` - Subdirectory for pattern matching tests

## Documentation

- [Wildcard System PRD](../docs/wildcards/WILDCARD_SYSTEM_PRD.md)
- [System Design](../docs/wildcards/WILDCARD_SYSTEM_DESIGN.md)
- [Testing Guide](../docs/wildcards/WILDCARD_TESTING_GUIDE.md)
