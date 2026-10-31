# Wildcard System Test Suite

Comprehensive test suite for ComfyUI Impact Pack wildcard system.

## Standalone regression tests

From the `tests/` directory, run `python3 -m pytest`.
Install dependencies with `python3 -m pip install -r tests/requirements.txt`
from the repository root. This runs configuration, populate, and strict
encoding checks without starting ComfyUI. Seed regressions cover negative seeds,
existing outputs, and shared RNG isolation during success, failure, and concurrent
execution. The configuration keeps pytest from
importing the extension's server-initializing `__init__.py`.
Circular-reference tests use `pytest-timeout` to stop after five seconds.

From the Impact Pack root, select the files explicitly:
`python3 -m pytest tests/wildcards/test_config.py tests/wildcards/test_populate_regression.py tests/wildcards/test_seed_regression.py tests/wildcards/test_wildcard_encoding_strict.py`.

Both suites also run directly with `python3 tests/wildcards/test_populate_regression.py`
and `python3 tests/wildcards/test_wildcard_encoding_strict.py`; both delegate
to pytest with the same configuration.
The three TXT encoding fixtures in `tests/wildcards/samples/` are included
in the repository. The older server-dependent suites below are run separately.

## Chrome UI E2E

`python3 tests/e2e_wildcards.py` creates and connects nodes, edits widgets,
clicks Run, and checks the displayed results. `python3 tests/e2e_dd_compat.py`
imports a prepared Detailer workflow through the UI and clicks Run to verify
inference and image previews. Both default to headless Google Chrome; add
`--headed` to watch the browser on a machine with a display.

See [E2E setup and artifacts](../docs/E2E_TEST_STRATEGY.md). These scripts need
a running test server and are separate from the curl-based API integration tests.
`python3 -m pytest tests/test_e2e_legacy_sam.py -v` checks the legacy SAM menu
and Clipspace button without importing the removed `ClipspaceDialog` module.
Set `IMPACT_E2E_SAM_IMAGE` and install the configured SAM model to include real
detection, repeated saving/pasting, RGB preservation, and native workflow reload
in both widget restoration modes, starting from input and nested output images.
The suite also checks keyboard shortcuts, cancellation, resizing, changing the
image, and repeated invalid-path rejection. Run
`node --experimental-vm-modules --test tests/legacy_sam.test.mjs` for current-image
resizing and saved image references with controlled browser objects, including
global and fallback UI exports.
Older frontends without named widget restoration skip the two named detection
cases; their other native reloads use positional restoration.
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
