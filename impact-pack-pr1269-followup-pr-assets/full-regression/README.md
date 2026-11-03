# Full repository regression

This is the completed pre-cleanup run. The [current regression record](../test-review/README.md) supersedes the affected unit/Control results and the three legacy diagnostic failures; unchanged production results remain applicable.

The full checked-in test inventory was executed against the final combined candidate: Main `082523b` + #1269 `609e411` + the follow-up, candidate tree `eaacc87ea003683ebcc38055235d4c8a3bf4a195`.

**570 unique pytest cases passed; 2 existing Processor Bypass cases remain xfail. No newly introduced product failure was found in the exercised paths.** Three historical wildcard diagnostics fail identically on Main + #1269. They remain failures in this record and are not counted as passed.

The [machine-readable report](validation.json), [entry-point inventory](test-inventory.md), [original source manifest](../source-before-test-review.json), [additional fixture provenance](setup.json) and [baseline diagnostic comparison](diagnostics.md) provide the audit trail. The inventory covers all 63 tracked Python, shell and JavaScript files under `tests`; support modules and manual restart/inspection wrappers are identified separately.

## Assertion-based suites

The rows below include the previously completed affected suites and the additional full repository run. Configuration reruns are excluded from the total.

| Suite | Passed | Existing xfail | JUnit |
|---|---:|---:|---|
| standalone | 205 | 0 | [report](../merged-unit.xml) |
| browser harness | 3 | 0 | [report](../browser-harness.xml) |
| classic cache controls | 97 | 2 | [report](../classic/cache-controls.xml) |
| catalog | 14 | 0 | [report](../catalog/catalog.xml) |
| native tutorials | 8 | 0 | [report](../catalog/tutorial-native.xml) |
| switches | 55 | 0 | [report](../catalog/switches.xml) |
| sam mask unit | 109 | 0 | [report](../full-regression/sam-mask-unit.xml) |
| animatediff assembly | 44 | 0 | [report](../full-regression/animatediff-unit.xml) |
| mask blur | 9 | 0 | [report](../full-regression/mask-blur.xml) |
| mask native | 4 | 0 | [report](../full-regression/cpu/mask-native.xml) |
| legacy sam | 6 | 0 | [report](../full-regression/cpu/legacy-sam.xml) |
| sam native | 10 | 0 | [report](../full-regression/gpu/sam.xml) |
| sam api | 3 | 0 | [report](../full-regression/gpu/sam-api.xml) |
| lora source | 2 | 0 | [report](../full-regression/lora/lora-source.xml) |
| lbw parity | 1 | 0 | [report](../full-regression/lbw-candidate/lbw-parity.xml) |

Classic isolation rerun: 18 passed. LRU64 selection: 32 passed. These repetitions are retained as separate configuration/isolation evidence and are not added to the 570 unique-case total. The unchanged trusted catalog covers 198 node types and 719 widgets, including positional and named native restoration.

## Additional execution and integration evidence

| Check | Evidence |
|---|---|
| JavaScript | [SAM: 3 passed](legacy-sam-js-rerun.log), [LoRA source suite](lora-source-js.log) |
| Native AnimateDiff tutorials | [Four original PNGs, both restoration modes](cpu/animatediff-native.log) |
| Wildcard UI/API, normal mode | [UI](cpu/wildcard-ui.log), [API](cpu/encoding-api.log) |
| Wildcard UI/API, actual on-demand mode | [UI](ondemand/wildcard-ui.log), [API](ondemand/encoding-api.log), [server mode confirmation](ondemand/server.log) |
| Real hook/cache/list mapping | [20 groups](../merged-integration.json); Encode uses a CLIP stand-in in this matrix |
| Real cache and LoRA tensors | [5 cache groups](../cache-tensors.json), [LoRA tensors](lora-tensors.log), [15 validation cases](lora-validation.log), [3 tutorial tensor groups](tutorial-tensors.json) |
| Real SD1.5 sampling with LoRA source modes and BREAK | [Browser result and exact pixel comparison](inference/lora-inference.log) |
| Original LBW tutorial image parity | [Fresh Main+#1269 reference capture](lbw-baseline/reference/reference.png), [candidate JUnit](lbw-candidate/lbw-parity.xml) |
| Differential Diffusion Detailer | [Real detected-face edit and pixel statistics](detailer-final/detailer.log) |
| Conditioning | [Exact parity with core nodes and native reload](gpu/conditioning.log) |
| Switch tutorials | [Four original tutorials, execution and reference pixels](gpu-retry/switch-tutorials.log) |
| Regional sampling | [Five real executions](gpu/regional.log), [localized effect of all three prompts](gpu/regional/regional-effects.json) |
| PR-specific constant/LoRA cache behavior | [Five observations](../constant-lora/result.json), [actual core cache-key comparisons](../constant-lora/node-cache-keys.json) |

## Historical diagnostic scripts

Ten of the eleven shell entry points exit 0; `test_progressive_ondemand.sh` exits 1. Of the three standalone Python loading diagnostics, one exits 0 and two exit 1. Those three failures reproduce on the Main+#1269 baseline. The separate final consistency script records seven matching result pairs. See [diagnostics.md](diagnostics.md) for the exact failures and the limits of these scripts; an exit code alone is not treated as proof that every printed diagnostic passed.

## Environment corrections and coverage boundaries

- The first SAM JavaScript invocation omitted Node's `--experimental-vm-modules`; the correctly invoked suite passes all three cases. The initial log is retained as an invocation error.
- The first Detailer and Switch tutorial attempts could not load the referenced images in the isolated input directory. Installed images were copied into the retry runtime; the Switch tests then executed successfully. Detailer additionally needed its initial Templates dialog dismissed with Escape; the accepted execution is in `detailer-final/`. Initial failures and accepted retry results are both retained.
- The general wildcard UI and Detailer scripts required an Escape keypress to dismiss the current frontend's Templates dialog. That test-copy-only change is recorded in [environment-adaptation.patch](environment-adaptation.patch); no assertions or production files changed.
- Legacy shell scripts receive isolated paths, ports and owned-server stop commands, recorded per case. The unadapted final-script invocation was terminated and excluded. Its accepted run uses a separate runtime and free port.
- AnimateDiff coverage includes 44 assembly tests with controlled VAE/sampling and eight native restoration cases. It does not claim actual video inference. SAM unit tests use controlled inputs/predictions; separate SAM browser tests use the installed model.
- The final consistency script's 1 MB threshold does not establish on-demand mode with its fixtures. Actual mode-specific UI/API validation uses 0.0001 MB and verifies the running mode in the server log.

The [cleanup record](cleanup.json) confirms all owned servers stopped and all 21 recorded job ports closed. All production files match the final source manifest across the tested copies. The PR patch and trusted widget fixtures are unchanged by this full run. No models were downloaded. Unrelated screenshots remain in raw test artifacts and are not inserted into the PR body.

## Reproduction

The exact commands, environment and server options are in `cpu.json`, `gpu.json`, `gpu-retry.json`, `detailer-final.json`, `lora.json`, `ondemand.json`, `inference.json`, `lbw-baseline.json` and `lbw-candidate.json`. Each server job verifies ownership using `/system_stats` and stops its own server in `finally`. Run `/home/rho/venv/bin/python run_server_suite.py JOB.json` after preparing the recorded source/runtime links and installed fixtures. Run `run_units.py` for the unit, JavaScript and tensor checks; use the corrected JavaScript invocation above.

`run_legacy.py`, `run_legacy_python.py`, `run_legacy_baseline.py`, `run_legacy_python_baseline.py` and `run_followups.py final` record the isolated historical diagnostics. The final-script case is excluded from the Python bootstrap; `run_followups.py final` owns its isolated runtime. `summarize_full.py` validates accepted results and source integrity; `write_inventory.py` maps all entry points.
