# Fresh full regression of pushed PR head

**575 unique pytest cases passed; 2 existing Processor Bypass cases remain xfail.** All accepted results in this directory were executed in this run from head `63d28f838c8040ade1ef122f8cd8985aef542b34`. No prior test result is reused in this total. Additional LRU-64 and loader-first runs repeat existing cases and are excluded from the unique count.

This supersedes the earlier cumulative 576-case report. That report tested a synthetic Main combination containing one extra path test (`test_package_paths_have_no_parent_components`). This run tests the actual pushed branch with 210 configured cases, so its unique total is 575. It does not validate a future rebase onto current Main.

The [machine-readable acceptance report](validation.json), [complete entry-point inventory](test-inventory.md), [source manifest](source.json), [environment](environment.json), and [owned-server cleanup](cleanup.json) identify the tested inputs and outcomes. The inventory classifies all 63 tracked Python, shell and JavaScript files, including support modules and manual utilities.

## Fresh pytest results

| Suite | Passed | Existing xfail | Report |
|---|---:|---:|---|
| configured units | 210 | 0 | [JUnit](configured-unit.xml) |
| browser harness | 3 | 0 | [JUnit](browser-harness.xml) |
| classic cache controls | 97 | 2 | [JUnit](classic/cache-controls.xml) |
| catalog | 14 | 0 | [JUnit](catalog/catalog.xml) |
| native tutorials | 8 | 0 | [JUnit](catalog/tutorial-native.xml) |
| switches | 55 | 0 | [JUnit](catalog/switches.xml) |
| sam mask units | 109 | 0 | [JUnit](sam-mask-unit.xml) |
| animatediff assembly | 44 | 0 | [JUnit](animatediff-unit.xml) |
| mask blur | 9 | 0 | [JUnit](mask-blur.xml) |
| mask native | 4 | 0 | [JUnit](cpu/mask-native.xml) |
| legacy sam | 6 | 0 | [JUnit](cpu/legacy-sam.xml) |
| sam native | 10 | 0 | [JUnit](gpu/sam.xml) |
| sam api | 3 | 0 | [JUnit](gpu/sam-api.xml) |
| lora source | 2 | 0 | [JUnit](lora/lora-source.xml) |
| lbw parity | 1 | 0 | [JUnit](lbw-candidate/lbw-parity.xml) |

The native catalog checks all 198 node types and 719 widget entries in both positional and named restoration modes. Widget names, order, values and source fixtures remain unchanged. The [source integrity records](validation.json) verify every candidate production file across the tested copies.

Separate repetitions: [32 LRU-64 cases](verified-lru/cache.xml) and [210 configured cases with loader tests first](unit-reordered.xml). The two populated-only Processor Bypass cases also reproduce against [dependency-only #1269 production](baseline-controls/bypass.xml), with the current test assertions supplied by a [test overlay](baseline-test-overlay.json). They are not counted as passes. Encode and connected-first-input Bypass/reactivation cases are part of the passing Classic suite.

## Actual execution and integration

| Check | Fresh evidence |
|---|---|
| Mixed-list bug, before and after | [#1269 observations](baseline-mixed-list/observations.json), [follow-up observations](mixed-list/observations.json). Baseline seed 5 incorrectly executes seed 2 and returns red; the follow-up preserves seed 5 and returns blue. |
| Actual LoRA/LBW cache behavior | [Five observations](constant-lora/result.json), [four actual cache-key transitions](constant-lora/node-cache-keys.json). The requested wildcard seed, source/populated text, execution seed, cached/executing nodes and pixel hashes are compared together. |
| Fresh browser captures for the same sampling scenario | [Unchanged text / request 732](constant-lora/same.png), [edited text / request 733](constant-lora/changed.png), [732 history](constant-lora/same-history.json), [733 history](constant-lora/changed-history.json), [732 events](constant-lora/same-events.json), [733 events](constant-lora/changed-events.json). The displayed widget holds the next incremented seed. |
| Real hook, cache and mapped lists | [20 integration groups](cache-hook.json); Encode uses a CLIP stand-in in this matrix. |
| Installed CLIP/LoRA tensors | [5 cache groups](cache-tensors.json), [LoRA patch/tensor checks](lora-tensors.log), [15 validation cases](lora-validation.log), [3 tutorial tensor groups](tutorial-tensors.json). |
| Real LoRA source sampling | [All modes, BREAK, native reload and exact core-node pixel parity](inference-corrected/lora-inference.log), [per-case hashes](inference-corrected/lora-inference/results.json). |
| Original LBW tutorial parity | [Fresh #1269 reference capture](lbw-baseline/reference/reference.png), [candidate parity JUnit](lbw-candidate/lbw-parity.xml). |
| Real Detailer and conditioning | [Detected-face edit](gpu/detailer.log), [exact core conditioning parity and native reload](gpu/conditioning.log). |
| Original Switch and regional tutorials | [Four Switch tutorials](gpu/switch-tutorials.log), [five regional executions](gpu/regional.log), [effect of each regional prompt](gpu/regional/regional-effects.json). |
| Native AnimateDiff workflows | [Four original PNGs in both restoration modes](cpu/animatediff-native.log); assembly/native coverage, without a video inference claim. |
| Wildcard UI/API | Full cache: [UI](wildcard-full/wildcard-ui.log), [API](wildcard-full/encoding-api.log). Actual on-demand mode: [UI](ondemand-corrected/wildcard-ui.log), [API](ondemand-corrected/encoding-api.log). |
| Two-server API parity | [Seven result pairs](pair-ondemand/pair.log), [verified actual modes](api-result.json). |
| JavaScript | [Legacy SAM: 3 passed](legacy-sam-js.log), [LoRA source suite](lora-source-js.log). |
| Historical loader entry points | [11 shell scripts exit 0](legacy-with-data/results.json), [remaining standalone Python diagnostic exits 0](legacy-python-with-data/result.json). Their logs are mapped in the inventory; exit codes are not additional asserted-case counts. |
| Failure detection | [10 API faults/controls and 4 Bypass guard probes](failure-detection.json), separate from real-server execution. |
| Relevant static checks | [Ruff comparison](ruff.json): no new F401/F841/F811/F821 violations; inherited impact_pack.py import warning remains. |

## Environment corrections and reproduction

The initial API pair launcher had a duplicated directory prefix and failed before invoking the test. Its [full-server log](invocation-errors/full/pair.log) and [on-demand log](invocation-errors/pair-ondemand/pair.log) are retained. Corrected launches verify actual modes and all seven matching results.

The first LoRA source inference run used generic wildcard data instead of the test's explicitly required `tests/wildcards/model_samples` directory. Its [failed assertion](inference/lora-inference.log) is retained. After correcting only the runtime configuration, the [same unmodified test](inference-corrected/lora-inference.log) passes. The LoRA source UI test uses its specified `tests/wildcards/lora_samples` directory. [Runner corrections](runner-corrections.json) identify these setup errors; no product or assertion change was made to resolve them.

The first generic wildcard UI/API launches also used the general wildcard directory instead of their required `tests/wildcards/samples`. The original [full-cache UI failure](cpu/wildcard-ui.log), [full-cache API failure](cpu/encoding-api.log), [on-demand UI failure](ondemand/wildcard-ui.log) and [on-demand API failure](ondemand/encoding-api.log) are retained. After correcting only that fixture path, both modes pass the same tests in `wildcard-full/` and `ondemand-corrected/`.

Three isolated standalone test copies dismiss the current frontend's initial Templates dialog with Escape: wildcard UI in full/on-demand mode and the Detailer script. [Exact source hashes and patches](source.json) record those test-only adaptations. Legacy shell copies receive isolated paths, ports and an owned-process stop command, recorded beside each script's log. Native workflow/snapshot fixtures are not modified.

The initial browser-harness JUnit destination contained a literal `{root}`. The completed result was moved unchanged to [browser-harness.xml](browser-harness.xml); the runner destination was corrected for reproduction.

The job JSON files retain exact commands, environment, cache mode, source path and server options: [Classic](classic.json), [LRU](verified-lru.json), [catalog/tutorial/switch](catalog.json), [CPU/native](cpu.json), [GPU](gpu.json), [LoRA source](lora.json), [LoRA inference](inference-corrected.json), [full-cache wildcard](wildcard-full.json), [on-demand](ondemand-corrected.json), [constant sampling](constant.json), [LBW baseline](lbw-baseline.json), [LBW candidate](lbw-candidate.json). Installed models and fixture files are used; no models were downloaded.

After preparing those source/runtime paths from the recorded revision and fixtures, run `python run_server_suite.py JOB.json`. Each job verifies its own server through `/system_stats` and stops it in `finally`. [Unit runner](run_units.py), [loader-first runner](run_unit_order.py), [legacy shell runner](run_legacy.py), [legacy Python runner](run_legacy_python.py) and [acceptance checks](summarize.py) retain the other commands. The configured suite runs from the archived source’s `tests` directory so pytest resolves its configured paths correctly.

The production branch remains unchanged. #1269 is still open; replay the two follow-up commits after it lands and validate the resulting Main combination before making the draft ready.
