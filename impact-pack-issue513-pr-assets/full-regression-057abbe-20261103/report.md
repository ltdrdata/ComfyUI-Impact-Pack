# Full regression — 057abbe

PR commit `057abbe915048910ab2e656f991af6a029747323` against its Main base `56a34b9c439470d07440f9104204a084e5fdb38c`. The tracked worktree is clean; production code was unchanged during this run. All 173 frozen source files and the two trusted widget fixtures match their starting hashes.

All **29 maintained suites passed**: Node **62**, standalone Python **373**, server/browser pytest **289**. Additional real tensor, model inference, native workflow and HTTP scripts passed. The inventory contains **58 executed entrypoints** and three administration/discovery helpers excluded with reasons.

| Server/browser suite | Tests | Failures | Errors | Skips |
| --- | ---: | ---: | ---: | ---: |
| catalog | 14 | 0 | 0 | 0 |
| wildcard-classic | 86 | 0 | 0 | 0 |
| wildcard-lru | 78 | 0 | 0 | 0 |
| connection-types | 30 | 0 | 0 | 0 |
| switches | 55 | 0 | 0 | 0 |
| mask-browser | 4 | 0 | 0 | 0 |
| sam-browser-api | 19 | 0 | 0 | 0 |
| lora-source-loading | 2 | 0 | 0 | 0 |
| lbw-tutorial-inference | 1 | 0 | 0 | 0 |

Exact [commands/results](summary.json), [entrypoint inventory](inventory.json), [source provenance](source-provenance.json) and [source hashes](source-manifest.json) are retained. The full catalog preserves 198 records, widget names/order, positional and named saved values through native Export/load. No trusted fixture was regenerated.

## Real model and tensor verification

Installed SD1.5/SDXL models and LoRAs were used without downloads. Conditioning and MODEL/CLIP patch tensors were compared with independent real core/Inspire references. LoRA selection/loading, Detailer, conditioning, Switch tutorials and Regional sampler scripts passed. Regional missing mask pixels were restored from the original companion tutorial PNG; [media provenance](regional-media-provenance.json) records the exact sources and existing widget positions.

The original LBW workflow was rendered freshly on Main with the same deterministic CUDA FP32 settings. The candidate output matches its RGB bytes exactly, then reuses fixed downstream caches. [Main reference](main-lbw-reference/reference.json) · [Main output](main-lbw-reference/output.png) · [Candidate output](lbw-tutorial-inference/test_lbw_tutorial_inference/output.png).

## Legacy diagnostics — failures retained

The full repository run has known legacy failures; it is not reported as all green. Three failing Python assertions and three failing shell/smoke checks are reproduced on the current PR base. Five diagnostic entrypoints are affected. The two exit-zero scripts are still treated as failed checks based on their output.

| Diagnostic | Candidate exit | Main exit | Same failure |
| --- | ---: | ---: | --- |
| legacy-python-progressive_loading | 1 | 1 | True |
| legacy-python-lazy_load_verification | 1 | 1 | True |
| legacy-config_quotes | 0 | 0 | True |
| legacy-progressive_ondemand | 1 | 1 | True |
| legacy-api-candidate-run_quick_test | 0 | 0 | True |

- `test_progressive_loading.py` expects all five available keys from a loaded-key list and gets zero on both revisions.
- `test_lazy_load_verification.py` expects on-demand mode in two cases but observes full-cache mode on both revisions. Its config import and the runtime config use separate module identities; its cached dictionary reference and return-value tests are independent legacy quality issues.
- `test_config_quotes.sh` reports an unresolved `test_nesting_level1` fixture, then prints success and exits zero.
- `test_progressive_ondemand.sh` expects the already-loaded YAML key to grow its count; both revisions report 6 → 6.
- `run_quick_test.sh` expects an unchanged list while on-demand entries load; both revisions report 3 → 6 and print a failure while exiting zero. Its optional diagnostic log path also does not exist.

[Legacy comparisons](legacy-diagnostics.json) · [Additional API comparisons](legacy-api-diagnostics.json) · [Path/port/PID adaptations](legacy-adaptations.json). Test assertions were not relaxed. The production comparison differs only in `js/impact-pack.js`; all backend files are identical. Corpus content, runtime directories, model weights and browser trace archives are excluded from publication.

## Environment and scope

Node 24.12.0, Python 3.12.3, frontend 1.57.6, Google Chrome 156.0.8078.79. SAM uses CPU on the server and independent predictor, cache-none, and only Impact/Subpack for the legacy global-resource assertion. All owned test servers have stopped.

Optional Styler/BNK graphs are checked for native loading and Impact tensor boundaries; their missing-plugin end-to-end inference is not claimed. AnimateDiff covers native workflows and tensor assembly, not full video diffusion. Independent GlobalSeed and native promoted populate/before projection bookkeeping retain the PR's documented boundaries.
