# Full regression diagnostics

This run retains unsuccessful attempts as well as corrected runs. Intentional RGR
Red phases are not counted as regressions. Production code is unchanged from
`6783bc5` throughout; only the twelve new edge cases and test documentation were
added. The source and trusted fixtures are hashed in the final report.

## Environment corrections

- The first SAM mask attempt compared a CPU server against an independent predictor
  that auto-selected CUDA. Exact pixels differed. It was stopped after the first
  failure. The corrected run uses `CUDA_VISIBLE_DEVICES=""` for the independent
  predictor. Pixel assertions and reference data were unchanged; all six mask and
  four tutorial cases passed. See `sam-browser.log` and
  `sam-browser-cpu-reference.log`.
- Six legacy SAM cases initially stopped at their global deprecated-resource
  assertion: Inspire's `js/regional.js` imports `/scripts/ui.js` unconditionally.
  The assertion examines all resource requests, including other extensions. The
  isolated retry enables only Impact Pack and Impact Subpack and keeps every
  assertion unchanged. See `sam-isolation.json` and `legacy-sam-isolated.log`.
- One of 38 LRU cases timed out opening the Widget control mode dropdown, before
  constructing or queuing the wildcard graph. The remaining 37 passed, including
  every new edge case. The unchanged failing case passed three consecutive retries.
  `wildcard-lru-browser` retains the failed trace/screenshot; `wildcard-lru-retry-1`
  through `-3` contain the retries. This is recorded as a browser interaction flake,
  not concealed as a clean first pass. Its intermittent cause was not established.

- The classic wildcard run completed all 63 test bodies, but one teardown caught
  an uncaught frontend `AxiosError: Network Error` from `getCustomNodesI18n` during
  page startup. All five queues in that case passed. The stack is preserved in
  `classic-network-error.json`. Unchanged isolated retries are retained separately
  as `wildcard-classic-retry-*` and all three passed; no JavaScript error filtering
  was added.

## Pre-existing legacy test failures and quality problems

These scripts are outside the maintained pytest default selection. They were run
separately as additional diagnostics. Base `244e93e` reproduces the failures;
`legacy-baseline-extra-sources.json` confirms all backend production files are
identical between base and candidate. No legacy assertion was relaxed.

1. `test_progressive_loading.py::test_wildcard_list_functions` expects all five
   available keys from an API that now returns loaded keys in on-demand mode.
   Both revisions report zero. See `preexisting-legacy-failures.json`.
2. Two `test_lazy_load_verification.py` cases expect on-demand mode after setting
   a 1 MB limit. The test modifies `modules.impact.config`, while wildcard code uses
   `impact.config`; the live limit remains 50 MB. `legacy-module-identity.json`
   records the two module identities and observed limits. Adding the local corpus
   did not fix the assertion. The test also holds an old `wildcard_dict` reference
   after reload replaces that dictionary, so empty-list comparisons are not useful
   coverage. Its pytest functions return values and depend on earlier test state.
3. `test_progressive_ondemand.sh` expects loaded count to grow when `__colors__`
   is used, although its YAML entries were already loaded at startup. Base and
   candidate both report 6 → 6 and exit 1. See `baseline-legacy-extra-progressive.log`
   and `legacy-extra-progressive_ondemand.log`.
4. `test_config_quotes.sh` case 04 references missing `test_nesting_level1` data,
   prints FAIL, then prints an unconditional success summary and exits 0. The case
   labelled "Path with spaces" also uses a path without spaces. Base and candidate
   print the same failure. This suite is **not** reported as passing based on exit
   status. See `baseline-legacy-extra-quotes.log` and `legacy-shell-config_quotes.log`.

Other legacy shell logs were scanned for failure markers in addition to checking
exit status. Print-only legacy checks are not used as proof of exhaustive coverage.
Fixing these independent historical test defects is outside this seed-control PR;
the maintained wildcard unit, browser and tensor suites provide its behavioral
coverage. The full run is therefore not described as an all-green repository run.

## Isolation and reproducibility

All servers use owned runtime copies below this directory, separate ports and
owned-PID termination. Legacy runner adaptations only correct paths, ports and
process cleanup; assertions remain unchanged. Model weights are read from the
installed model directory. The shared checkout and its configuration are not
modified. Runtime copies and local wildcard corpus are excluded from the evidence
manifest. Logs, exact commands, XML results and source hashes remain indexed.
