# Historical wildcard diagnostics

These scripts are included in the full regression inventory, but their exit codes and printed messages are recorded separately from the assertion-based pytest suites. The comparison baseline is the clean Main + #1269 merge, without this follow-up.

| Diagnostic | Candidate and baseline observation | Evidence |
|---|---|---|
| `test_progressive_ondemand.sh` | Both exit 1: the script expects the loaded count to increase when accessing `colors`, although `colors`, `cold` and `warm` are already loaded. Counts stay 6 → 6. | [candidate log](legacy-with-data/test_progressive_ondemand/test.log), [baseline log](legacy-data-baseline/test_progressive_ondemand/test.log) |
| `test_progressive_loading.py` | Both exit 1 at `Should return all available wildcards`: the legacy test expects five entries and observes zero. Its preceding early-termination, metadata scan and progressive-loading checks pass. | [candidate log](legacy-python-with-data/test_progressive_loading.log), [baseline log](legacy-python-baseline/test_progressive_loading.log) |
| `test_lazy_load_verification.py` | Both exit 1 at `Should be in On-Demand mode`. Its setup observes zero available wildcards and remains in full-cache mode. | [candidate log](legacy-python-with-data/test_lazy_load_verification.log), [baseline log](legacy-python-baseline/test_lazy_load_verification.log) |

These are existing diagnostic failures under the recorded environment. They are not counted as passed or converted to skips. The follow-up does not alter these scripts or their assertions in the PR patch.

`test_wildcard_final.py` compares seven request results successfully, but its 1 MB threshold does not establish on-demand mode with these fixtures. It is result-consistency evidence only. Actual on-demand behavior is covered by the separate [0.0001 MB server job](ondemand.json): its [server log](ondemand/server.log) records metadata-only TXT discovery, and the [browser](ondemand/wildcard-ui.log) and [API](ondemand/encoding-api.log) checks pass in that running mode. The corresponding normal-mode job is recorded in [cpu.json](cpu.json).

The shell and Python scripts run against copies of the same candidate source with isolated ports, input directories and server ownership checks. Recorded `environment-adaptation.patch` files change only environment paths, ports and broad process-stop commands. The accepted final-script run is in [legacy-final/result.json](legacy-final/result.json). An initial unadapted final-script invocation was terminated and excluded; its output is not accepted as evidence.

The [complete shell-output marker review](legacy-marker-review.json) checks FAIL/WARNING/PARTIAL output as well as exit codes. In this run, only `test_progressive_ondemand.sh` emits a non-pass marker; the other ten shell logs do not.
