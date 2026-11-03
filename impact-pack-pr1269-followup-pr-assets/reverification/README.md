# Fresh revalidation

The current candidate was extracted into a new directory and tested with new isolated ComfyUI servers. **211 unit cases passed; the same 211 passed again with loader tests first. Chrome Control coverage passed 10 cases with the same 2 baseline Processor Bypass xfails.** No new failure was found in this run.

This is a targeted revalidation, not a fresh run of all 576 cases in the cumulative acceptance record. The candidate tree remains `4897facaf7f842719bf37180fab346ab2fda17ab`; no production or test source was changed during revalidation. The [checked report](validation.json) binds the results to the [source manifest](source.json) and [current patch](../followup.patch).

| Check | Result and evidence |
|---|---|
| Complete configured unit suite | 211 passed: [JUnit](unit.xml), [log](unit.log) |
| Loader fixtures before the remaining unit suite | The same 211 passed: [JUnit](unit-reordered.xml), [command and exit status](unit-reordered-result.json). This repetition is not an additional unique count. |
| Real Chrome Control mode changes | 10 passed, 2 existing xfails: [JUnit](controls/bypass.xml), [log](controls/bypass.log), [server and command](controls/result.json). Both positional and named native restoration are covered. |
| Real prompt hooks, cache signatures and list mapping | 20 groups passed: [report](integration.json), [log](integration.log). The Encode text matrix uses a CLIP stand-in; this run makes no new GPU-inference claim. |
| Actual full-cache versus on-demand API | All 7 result pairs matched: [expanded results](ondemand/pair.log), [mode responses and command](api-result.json) |
| Failure detection | 10 API checks and 4 Bypass checks behaved as expected: [observations](failure-detection.json). Local fault responses and extracted assertion checks are separate from real Chrome/API execution. |
| Progressive loading diagnostic | Passed on a real isolated on-demand server: [log](progressive/test_progressive_ondemand/test.log), [exit status](progressive/results.json), [environment-only adaptation](progressive/test_progressive_ondemand/environment-adaptation.patch) |
| Static checks | Relevant Ruff checks, shell syntax and patch whitespace passed: [commands and output](static-checks.json) |

The two xfails are the Processor populated-only Bypass limitation already reproduced on [Main + #1269](../baseline-controls/bypass.xml). Encode populated-only Bypass, connected-first-input Bypass for both nodes, and Mute/reactivation passed. The failure probes separately show that a broken Encode preview fails rather than becoming xfail.

The report verifies all 170 tracked source files in each runtime copy. Only the legacy shell runner has a recorded path/port/process-ownership adaptation. It also verifies all 13 follow-up files against the editable worktree, checks both trusted widget-catalog fixture hashes, confirms all four owned server ports are closed, and confirms no owned ComfyUI server remains. [Server job records](controls/result.json), [full-cache record](full/result.json) and [on-demand record](ondemand/result.json) retain their exact commands and runtime identity.

To repeat the configured unit run, use `source/tests` as the working directory:

```bash
/home/rho/venv/bin/python -m pytest -c pytest.ini -q --tb=short --junitxml=../../unit.xml
```

`run_unit_order.py` records the alternative order. `controls.json`, `full.json` and `ondemand.json` specify the isolated server jobs consumed by `run_server_suite.py`; the full/on-demand jobs run concurrently through `pair_check.py`. `run_progressive.py` scopes the legacy shell script to its own runtime. `check_failure_detection.py` runs the negative probes, and `summarize.py` verifies results, source hashes and cleanup before writing the report. The [previous test-review record](../test-review/README.md) connects this unchanged candidate to the earlier full repository run.
