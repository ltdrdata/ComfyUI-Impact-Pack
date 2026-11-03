# Test review corrections

A subsequent [fresh-checkout revalidation](../reverification/README.md) reproduced the targeted results below without changing this candidate's source.

The accepted regression evidence now contains **576 unique passing pytest cases and 2 existing Processor Bypass xfails**. This update reran the complete configured unit suite (211 cases) and all 12 cases of the changed Control test. The remaining results come from the [previous complete repository run](../full-regression/README.md), with identical production files and unchanged unaffected tests. It is not a claim that all 576 cases were rerun after these test-only edits.

The [current entry-point inventory](test-inventory.json), [acceptance report](validation.json), [final source manifest](../source.json), [original manifest](source-before-review.json) and [follow-up patch](../followup.patch) identify the exact revisions. The [runtime source check](runtime-source-check.json) verifies all 170 tracked files in each server copy, with only recorded environment adaptations in the legacy runner. Production behavior and the trusted widget catalog are unchanged.

| Review issue | Correction | Verification |
|---|---|---|
| Broad Bypass xfail could hide an Encode regression | Restrict the condition to Processor with only populated_text connected | [Real Chrome: 10 passed, 2 existing xfails](controls/bypass.xml); [fault injection](failure-detection.json) proves a broken Encode result fails |
| Empty results and matching request errors could report success | Require both server modes, successful HTTP responses, valid expanded text and all seven results; use supplied server URLs instead of managing processes/configuration | [Actual two-server comparison](ondemand/pair.log), [mode responses and command](api-result.json), [negative HTTP cases](failure-detection.json) |
| Duplicated list access and stale imported dictionaries | Use the real loader module and existing accessors; isolate configuration and cache state with temporary fixtures | [211 passing unit cases](unit.xml), including the two loader-mode cases |
| Assertions expected discovered TXT and preloaded YAML to behave identically | Assert loaded-only list results, progressive TXT access, unchanged YAML/cache access and fresh reloads | [Four progressive unit cases](unit.xml), [corrected shell diagnostic](progressive/test_progressive_ondemand/test.log), [exit status](progressive/results.json) |

The two obsolete Python diagnostics are now six pytest cases in the configured suite. They cover directory-size termination, metadata-only scanning, progressive TXT loading, loaded-list semantics, YAML preload, nested references, cache hits after a disk edit, and fresh reloads in both modes. No implementation-specific list accessor or imported dictionary snapshot remains.

The API negative checks use local HTTP servers to supply HTTP 503, missing text, empty text, an error string, unresolved wildcard text, an incorrect mode and unequal results. A successful control makes seven requests to each server. Separate guards reject empty or incomplete result dictionaries. The Bypass guard check exercises the exact assertion block with broken and valid histories. These fault checks are separate from real Chrome/ComfyUI execution, and are not added to the pytest total.

The [combined cache/Control JUnit](cache-controls.xml) retains the original 99 cases and replaces the 12 affected cases with the fresh results. [Per-case provenance](case-provenance.json) records which execution supplies each case. Previously completed Classic/LRU repetitions remain separately reported, not added to the unique total.

The three historical diagnostic failures are resolved: the two Python files pass as isolated pytest coverage and the shell diagnostic passes against a real on-demand server. The remaining two xfails describe the same native Processor Bypass limitation reproduced before this PR. The assertions were not broadened to accept that failure in Encode.

`unit-invalid-invocation.*` is excluded: pytest was initially launched outside `tests`, so the relative testpaths fell back to broad collection. The accepted command is run from `source/tests`:

```bash
/home/rho/venv/bin/python -m pytest -c pytest.ini -q --tb=short --junitxml=/mnt/teratera/git/impact-pack-pr1269-followup-pr-assets/test-review/unit.xml
```

`controls.json`, `full.json` and `ondemand.json` record exact server jobs. `pair_check.py` coordinates the two isolated API servers; `run_progressive.py` adapts only legacy environment paths, ports and process ownership. [Cleanup](cleanup.json) confirms no owned ComfyUI server remains. No models were downloaded or widget fixtures regenerated.
