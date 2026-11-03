# Evidence and reproduction

The tested source combines Main `082523b2dbbe0a38282f021ad3a0e33e9f222562`, #1269 `609e411e13c6da94fefb07982d63b90417122412`, and follow-up commit `432fd24ef0e05c42e2f865fb05575b01567097fe`. Its tree is `4897facaf7f842719bf37180fab346ab2fda17ab`. [Source hashes](source.json), [environment and model identifiers](environment.json), [patch](followup.patch), and [remote commit verification](push-record.json) identify the exact inputs.

## What the evidence establishes

- [Cold requests](cold-queue-comparison.json): two pending requests reuse constant results before a previous target execution can teach the cache a seed.
- [List baseline](baseline-mixed-list/observations.json) and [fixed list](focus-mixed-list/observations.json): seed 5 must change the wildcard item from red to blue while retaining the static item.
- [Sampling observations](constant-lora/result.json) and [core cache signatures](constant-lora/node-cache-keys.json): unchanged text reuses all eight nodes; edited text changes only its dependent path. Source text, execution seeds, cached/executing node sets and pixel hashes are recorded together.
- [Screenshot provenance](annotated-screenshots/manifest.json): four annotated images retain the original browser captures and identify their requests, histories and events. The PR keeps source links outside the images.
- [Accepted regression](validation.json), [test cleanup](test-review/validation.json), [case provenance](test-review/case-provenance.json), and [fresh revalidation](reverification/validation.json): 576 unique passes plus 2 baseline xfails; repetitions are not added to the total. [Earlier full run](full-regression/validation.json) records the historical diagnostic failures resolved by the cleanup.

The Processor populated-only Bypass limitation is reproduced in the [baseline JUnit](baseline-controls/bypass.xml). The [fresh Chrome run](reverification/controls/bypass.xml) verifies that Encode and connected-first-input combinations pass. [Fault injection](reverification/failure-detection.json) is separate from real browser/API execution and proves that these tests reject representative wrong results.

## Commands

Use an isolated ComfyUI installation with the tested extension source, Google Chrome and Python Playwright. The recorded environment used ComfyUI `0.43.0`, frontend `1.57.6`, Python `3.12.3` and Chrome `156.0.8078.79`. Native catalog tests require the trusted fixtures and fully loaded wildcard categories. See the [committed E2E strategy](https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/432fd24ef0e05c42e2f865fb05575b01567097fe/docs/E2E_TEST_STRATEGY.md) for setup.

Configured unit suite, from the extension root:

```bash
(cd tests && python -m pytest -c pytest.ini -q)
```

Chrome cache/Control suite against an isolated `--cpu --cache-classic` server:

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
python -m pytest -c tests/pytest.ini \
  tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_controls.py -v
```

The [Classic job](classic.json), [LRU-64 job](verified-lru.json), [native catalog/tutorial/switch job](catalog.json) and [latest Control job](reverification/controls.json) retain exact selections, server options and artifact paths. The [unit order command](reverification/unit-reordered-result.json) reruns the same 211 cases with loader tests first.

Hook/cache/list-mapper integration:

```bash
python tests/integration_wildcard_cache.py --comfy-root /path/to/ComfyUI --report /tmp/wildcard-cache.json
```

The Encode matrix uses a CLIP stand-in. [Real tensor comparisons](cache-tensors.json) and the actual sampling evidence above cover installed models separately.

API parity requires two isolated servers with the same `samples/flower`, `samples/jewel` and `colors` fixtures, one in full-cache mode and one in on-demand mode:

```bash
python tests/wildcards/test_wildcard_final.py \
  --full-url http://127.0.0.1:18188 --on-demand-url http://127.0.0.1:18189
```

The [actual API output](reverification/ondemand/pair.log) records all seven matching expansions. The [progressive diagnostic log](reverification/progressive/test_progressive_ondemand/test.log), [exit status](reverification/progressive/results.json) and [runner adaptation](reverification/progressive/test_progressive_ondemand/environment-adaptation.patch) show the separate on-demand check; the adaptation changes only paths, port and owned-process cleanup. [Static checks](reverification/static-checks.json) cover relevant Ruff diagnostics, shell syntax and patch whitespace.
