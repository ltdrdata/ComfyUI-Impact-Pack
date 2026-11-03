# Evidence and reproduction

The latest [full regression](full-head-63d28f8/README.md) reruns pushed head `63d28f838c8040ade1ef122f8cd8985aef542b34` with 575 unique pytest passes and two existing Processor Bypass xfails. [Source hashes](full-head-63d28f8/source.json), [environment](full-head-63d28f8/environment.json), [entry-point inventory](full-head-63d28f8/test-inventory.md) and [accepted results](full-head-63d28f8/validation.json) bind the results to that exact revision.

## Behavior evidence

- [Before](full-head-63d28f8/baseline-mixed-list/observations.json) and [after](full-head-63d28f8/mixed-list/observations.json): seed 5 selects blue instead of reusing seed 2's red; the plain list item remains unchanged.
- [Five real sampling observations](full-head-63d28f8/constant-lora/result.json) and [actual core cache signatures](full-head-63d28f8/constant-lora/node-cache-keys.json): unchanged source/populated text reuses all nodes; edited text invalidates only its dependent path. Requests, execution seeds, cache/execution events and pixel hashes are checked together.
- [Native catalog](full-head-63d28f8/catalog/catalog.xml): all 198 types and 719 widgets retain names, order and saved values in both restoration modes, against unchanged trusted fixtures.
- [Classic Control/cache suite](full-head-63d28f8/classic/cache-controls.xml), [LRU configuration](full-head-63d28f8/verified-lru/cache.xml), and [baseline Bypass reproduction](full-head-63d28f8/baseline-controls/bypass.xml) separate new behavior from the two existing Processor limitations.
- [Annotated screenshot provenance](annotated-screenshots/manifest.json) retains original captures behind the four PR images. The fresh full report also links new original browser captures of the same sampling behavior.
- [Fourteen failure probes](full-head-63d28f8/failure-detection.json) test API error/missing-result rejection and the narrow Bypass xfail guard separately from real browser execution.

## Reproduction

Use isolated servers, Google Chrome and Playwright with the installed models and fixture paths recorded in the full report. The [committed E2E strategy](https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/63d28f838c8040ade1ef122f8cd8985aef542b34/docs/E2E_TEST_STRATEGY.md) documents repository setup. The full report links every job JSON with exact commands, environment and cache options, along with test-only environment adaptations and failed setup logs.

Configured unit suite, from the extension root:

```bash
(cd tests && python -m pytest -c pytest.ini -q)
```

Classic Chrome suite against an isolated CPU server:

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
python -m pytest -c tests/pytest.ini \
  tests/test_e2e_wildcard_cache.py tests/test_e2e_wildcard_controls.py -v
```

Real core hook/list integration:

```bash
python tests/integration_wildcard_cache.py --comfy-root /path/to/ComfyUI --report /tmp/wildcard-cache.json
```

The Encode matrix uses a CLIP stand-in. Installed-model tensor and actual sampling checks are separate.

Actual full-cache/on-demand API parity, with the same required wildcard fixtures on both servers:

```bash
python tests/wildcards/test_wildcard_final.py \
  --full-url http://127.0.0.1:18188 --on-demand-url http://127.0.0.1:18189
```

The final head has 210 configured unit cases. The earlier synthetic Main combination contained one additional path test, hence its cumulative total of 576. That historical total is superseded by the fresh exact-head report, not combined with it.

[Split verification](commit-split/validation.json) preserves the reviewed file tree while separating behavior and loader cleanup. #1269 remains open; after it lands, rebase the two follow-up commits onto current Main and revalidate before making the PR ready.
