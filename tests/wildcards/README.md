# Wildcard System - Complete Test Suite

The automated loader regressions run with
`(cd tests && python -m pytest -c pytest.ini)` from the repository root. This includes
`test_progressive_loading.py` and `test_lazy_load_verification.py`, which use
isolated temporary fixtures rather than external wildcard collections.
`test_wildcard_final.py` requires `--full-url` and `--on-demand-url` for two
already-running isolated servers; see [current validation commands](../../docs/E2E_TEST_STRATEGY.md).
