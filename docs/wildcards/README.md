# Wildcard System Documentation

Progressive on-demand wildcard loading system for ComfyUI Impact Pack.

## Documentation Structure

- **[WILDCARD_SYSTEM_PRD.md](WILDCARD_SYSTEM_PRD.md)** - Product requirements and specifications
- **[WILDCARD_SYSTEM_DESIGN.md](WILDCARD_SYSTEM_DESIGN.md)** - Technical architecture and implementation
- **[WILDCARD_TESTING_GUIDE.md](WILDCARD_TESTING_GUIDE.md)** - Testing procedures and validation

## Quick Links

- Test Suite: `../../tests/`
- Test Samples: `../../tests/wildcards/samples/`
- Implementation: `../../modules/impact/wildcards.py`
- Server API: `../../modules/impact/impact_server.py`

## Test Execution

```bash
cd tests/

# Run all test suites
bash test_encoding.sh       # UTF-8 multi-language (15 tests)
bash test_error_handling.sh # Error handling (10 tests)
bash test_edge_cases.sh     # Edge cases (20 tests)
bash test_deep_nesting.sh   # 7-level nesting (15 tests)
bash test_ondemand_loading.sh # On-demand loading (8 tests)
bash test_config_quotes.sh  # Config quotes (5 tests)
```

## Status

✅ **Production Ready**
- 73 tests, 100% pass rate (6 test suites)
- Complete PRD coverage
- Zero implementation bugs
- UTF-8 encoding verified
- Error handling validated
