# Run All Tests

Execute the complete wildcard system test suite.

## Quick Run

```bash
cd /mnt/teratera/git/ComfyUI/custom_nodes/comfyui-impact-pack/tests

bash test_encoding.sh && \
bash test_error_handling.sh && \
bash test_edge_cases.sh && \
bash test_deep_nesting.sh && \
bash test_ondemand_loading.sh && \
bash test_config_quotes.sh && \
bash test_dynamic_prompts_full.sh

echo ""
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
echo "Total: 86 tests across 7 suites"
echo ""
```

## Individual Tests

```bash
# UTF-8 Encoding (15 tests)
bash test_encoding.sh

# Error Handling (10 tests)
bash test_error_handling.sh

# Edge Cases (20 tests)
bash test_edge_cases.sh

# Deep Nesting (15 tests)
bash test_deep_nesting.sh

# On-Demand Loading (8 tests)
bash test_ondemand_loading.sh

# Config Quotes (5 tests)
bash test_config_quotes.sh

# Dynamic Prompts Full (11 tests)
bash test_dynamic_prompts_full.sh
```

## Test Summary

Each test suite:
- ✅ Starts dedicated ComfyUI server on unique port
- ✅ Configures test wildcard path
- ✅ Runs comprehensive test cases
- ✅ Validates results
- ✅ Cleans up automatically

## Expected Results

All 89 tests should pass (100% pass rate).

## Logs

Test logs are saved in `/tmp/`:
- `/tmp/encoding_test.log`
- `/tmp/error_handling_test.log`
- `/tmp/edge_cases_test.log`
- `/tmp/deep_nesting_test.log`
- `/tmp/ondemand_test.log`
- `/tmp/config_quotes_test.log`
- `/tmp/dynamic_prompt_full_validation.log`
