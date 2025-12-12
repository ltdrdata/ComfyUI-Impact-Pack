# Wildcard System Test Suite

Comprehensive test suite for ComfyUI Impact Pack wildcard system.

## Test Suites

### test_encoding.sh (15 tests)
**Purpose**: UTF-8 multi-language encoding validation
**Port**: 8198
**Coverage**:
- Korean Hangul characters
- Emoji support
- Chinese characters
- Arabic RTL text
- Mathematical and currency symbols
- Mixed multi-language content
- UTF-8 in dynamic prompts, quantifiers, multi-select

### test_error_handling.sh (10 tests)
**Purpose**: Graceful error handling verification
**Port**: 8197
**Coverage**:
- Non-existent wildcards
- Circular reference detection (max 100 iterations)
- Malformed syntax
- Deep nesting without crashes
- Multiple circular references

### test_edge_cases.sh (20 tests)
**Purpose**: Edge case and boundary condition validation
**Port**: 8196
**Coverage**:
- Empty lines and whitespace filtering
- Very long lines (>1000 characters)
- Special characters preservation
- Case-insensitive matching
- Comment line filtering
- Pattern matching (__*/name__)
- Quantifiers (N#__wildcard__)
- Complex syntax combinations

### test_deep_nesting.sh (17 tests)
**Purpose**: Transitive wildcard expansion and depth-agnostic pattern matching
**Port**: 8194
**Coverage**:
- 7-level transitive expansion (directory depth + file references)
- All depth levels (1-7) individually
- Mixed depth combinations
- Nesting with quantifiers and multi-select
- Nesting with weighted selection
- Depth-agnostic pattern matching (`__*/name__`)
- Complex multi-wildcard prompts

### test_ondemand_loading.sh (8 tests)
**Purpose**: Progressive on-demand wildcard loading
**Port**: 8193
**Coverage**:
- Small cache (1MB) - on-demand enabled
- Moderate cache (10MB) - progressive loading
- Large cache (100MB) - eager loading
- Aggressive lazy loading (0.5MB)
- Balanced mode (50MB default)
- On-demand with deep nesting
- On-demand with multiple wildcards
- Cache boundary testing

### test_config_quotes.sh (5 tests)
**Purpose**: Configuration path handling validation
**Port**: 8192
**Coverage**:
- Unquoted paths
- Double-quoted paths
- Single-quoted paths
- Paths with spaces
- Mixed quote scenarios

### test_dynamic_prompts_full.sh (11 tests)
**Purpose**: Comprehensive dynamic prompt feature validation with statistical analysis
**Port**: 8188
**Coverage**:
- **Multiselect** (4 tests): 2-item, 3-item, single-item, max-item with separator validation
- **Weighted Selection** (5 tests): 10:1 ratio, equal weights, extreme bias, multi-level weights, default mixing
- **Basic Selection** (2 tests): Simple random, nested selection
- Statistical distribution verification (100+ iterations per test)
- Duplicate detection and item count validation
- Separator correctness validation

## Quick Start

```bash
# Run individual test
bash test_encoding.sh

# Run all tests
bash test_encoding.sh
bash test_error_handling.sh
bash test_edge_cases.sh
bash test_deep_nesting.sh
bash test_ondemand_loading.sh
bash test_config_quotes.sh
bash test_dynamic_prompts_full.sh
```

## Test Infrastructure

- **Configuration**: Each test creates `impact-pack.ini` with test wildcard path
- **Server Lifecycle**: Automatic server start/stop with dedicated ports
- **Cleanup**: Automatic cleanup on test completion
- **Logging**: Detailed logs in `/tmp/*_test.log`

## Test Samples

Located in `wildcards/samples/`:
- `아름다운색.txt` - Korean UTF-8 test with 12 symbolic colors
- `test_encoding_*.txt` - UTF-8 encoding test files
- `test_edge_*.txt` - Edge case test files
- `test_error_*.txt` - Error handling test files
- `test_nesting_*.txt` - Nesting test files (7 levels)
- `patterns/` - Subdirectory for pattern matching tests

## Status

✅ **86 tests, 100% pass rate** (15+10+20+17+8+5+11)
✅ **Production ready**
✅ **Complete PRD coverage**
✅ **On-demand loading validated**
✅ **Config quotes handling validated**
✅ **Dynamic prompts statistically validated**
✅ **Weighted selection verified (correct {weight::option} syntax)**
✅ **Pattern matching validated (depth-agnostic __*/name__)**

## Documentation

- [Wildcard System PRD](../docs/wildcards/WILDCARD_SYSTEM_PRD.md)
- [System Design](../docs/wildcards/WILDCARD_SYSTEM_DESIGN.md)
- [Testing Guide](../docs/wildcards/WILDCARD_TESTING_GUIDE.md)
