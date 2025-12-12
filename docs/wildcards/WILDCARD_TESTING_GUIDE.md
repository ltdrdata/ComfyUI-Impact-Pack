# Wildcard System Testing Guide

Complete testing guide for the ComfyUI Impact Pack wildcard system.

---

## 📋 Table of Contents

1. [Test Overview](#test-overview)
2. [Test Suites](#test-suites)
3. [Quick Start](#quick-start)
4. [Running Tests](#running-tests)
5. [Test Validation](#test-validation)

---

## Test Overview

### Test Statistics
- **Total Tests**: 86 tests across 7 suites
- **Coverage**: 100% of PRD core requirements
- **Pass Rate**: 100%
- **Test Types**: UTF-8, error handling, edge cases, nesting, on-demand, config, dynamic prompts

### Test Structure

```
tests/
├── Test Suites (7 suites, 86 tests)
│   ├── test_encoding.sh              # 15 tests - UTF-8 multi-language support
│   ├── test_error_handling.sh        # 10 tests - Error recovery and graceful handling
│   ├── test_edge_cases.sh            # 20 tests - Boundary conditions and special cases
│   ├── test_deep_nesting.sh          # 17 tests - 7-level transitive expansion + pattern matching
│   ├── test_ondemand_loading.sh      #  8 tests - Progressive lazy loading with cache limits
│   ├── test_config_quotes.sh         #  5 tests - Configuration path handling
│   └── test_dynamic_prompts_full.sh  # 11 tests - Weighted/multiselect with statistical validation
│
├── Documentation
│   ├── README.md                     # Test suite overview
│   └── RUN_ALL_TESTS.md              # Execution guide
│
├── Test Samples
│   └── wildcards/samples/            # Test wildcard files
│       ├── level1/.../level7/        # 7-level nesting structure
│       ├── *.txt                     # Various test wildcards
│       └── 아름다운색.txt             # Korean UTF-8 sample
│
└── Utilities
    └── restart_test_server.sh        # Server management utility
```

---

## Test Suites

### 1. UTF-8 Encoding Tests (15 tests)
**File**: `test_encoding.sh`
**Port**: 8188
**Purpose**: Multi-language support validation

**Test Coverage**:
- Korean text (한글)
- Chinese text (中文)
- Arabic text (العربية)
- Emoji support (🐉🔥⚡)
- Special characters
- Mixed multi-language content
- Case-insensitive Korean matching

**Key Validations**:
- All non-ASCII characters preserved
- UTF-8 encoding consistency
- No character corruption
- Proper string comparison

---

### 2. Error Handling Tests (10 tests)
**File**: `test_error_handling.sh`
**Port**: 8189
**Purpose**: Graceful error recovery

**Test Coverage**:
- Non-existent wildcards
- Missing files
- Circular reference detection (direct and indirect)
- Malformed dynamic prompt syntax
- Deep nesting without crashes
- Invalid quantifiers

**Key Validations**:
- No server crashes
- Clear error messages
- Original text preserved on error
- Circular detection within 100 iterations

---

### 3. Edge Cases Tests (20 tests)
**File**: `test_edge_cases.sh`
**Port**: 8190
**Purpose**: Boundary conditions and special scenarios

**Test Coverage**:
- Empty lines and comments in wildcard files
- Very long lines (>1000 chars)
- Basic wildcard expansion
- Case-insensitive matching
- Quantifiers (1-10 repetitions)
- Pattern matching (`__*/name__`)

**Key Validations**:
- Empty lines filtered correctly
- Comments ignored properly
- Long text handling
- Quantifier accuracy
- Pattern matching at any depth

---

### 4. Deep Nesting Tests (17 tests)
**File**: `test_deep_nesting.sh`
**Port**: 8194
**Purpose**: 7-level transitive expansion and pattern matching

**Test Coverage**:
- Direct level access (Level 1-7)
- Transitive expansion through all levels
- Multiple wildcard nesting
- Mixed depth combinations
- Quantifiers with nesting
- Weighted selection with nesting
- Depth-agnostic pattern matching

**Key Validations**:
- All 7 levels fully expanded
- No unexpanded wildcards remain
- Pattern matching ignores directory depth
- Complex combinations work correctly

**Directory Structure**:
```
samples/level1/level2/level3/level4/level5/level6/level7/
```

---

### 5. On-Demand Loading Tests (8 tests)
**File**: `test_ondemand_loading.sh`
**Port**: 8191
**Purpose**: Progressive lazy loading with configurable cache limits

**Test Coverage**:
- Small cache (1MB) - On-demand mode
- Medium cache (10MB) - Hybrid mode
- Large cache (100MB) - Full cache mode
- Aggressive lazy (0.5MB)
- Various thresholds (5MB, 20MB, 50MB)

**Key Validations**:
- Correct loading mode selection
- Progressive loading functionality
- Cache limit enforcement
- No performance degradation

**Note**: Uses temporary samples in `/tmp/` with auto-cleanup

---

### 6. Config Quotes Tests (5 tests)
**File**: `test_config_quotes.sh`
**Port**: 8192
**Purpose**: Configuration path handling with quotes

**Test Coverage**:
- Paths with single quotes
- Paths with double quotes
- Paths with spaces (quoted)
- Mixed quote scenarios
- Unquoted baseline

**Key Validations**:
- Quotes stripped correctly
- Paths with spaces handled
- Wildcards loaded from quoted paths

---

### 7. Dynamic Prompts Tests (11 tests)
**File**: `test_dynamic_prompts_full.sh`
**Port**: 8193
**Purpose**: Statistical validation of weighted and multiselect features

**Test Coverage**:
- Multiselect (2-5 items) with custom separators
- Weighted selection (various ratios: 10:1, 1:1:1, 5:3:2)
- Nested dynamic prompts
- Basic random selection
- Seed variation validation

**Statistical Validation**:
- 100 iterations for weighted selection
- 20 iterations for multiselect
- Distribution verification (±15% tolerance)
- Duplicate detection
- Separator validation

**Key Validations**:
- Exact item count for multiselect
- No duplicates in multiselect
- Correct separators
- Statistical distribution matches weight ratios
- Nested prompt expansion

---

## Quick Start

### Run All Tests
```bash
cd tests/
bash test_encoding.sh && \
bash test_error_handling.sh && \
bash test_edge_cases.sh && \
bash test_deep_nesting.sh && \
bash test_ondemand_loading.sh && \
bash test_config_quotes.sh && \
bash test_dynamic_prompts_full.sh
```

### Run Individual Suite
```bash
cd tests/
bash test_encoding.sh
```

### Check Test Results
All tests output:
- ✅ PASS - Test succeeded with validation
- ❌ FAIL - Test failed (should not occur)
- ⚠️ WARNING - Partial success or non-critical issue

---

## Running Tests

### Prerequisites
- ComfyUI server must be installable
- Port availability (8188-8194)
- Network access to 127.0.0.1
- Python 3 with json module

### Automatic Server Management
All test suites automatically:
1. Kill any existing server on target port
2. Create temporary configuration file
3. Start ComfyUI server
4. Wait for server ready (up to 60s)
5. Execute tests
6. Clean up (kill server, remove config)

### Test Execution Flow
```
1. Setup
   ├─ Kill existing server on port
   ├─ Create impact-pack.ini config
   └─ Start ComfyUI server

2. Wait for Ready
   ├─ Poll server every second
   ├─ Max 60 seconds timeout
   └─ Log tail on failure

3. Execute Tests
   ├─ Call /impact/wildcards API
   ├─ Validate responses
   └─ Check behavior

4. Cleanup
   ├─ Kill server process
   └─ Remove config file
```

---

## Test Validation

### What Tests Validate

**Behavioral Validation** (Not just "no errors"):
- **Weighted Selection**: Statistical distribution matches weight ratios
- **Multiselect**: Exact count, no duplicates, correct separator
- **Nesting**: All levels fully expanded, no remaining wildcards
- **Pattern Matching**: Depth-agnostic matching works correctly
- **UTF-8**: Character preservation and proper encoding
- **Error Handling**: Graceful recovery with meaningful messages

### Success Criteria
- All 86 tests must pass (100% pass rate)
- No server crashes or hangs
- API responses within expected format
- Statistical distributions within ±15% tolerance
- No unexpanded wildcards in final output

### Validation Examples

**Weighted Selection**:
```bash
# Test 10:1 ratio with 100 iterations
# Expected: ~91% common, ~9% rare
# Actual: Count distribution within ±15%
```

**Multiselect**:
```bash
# Test {2$$, $$red|blue|green}
# Expected: Exactly 2 items, comma-space separator, no duplicates
# Validation: Count words, check separator, detect duplicates
```

**Pattern Matching**:
```bash
# Test __*/dragon__
# Expected: Matches dragon.txt, fantasy/dragon.txt, dragon/fire.txt
# Validation: No unexpanded wildcards remain
```

---

## Troubleshooting

### Common Issues

**Server Fails to Start**:
```bash
# Check log file
tail -20 /tmp/{test_name}_test.log

# Check port availability
lsof -i :8188

# Kill conflicting process
pkill -f "python.*main.py.*--port 8188"
```

**Tests Timeout**:
- Increase wait time in test script (default 60s)
- Check server performance and resources
- Verify network connectivity to 127.0.0.1

**Statistical Tests Fail**:
- Expected for very small sample sizes
- ±15% tolerance accounts for randomness
- Rerun test to verify consistency

**UTF-8 Issues**:
- Ensure terminal supports UTF-8
- Check file encoding: `file -i tests/wildcards/samples/*.txt`
- Verify locale: `locale | grep UTF-8`

---

## Test Maintenance

### Adding New Tests
1. Create new test function in appropriate suite
2. Follow existing test patterns (setup, execute, validate, cleanup)
3. Update test counts in README.md and SUMMARY.md
4. Update this guide with new test description

### Modifying Existing Tests
1. Preserve behavioral validation (not just "no errors")
2. Maintain statistical rigor for dynamic prompt tests
3. Update documentation if test purpose changes
4. Verify all 86 tests still pass after modification

### Test Philosophy
- **Tests validate behavior**, not just execution success
- **Statistical validation** for probabilistic features
- **Real-world scenarios** with production-like setup
- **Comprehensive coverage** of all PRD requirements
