# Wildcard System - Project Summary

## Overview

Progressive on-demand wildcard loading system for ComfyUI Impact Pack with dynamic prompt support, UTF-8 encoding, and comprehensive testing.

**Status**: ✅ Production Ready
**Test Coverage**: 86 tests, 100% pass rate
**Documentation**: Complete PRD, design docs, and testing guide

---

## Core Features

- **Wildcard Expansion**: `__wildcard__` syntax with transitive multi-level expansion
- **Dynamic Prompts**:
  - Basic selection: `{option1|option2|option3}`
  - Weighted selection: `{10::common|1::rare}` (weight comes first)
  - Multi-select: `{2$$, $$red|blue|green}` with custom separators
- **UTF-8 Support**: Korean, Chinese, Arabic, emoji, special characters
- **Pattern Matching**: Depth-agnostic `__*/name__` syntax
- **On-Demand Loading**: Progressive lazy loading with configurable cache limits
- **Error Handling**: Circular reference detection, graceful fallbacks

---

## Architecture

### Implementation
- `modules/impact/wildcards.py` - Core LazyWildcardLoader and expansion engine
- `modules/impact/impact_server.py` - Server API endpoint (/impact/wildcards)
- `modules/impact/config.py` - Configuration with quoted path support

### Key Design Decisions
- **Lazy Loading**: Memory-efficient progressive loading strategy
- **Transitive Expansion**: Multi-level wildcard references through directory hierarchy
- **Case-Insensitive Matching**: Fuzzy matching for user convenience
- **Circular Reference Detection**: Max 100 iterations with clear error messages

---

## Testing

### Test Suites (86 tests)
1. **UTF-8 Encoding** (15 tests) - Multi-language support validation
2. **Error Handling** (10 tests) - Graceful error recovery
3. **Edge Cases** (20 tests) - Boundary conditions and special scenarios
4. **Deep Nesting** (17 tests) - 7-level transitive expansion + pattern matching
5. **On-Demand Loading** (8 tests) - Progressive loading with cache limits
6. **Config Quotes** (5 tests) - Configuration path handling
7. **Dynamic Prompts** (11 tests) - Statistical validation of dynamic features

### Test Infrastructure
- Dedicated ports per suite (8188-8198)
- Automated server lifecycle management
- Comprehensive logging in `/tmp/`
- 100% pass rate with statistical validation

---

## Documentation

- **[README](README.md)** - Quick start and feature overview
- **[PRD](WILDCARD_SYSTEM_PRD.md)** - Complete product requirements
- **[Design](WILDCARD_SYSTEM_DESIGN.md)** - Technical architecture
- **[Testing Guide](WILDCARD_TESTING_GUIDE.md)** - Test procedures and validation

---

## Quick Start

### Basic Usage
```python
# Simple wildcard
"a photo of __animal__"

# Dynamic prompt
"a {red|blue|green} __vehicle__"

# Weighted selection (weight comes FIRST)
"{10::common|1::rare} scene"

# Multi-select
"{2$$, $$happy|sad|angry|excited} person"
```

### Running Tests
```bash
cd tests/
bash test_encoding.sh
bash test_error_handling.sh
bash test_edge_cases.sh
bash test_deep_nesting.sh
bash test_ondemand_loading.sh
bash test_config_quotes.sh
bash test_dynamic_prompts_full.sh
```

---

## Key Implementations

### Weighted Selection Syntax
**Correct**: `{weight::option}` - Weight comes FIRST
- `{10::common|1::rare}` → 91% common, 9% rare ✅
- `{5::red|3::green|2::blue}` → 50%, 30%, 20% ✅

**Incorrect**: `{option::weight}` - Treated as equal weights
- `{common::10|rare::1}` → 50% each ❌

### Empty Line Filtering
Filter empty lines AND comment lines:
```python
[x for x in lines if x.strip() and not x.strip().startswith('#')]
```

### Config Path Quotes
Strip quotes from configuration paths:
```python
custom_wildcards_path = default_conf.get('custom_wildcards', '').strip('\'"')
```

---

## Limitations

- Weighted selection supports integers and simple decimals only
- Complex decimal weights may conflict with multiselect pattern detection
- Circular references limited to 100 iterations
- Prefer integer weight ratios for clarity

---

## Performance

- **Lazy Loading**: Only load wildcards when needed
- **On-Demand Mode**: Progressive loading based on cache limits
- **Memory Efficient**: Configurable cache size (0.5MB - 100MB)
- **Fast Lookup**: Optimized directory traversal with pattern matching

---

## Production Ready

✅ Zero known bugs
✅ Complete PRD coverage
✅ 100% test pass rate
✅ Statistical validation
✅ Comprehensive documentation
✅ Multi-language support
✅ Graceful error handling
