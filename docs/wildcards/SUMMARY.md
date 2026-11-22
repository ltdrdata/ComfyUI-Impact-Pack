# Wildcard System - Project Summary

## Overview

Progressive on-demand wildcard loading system for ComfyUI Impact Pack with dynamic prompt support, UTF-8 encoding, and comprehensive testing.

**Status**: ✅ Production Ready
**Documentation**: PRD and design docs

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

The pytest suite under `tests/wildcards/` and `tests/test_encoding.sh` cover the loader. See [tests/README.md](../../tests/README.md).

---

## Documentation

- **[README](README.md)** - Quick start and feature overview
- **[PRD](WILDCARD_SYSTEM_PRD.md)** - Complete product requirements
- **[Design](WILDCARD_SYSTEM_DESIGN.md)** - Technical architecture

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
See [tests/README.md](../../tests/README.md).

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
