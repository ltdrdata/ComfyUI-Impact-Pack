# Test Wildcard Files Documentation

This directory contains test wildcard files created to validate various features and edge cases of the wildcard system.

## Test Categories

### 1. Error Handling Tests

**test_error_cases.txt**
- Purpose: Test handling of non-existent wildcard references
- Contains: References to `__nonexistent_wildcard__` that should be handled gracefully
- Expected: System should not crash, provide meaningful error or leave unexpanded

**test_circular_a.txt + test_circular_b.txt**
- Purpose: Test circular reference detection (A→B→A)
- Contains: Mutual references between two wildcards
- Expected: System should detect cycle and prevent infinite loop (max 100 iterations)

### 2. Encoding Tests

**test_encoding_utf8.txt**
- Purpose: Test UTF-8 multi-language support
- Contains:
  - Emoji: 🌸🌺🌼🌻🌷
  - Japanese: さくら, はな, 美しい花, 桜の木
  - Chinese: 花, 玫瑰, 莲花, 牡丹
  - Korean: 꽃, 장미, 벚꽃
  - Arabic (RTL): زهرة, وردة
  - Mixed: `🌸 beautiful 美しい flower زهرة 꽃`
- Expected: All characters render correctly, no encoding errors

**test_encoding_emoji.txt**
- Purpose: Test emoji handling across categories
- Contains: Nature, animals, food, hearts, and mixed emoji with text
- Expected: Emojis render correctly in results

**test_encoding_special.txt**
- Purpose: Test special Unicode characters
- Contains:
  - Mathematical symbols: ∀∂∃∅∆∇∈∉
  - Greek letters: α β γ δ ε ζ
  - Currency: $ € £ ¥ ₹ ₽ ₩
  - Box drawing: ┌─┬─┐
  - Diacritics: Café résumé naïve Zürich
  - Special punctuation: … — – • · °
- Expected: All symbols preserved correctly

### 3. Edge Case Tests

**test_edge_empty_lines.txt**
- Purpose: Test handling of empty lines and whitespace-only lines
- Contains: Options separated by variable empty lines
- Expected: Empty lines ignored, only non-empty options selected

**test_edge_whitespace.txt**
- Purpose: Test leading/trailing whitespace handling
- Contains: Options with tabs, spaces, mixed whitespace
- Expected: Whitespace handling according to parser rules

**test_edge_long_lines.txt**
- Purpose: Test very long line handling
- Contains:
  - Short lines
  - Medium lines (~100 chars)
  - Very long lines with spaces (>200 chars)
  - Ultra-long lines without spaces (continuous text)
- Expected: No truncation or memory issues, proper handling

**test_edge_special_chars.txt**
- Purpose: Test special characters that might cause parsing issues
- Contains:
  - Embedded wildcard syntax: `__wildcard__` as literal text
  - Dynamic prompt syntax: `{option|option}` as literal text
  - Regex special chars: `.`, `*`, `+`, `?`, `|`, `\`, `$`, `^`
  - Quote characters: `"`, `'`, `` ` ``
  - HTML special chars: `&`, `<`, `>`, `=`
- Expected: Special chars treated as literal text in final output

**test_edge_case_insensitive.txt**
- Purpose: Validate case-insensitive wildcard matching
- Contains: Options in various case patterns
- Expected: `__test_edge_case_insensitive__` and `__TEST_EDGE_CASE_INSENSITIVE__` return same results

**test_comments.txt**
- Purpose: Test comment handling with `#` prefix
- Contains: Lines starting with `#` mixed with valid options
- Expected: Comment lines ignored, only non-comment lines selected

### 4. Deep Nesting Tests (7 levels)

**test_nesting_level1.txt → test_nesting_level7.txt**
- Purpose: Test transitive wildcard expansion up to 7 levels
- Structure:
  - Level 1 → references Level 2
  - Level 2 → references Level 3
  - ...
  - Level 7 → final options (no further references)
- Usage: Access `__test_nesting_level1__` to trigger 7-level expansion
- Expected: All levels expand correctly, result from level 7 appears

### 5. Syntax Feature Tests

**test_quantifier.txt**
- Purpose: Test quantifier syntax `N#__wildcard__`
- Contains: List of color options
- Usage: `3#__test_quantifier__` should expand to 3 repeated wildcards
- Expected: Correct repetition and expansion

**test_pattern_match.txt**
- Purpose: Test pattern matching `__*/name__`
- Contains: Options with identifiable pattern
- Usage: `__*/test_pattern_match__` should match this file
- Expected: Depth-agnostic matching works correctly

## Test Usage Examples

### Basic Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__test_encoding_emoji__", "seed": 42}'
```

### Nesting Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__test_nesting_level1__", "seed": 42}'
```

### Error Handling Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__test_error_cases__", "seed": 42}'
```

### Circular Reference Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__test_circular_a__", "seed": 42}'
```

### Quantifier Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "3#__test_quantifier__", "seed": 42}'
```

### Pattern Matching Test
```bash
curl -X POST http://127.0.0.1:8188/impact/wildcards \
  -H "Content-Type: application/json" \
  -d '{"text": "__*/test_pattern_match__", "seed": 42}'
```

## Test Coverage

These test files address the following critical gaps identified in the test coverage analysis:

1. ✅ **Error Handling** - Missing wildcard files, circular references
2. ✅ **UTF-8 Encoding** - Multi-language support (emoji, CJK, RTL)
3. ✅ **Edge Cases** - Empty lines, whitespace, long lines, special chars
4. ✅ **Deep Nesting** - 7-level transitive expansion
5. ✅ **Comment Handling** - Lines starting with `#`
6. ✅ **Case Insensitivity** - Case-insensitive wildcard matching
7. ✅ **Pattern Matching** - `__*/name__` syntax
8. ✅ **Quantifiers** - `N#__wildcard__` syntax

## Expected Test Results

All tests should:
- Not crash the system
- Return valid results or graceful error messages
- Preserve character encoding correctly
- Handle edge cases without data corruption
- Respect the 100-iteration limit for circular references
- Demonstrate deterministic behavior with same seed

---

**Created**: 2025-11-18
**Purpose**: Test coverage validation for wildcard system
**Total Files**: 21 test wildcard files
