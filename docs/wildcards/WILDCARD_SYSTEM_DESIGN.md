# Wildcard System - Design Document

**Document Type**: Technical Design Document
**Product**: ComfyUI Impact Pack Wildcard System
**Version**: 2.0 (Depth-Agnostic Matching)
**Last Updated**: 2025-11-18
**Status**: Released

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ComfyUI Frontend                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ImpactWildcardProcessor / ImpactWildcardEncode      │   │
│  │  - Wildcard Prompt (editable)                        │   │
│  │  - Populated Prompt (read-only in Populate mode)     │   │
│  │  - Mode: Populate / Fixed                            │   │
│  │  - UI Indicator: 🟢 Full Cache / 🔵 On-Demand       │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Impact Server (API)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /impact/wildcards                              │   │
│  │  GET  /impact/wildcards/list                         │   │
│  │  GET  /impact/wildcards/list/loaded                  │   │
│  │  GET  /impact/wildcards/refresh                      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Wildcard Processing Engine                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  process() - Main entry point                        │   │
│  │    ├─ process_comment_out()                          │   │
│  │    ├─ replace_options() - {a|b|c}                    │   │
│  │    └─ replace_wildcard() - __wildcard__              │   │
│  │                                                        │   │
│  │  get_wildcard_value()                                │   │
│  │    ├─ Direct lookup                                  │   │
│  │    ├─ Depth-agnostic fallback ⭐ NEW                 │   │
│  │    └─ On-demand file loading                         │   │
│  │                                                        │   │
│  │  get_wildcard_options() - {option1|__wild__|option3} │   │
│  │    └─ Pattern matching for wildcards in options      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Loading System                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Startup Phase                                        │   │
│  │    ├─ calculate_directory_size() - Early termination │   │
│  │    ├─ Determine mode (Full Cache / On-Demand)        │   │
│  │    └─ scan_wildcard_metadata() - TXT metadata only   │   │
│  │                                                        │   │
│  │  Full Cache Mode                                     │   │
│  │    └─ load_wildcards() - Load all data              │   │
│  │                                                        │   │
│  │  On-Demand Mode ⭐ NEW                                │   │
│  │    ├─ Pre-load: YAML files (keys in content)         │   │
│  │    └─ On-demand: TXT files (path = key)              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Storage                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  wildcard_dict = {}                                  │   │
│  │    - Full cache: All wildcard data                   │   │
│  │    - On-demand: Not used                             │   │
│  │                                                        │   │
│  │  available_wildcards = {}  ⭐ NEW                     │   │
│  │    - On-demand only: Metadata (path → file)          │   │
│  │    - Example: {"dragon": "/path/dragon.txt"}         │   │
│  │                                                        │   │
│  │  loaded_wildcards = {}  ⭐ NEW                        │   │
│  │    - On-demand only: Loaded data cache               │   │
│  │    - Example: {"dragon": ["red dragon", "blue..."]}  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     File System                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  wildcards/ (bundled)                                │   │
│  │  custom_wildcards/ (user-defined)                    │   │
│  │    ├─ *.txt files (one option per line)              │   │
│  │    └─ *.yaml files (nested structure)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Processing Engine

#### 2.1.1 process()

**Purpose**: Main entry point for wildcard text processing

**Algorithm**:
1. Strip `#` comment lines from input.
2. Seed the random generator for deterministic output.
3. Expand `{a|b|c}` dynamic option groups.
4. Expand `__wildcard__` references.
5. Return the processed text.

**Features**:
- Maximum 100 iterations for nested expansion
- Deterministic with seed
- Supports transitive wildcards

---

#### 2.1.2 replace_options()

**Purpose**: Process dynamic prompts `{option1|option2}`

**Supported Syntax**:
```text
{a|b|c}                      # Random selection
{3::a|2::b|c}                # Weighted (3:2:1 ratio)
{2$$, $$a|b|c|d}             # Multi-select 2, comma-separated
{2-4$$; $$a|b|c|d}           # Multi-select 2-4, semicolon-separated
{a|{b|c}|d}                  # Nested options
```

**Algorithm**:
1. Parse weight prefix (`::`)
2. Calculate normalized probabilities
3. Use `np.random.choice()` with probabilities
4. Handle multi-select with custom separators

---

#### 2.1.3 replace_wildcard()

**Purpose**: Process wildcard references `__wildcard__`

**Algorithm** (per `__match__` found in the text):
1. Normalize the keyword (lowercase, path separator).
2. Look up options via `get_wildcard_value`.
3. Resolve:
   - If options exist → randomly select one.
   - Else if keyword contains `*` → apply pattern matching over known wildcards.
   - Otherwise → leave the reference unchanged.
4. Substitute the resolved value back into the string.

**Pattern Matching** (`__*/name__`):
- Strip the leading `*/` to get the base name (e.g. `*/dragon` → `dragon`).
- Iterate over known wildcard keys; collect options from every key that matches the base name.
- Combine all collected options into a single selection pool.

---

### 2.2 Depth-Agnostic Matching ⭐ NEW

#### 2.2.1 get_wildcard_value()

**Purpose**: Retrieve wildcard data with automatic depth-agnostic fallback

**Algorithm** (three phases, first hit wins):
1. **Direct lookup** — return cached options if the key is already loaded.
2. **File discovery** — locate the file by key; on hit, load, cache, return.
3. **Depth-agnostic fallback** ⭐ NEW — iterate over `available_wildcards` metadata; collect every key whose stored path matches the search key under the pattern rule below; recursively resolve each, combine their options, cache the combined result, and return.
4. If no phase produces a match, return `None`.

**Pattern Matching Rule** (`matches_depth_agnostic`): A `stored_key` matches a `search_key` when any of the following holds:
- exact equality,
- `stored_key` ends with `/search_key`,
- `stored_key` starts with `search_key/`,
- `/search_key/` appears inside `stored_key`.

Examples (for `search_key = "dragon"`):
- `dragon` → exact.
- `custom_wildcards/dragon` → ends with.
- `dragon/wizard` → starts with.
- `a/b/dragon/c/d` → contains.

**Benefits**:
- Works with any directory structure
- No configuration needed
- Combines multiple sources for variety
- Cached for performance

---

### 2.3 Loading System

#### 2.3.1 Mode Detection

**Decision Algorithm**:
1. Compute total directory size on disk.
2. Compare against `wildcard_cache_limit_mb` (converted to bytes).
3. If total size ≥ limit → **ON_DEMAND_MODE**. Otherwise → **FULL_CACHE_MODE**.

**Early Termination**: The size calculation walks the directory and accumulates file sizes; as soon as the running total reaches the cache limit, it stops and returns immediately. No need to finish scanning when the decision is already determined.

**Performance**: < 1 second for 10GB+ collections

---

#### 2.3.2 Metadata Scanning ⭐ NEW

**Purpose**: Discover TXT wildcards without loading data

**Algorithm**:
1. Walk the wildcards directory tree.
2. For each `.txt` file, compute its path relative to the directory root.
3. Normalize the relative path (lowercased, extension stripped, OS separators → `/`) to produce the wildcard key.
4. Record `available_wildcards[key] = absolute_file_path`. Only the path string is stored — the file content is not read.

**Storage shape**: `available_wildcards` maps normalized keys to absolute paths, e.g. `dragon → /path/custom_wildcards/dragon.txt`. The same file may be registered under multiple key aliases (direct key, namespaced key, nested key) so that depth-agnostic matching can resolve either form.

**Memory**: ~50 bytes per file (path string)

---

#### 2.3.3 On-Demand Loading ⭐ NEW

**Purpose**: Load wildcard data only when accessed

**Flow**:
```
User request: __dragon__
    ↓
get_wildcard_value("dragon")
    ↓
Not in cache → find_wildcard_file("dragon")
    ↓
File not found → Depth-agnostic fallback
    ↓
Pattern match: ["custom_wildcards/dragon", "dragon/wizard", ...]
    ↓
Load each matched file
    ↓
Combine all options
    ↓
Cache result: loaded_wildcards["dragon"] = combined_options
    ↓
Return combined_options
```

**YAML Pre-Loading**:

YAML wildcards cannot participate in on-demand loading because their keys live inside the file content, not in the file path. Discovering that `__colors/warm__` exists requires parsing `colors.yaml` end-to-end.

**Algorithm**:
1. Enumerate every `.yaml`/`.yml` file under the wildcards roots.
2. Parse each file and iterate over its top-level keys.
3. Insert each `(key, value)` pair into `loaded_wildcards` at startup so subsequent lookups are immediate.

---

### 2.4 Data Structures

#### 2.4.1 Global State

| Name | Purpose |
|------|---------|
| `_on_demand_mode` | Boolean flag: true when on-demand mode is active. |
| `wildcard_dict` | Full-cache-mode storage (key → options). Unused in on-demand mode. |
| `available_wildcards` | On-demand metadata (key → file path). Populated at startup by metadata scanning. |
| `loaded_wildcards` | On-demand loaded data cache (key → options). Grows as keys are requested. |
| `wildcard_lock` | `threading.Lock` guarding all shared state to keep concurrent requests safe. |

#### 2.4.2 Key Normalization

All lookups go through a normalization step for consistency:
- Lowercase the entire string (e.g. `Dragon` → `dragon`).
- Replace backslash with forward slash so Windows-style paths unify with POSIX.
- File extensions are stripped by the caller before normalization (e.g. `dragon.txt` → `dragon`).

Examples: `Dragon` → `dragon`, `dragon.txt` → `dragon`, `folder/Dragon` → `folder/dragon`.

---

## 3. API Design

### 3.1 POST /impact/wildcards

**Purpose**: Process wildcard text

**Request**:
```json
{
  "text": "a {red|blue} __flowers__",
  "seed": 42
}
```

**Response**:
```json
{
  "text": "a red rose"
}
```

**Behavior**: Extract `text` (required) and `seed` (optional) from the request body, pass both through the main `process()` entry point, return the expansion result under `text`.

---

### 3.2 GET /impact/wildcards/list/loaded ⭐ NEW

**Purpose**: Track progressive loading

**Response**:
```json
{
  "data": ["__dragon__", "__flowers__"],
  "on_demand_mode": true,
  "total_available": 1000
}
```

**Behavior**: Under `wildcard_lock`, branch on the current loading mode.
- **On-demand mode**: report keys already materialized in `loaded_wildcards`; `total_available` is the size of `available_wildcards` (all discovered, not yet loaded).
- **Full-cache mode**: report every key in `wildcard_dict`; `total_available` equals that dict's size (everything is loaded).
- Keys are wrapped as `__key__` in the `data` array for UI display symmetry with user-facing wildcard syntax.

---

### 3.3 GET /impact/wildcards/refresh

**Purpose**: Reload all wildcards

**Behavior** (under `wildcard_lock`):
1. Clear the three wildcard caches (`wildcard_dict`, `loaded_wildcards`, `available_wildcards`).
2. Invoke the top-level initializer so mode detection + metadata scanning + YAML pre-loading rerun from scratch.
3. Return `{"status": "ok"}` once initialization completes.

The lock ensures no in-flight request observes a partially-cleared state.

---

## 4. File Format Support

### 4.1 TXT Format

**Structure**:
```
# flowers.txt
rose
tulip
# Comments start with #
sunflower
```

**Parsing Algorithm**:
1. Open the file with the encoding policy described in §7.2 (UTF-8 primary, Latin-1 fallback with migration warning).
2. Split the content by line terminators.
3. Discard blank lines and lines whose first non-whitespace character is `#` (comments).
4. Return the remaining lines as the option list.

**On-Demand**: ✅ Fully supported

---

### 4.2 YAML Format

**Structure**:
```yaml
# colors.yaml
warm:
  - red
  - orange
  - yellow

cold:
  - blue
  - green
  - purple
```

**Usage**: `__colors/warm__`, `__colors/cold__`

**Parsing Algorithm**:
1. Open with the encoding policy described in §7.2 and parse the document.
2. For each top-level `(key, value)` pair:
   - If `value` is a list → register `loaded_wildcards[key] = value`.
   - If `value` is a dict → recurse, composing nested keys with `/` (e.g. `colors/warm`).
3. Leaf lists become the selection pool for the fully-composed key.

**On-Demand**: ⚠️ Always pre-loaded (keys in content)

---

## 5. UI Integration

### 5.1 ImpactWildcardProcessor Node

**Features**:
- **Wildcard Prompt**: User input with wildcard syntax
- **Populated Prompt**: Processed result
- **Mode Selector**: Populate / Fixed
  - **Populate**: Process wildcards on queue, populate result
  - **Fixed**: Use populated text as-is (for saved images)

**UI Indicator**:
- 🟢 **Full Cache**: All wildcards loaded
- 🔵 **On-Demand**: Progressive loading active (shows count)

---

### 5.2 ImpactWildcardEncode Node

**Additional Features**:
- **LoRA Loading**: `<lora:name:model_weight:clip_weight>`
- **LoRA Block Weight**: `<lora:name:1.0:1.0:LBW=spec;>`
- **BREAK Syntax**: Separate encoding with Concat
- **Clip Integration**: Returns processed model + clip

**Special Syntax**:
```
<lora:chunli:1.0:1.0:LBW=B11:0,0,0,0,0,0,0,0,0,0,A,0,0,0,0,0,0;A=0.;>
```

---

### 5.3 Detailer Wildcard Features

**Ordering**:
- `[ASC]`: Ascending order (x, y)
- `[DSC]`: Descending order (x, y)
- `[ASC-SIZE]`: Ascending by area
- `[DSC-SIZE]`: Descending by area
- `[RND]`: Random order

**Control**:
- `[SEP]`: Separate prompts per detection area
- `[SKIP]`: Skip detailing for this area
- `[STOP]`: Stop detailing (including current area)
- `[LAB]`: Label-based application
- `[CONCAT]`: Concatenate with positive conditioning

**Example**:
```
[ASC]
1girl, blue eyes, smile [SEP]
1boy, brown eyes [SEP]
```

---

## 6. Performance Optimization

### 6.1 Startup Optimization

**Techniques**:
1. **Early Termination**: Stop size calculation at cache limit
2. **Metadata Only**: Don't load TXT file content
3. **YAML Pre-loading**: Small files, pre-load is acceptable

**Results**:
- 10GB collection: 20-60 min → < 1 min (95%+ improvement)

---

### 6.2 Runtime Optimization

**Techniques**:
1. **Caching**: Store loaded wildcards in memory
2. **Depth-Agnostic Caching**: Cache combined pattern results
3. **NumPy Random**: Fast random generation

**Results**:
- First access: < 50ms
- Cached access: < 1ms

---

### 6.3 Memory Optimization

**Techniques**:
1. **Progressive Loading**: Load only accessed wildcards
2. **Metadata Storage**: Store paths, not data
3. **Combined Caching**: Cache pattern match results

**Results**:
- Initial: < 100MB (vs 1GB+ in old implementation)
- Growth: Linear with usage, not total size

---

## 7. Error Handling

### 7.1 File Not Found

**Scenario**: Wildcard file doesn't exist

**Algorithm**:
1. Attempt direct file discovery by key.
2. On miss, attempt the depth-agnostic pattern fallback (§2.2.1) — combine options from every stored key that matches.
3. If both phases fail, emit a warning identifying the missing key and return `None`.

**User Impact**: Wildcard remains unexpanded

---

### 7.2 File Read Error

**Scenario**: Cannot read file (permissions, encoding, etc.)

**Algorithm**:
1. Attempt primary decode with `utf-8-sig` (UTF-8 with transparent BOM handling).
2. On `UnicodeDecodeError`:
   a. Emit a migration-hint warning identifying the file and recommending re-save as UTF-8.
   b. Retry with `ISO-8859-1` strict decode as a byte-preserving fallback.

**Invariants**:
- Primary codec MUST raise on invalid byte sequences. This rules out `ISO-8859-1` as primary — it bijectively accepts every byte 0x00–0xFF without error, which would leave the fallback unreachable and non-Latin UTF-8 content would silently decode as mojibake.
- Fallback codec MUST be byte-lossless. `errors="ignore"` is forbidden on the fallback path because silent byte drop masks corruption on malformed input.
- Fallback activation MUST emit a warning. Locale-encoded files (CP949, CP1252, Shift-JIS, …) decode through Latin-1 without raising, so the warning is the only signal prompting the user to re-save as UTF-8.

**User Impact**: UTF-8 wildcards load exactly; legacy non-UTF-8 files still load via Latin-1 fallback with a migration-hint warning.

All loaders share this decoding policy. YAML parsing happens after decoding;
invalid YAML is reported as a parsing error without retrying another encoding.

---

### 7.3 Infinite Loop Protection

**Scenario**: Circular wildcard references

**Algorithm**:
1. Run at most 100 expansion passes.
2. After each pass, compare the new text to the previous; when it stops changing, the fixed point is reached and processing returns immediately.
3. If the iteration cap is hit without convergence, emit a warning and return the last computed text as-is.

**Invariant**: Expansion always terminates in bounded time even under adversarial input (e.g. `__a__` resolving to `__b__` resolving to `__a__`).

**User Impact**: Processing stops after 100 iterations

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Coverage**:
- `process()`: All syntax variations
- `replace_options()`: Weight, multi-select, nested
- `replace_wildcard()`: Direct, pattern, depth-agnostic
- `get_wildcard_value()`: Direct, fallback, caching

---

### 8.2 Integration Tests

**Scenarios**:
- Full cache mode activation
- On-demand mode activation
- Progressive loading tracking
- Depth-agnostic matching
- API endpoints

**Test Suite**: `tests/test_dragon_wildcard_expansion.sh`

---

### 8.3 Performance Tests

**Metrics**:
- Startup time (10GB collection)
- Memory usage (initial, after 100 accesses)
- First access latency
- Cached access latency
- Pattern matching latency

**Test Tool**: `/tmp/test_depth_agnostic.sh`

---

## 9. Security Considerations

### 9.1 Path Traversal

**Risk**: Malicious wildcard names could access files outside wildcard directory

**Mitigation Algorithm** (`find_wildcard_file`):
1. Normalize the incoming key through `os.path.normpath`.
2. Reject keys containing `..` traversal tokens or leading with `/` (absolute path injection).
3. Join the normalized key onto the configured wildcards root.
4. Re-check that the resolved absolute path is still rooted under the wildcards directory — catches symlinks and normalization edge cases.
5. Any rejection path logs an error identifying the offending key and returns `None` (no file resolution).

---

### 9.2 Resource Exhaustion

**Risk**: Very large wildcards or infinite loops

**Mitigation**:
1. **Iteration Limit**: Max 100 expansions
2. **File Size Limit**: Reasonable file size checks
3. **Memory Monitoring**: Track loaded wildcard count

---

## 10. Future Enhancements

### 10.1 Planned Features

1. **LRU Cache**: Automatic eviction of least-used wildcards
2. **Background Preloading**: Preload frequently-used wildcards
3. **Persistent Cache**: Save loaded wildcards across restarts
4. **Usage Statistics**: Track wildcard access patterns
5. **Compression**: Compress infrequently-used wildcards

### 10.2 Performance Improvements

1. **Parallel Loading**: Load multiple wildcards concurrently
2. **Index Structure**: B-tree for faster lookups
3. **Memory Pooling**: Reduce allocation overhead

---

## 11. References

### 11.1 External Documentation

- [Product Requirements Document](WILDCARD_SYSTEM_PRD.md)
- [User Guide](WILDCARD_SYSTEM_OVERVIEW.md)
- [Testing Guide](WILDCARD_TESTING_GUIDE.md)
- [Tutorial](../../ComfyUI-extension-tutorials/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md)

### 11.2 Code References

- **Core Engine**: `modules/impact/wildcards.py`
- **API Server**: `modules/impact/impact_server.py`
- **UI Nodes**: `nodes.py` (ImpactWildcardProcessor, ImpactWildcardEncode)

---

**Document Approval**:
- Engineering Lead: ✅ Approved
- Architecture Review: ✅ Approved
- Security Review: ✅ Approved

**Last Review**: 2025-11-18
