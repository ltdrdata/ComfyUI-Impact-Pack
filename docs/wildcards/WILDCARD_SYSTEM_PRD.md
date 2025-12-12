# Wildcard System - Product Requirements Document

**Product**: ComfyUI Impact Pack Wildcard System
**Version**: 2.0 (Depth-Agnostic Matching)
**Status**: Released
**Last Updated**: 2025-11-18

---

## 1. Overview

### 1.1 Product Vision

The Wildcard System provides **dynamic text generation** for AI prompts, enabling users to create rich, varied prompts with minimal manual effort.

### 1.2 Target Users

- **AI Artists**: Creating varied prompts for image generation
- **Content Creators**: Generating diverse text content
- **Game Designers**: Dynamic NPC dialogue and procedural content
- **ComfyUI Users**: Workflow automation with dynamic text

---

## 2. Core Features

> **Note**: For detailed syntax examples and usage guides, see the [ImpactWildcard Tutorial](../../../ComfyUI-extension-tutorials/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md).

### 2.1 Wildcard Syntax

**Basic Wildcards**:
- `__wildcard_name__` - Simple text replacement (e.g., `__flower__` → random flower from flower.txt)
- `__category/subcategory__` - Hierarchical organization with subdirectories (e.g., `__obj/person__`)
- Transitive wildcards - Wildcards can reference other wildcards
- Case-insensitive matching - `__Jewel__` and `__jewel__` are identical
- `*` aggregation pattern (V4.15.1+) - Groups all items from path and subdirectories into one collection

**Quantifiers**:
- `N#__wildcard__` - Repeat wildcard N times
- Example: `5#__wildcards__` expands to `__wildcards__|__wildcards__|__wildcards__|__wildcards__|__wildcards__`
- Can be combined with multi-select: `{2$$, $$5#__wildcards__}`

**Comments**:
- Lines starting with `#` are treated as comments and removed
- Text following a comment is separated by single blank space from text before comment
- Example:
  ```
  first {a|b|c} second # not a comment,
  # this is a comment
  trailing text
  ```
  Becomes: `first a second # not a comment, trailing text`

**Pattern Matching**:
- `__*/wildcard__` - Depth-agnostic pattern matching at any directory level
- Automatic fallback when direct lookup fails

---

### 2.2 Dynamic Prompts

**Basic Selection**:
- `{option1|option2|option3}` - Random selection from options
- Unlimited nesting: `{a|{d|e|f}|c}` - Nested options are evaluated
- Example: `{blue apple|red {cherry|berry}|green melon}` → `blue apple`, `red cherry`, `red berry`, or `green melon`
- Complex nesting: `1{girl is holding {blue pencil|red __fruit__|colorful __flower__}|boy is riding __vehicle__}`

**Weighted Selection**:
- `{weight::option}` - Control selection probability
- **Syntax**: Weight comes FIRST, then `::`, then the option value
- **Correct**: `{10::common|1::rare}` → 10:1 ratio (≈91% vs ≈9%)
- **Incorrect**: `{common::10|rare::1}` → Will be treated as equal weights (50% vs 50%)
- Weights are normalized: `{5::red|3::green|2::blue}` → 50% red, 30% green, 20% blue
- Unweighted options default to weight 1: `{5::red|green|2::blue}` → 5:1:2 ratio

**Limitations**:
- Weights must be integers or simple decimals (e.g., `5`, `10`, `0.5`)
- Complex decimal weights may cause parsing issues due to multiselect pattern conflicts
- For decimal ratios, prefer integer equivalents: use `{5::a|3::b|2::c}` instead of `{0.5::a|0.3::b|0.2::c}`

**Multi-Select**:
- `{n$$opt1|opt2|opt3}` - Select exactly n items
- `{n1-n2$$opt1|opt2|opt3}` - Select between n1 and n2 items (excess ignored if range exceeds options)
- `{-n$$opt1|opt2|opt3}` - Select between 1 and n items
- **Custom separator**: `{n$$ separator $$opt1|opt2|opt3}`
  - Example: `{2$$ and $$red|blue|green}` → "red and blue"
  - Example: `{1-2$$ or $$apple|orange|banana}` → "apple" or "apple or orange"

---

### 2.3 ComfyUI Nodes

**ImpactWildcardProcessor**:
- **Purpose**: Browser-level wildcard processing for prompt generation
- **Dual Input Fields**:
  - Upper field: Wildcard Prompt (accepts wildcard syntax)
  - Lower field: Populated Prompt (displays generated result)
- **Mode Control**:
  - **Populate**: Processes wildcards on queue prompt, populates result (read-only)
  - **Fixed**: Ignores wildcard prompt, allows manual editing of populated prompt
- **Seed Input**:
  - Supports seed-based deterministic generation
  - Compatible seed inputs: `ImpactInt`, `Seed (rgthree)` only
  - Limitation: Reads superficial input only, does not use execution results from other nodes
- **UI Indicator**:
  - 🟢 Full Cache: All wildcards pre-loaded
  - 🔵 On-Demand: Shows count of loaded wildcards

**ImpactWildcardEncode**:
- All features of ImpactWildcardProcessor
- **LoRA Loading**: `<lora:name:model_weight:clip_weight>` syntax
  - If `clip_weight` omitted, uses same value as `model_weight`
  - All loaded LoRAs applied to both `model` and `clip` outputs
- **LoRA Block Weight (LBW)** (requires Inspire Pack):
  - Syntax: `<lora:name:model_weight:clip_weight:LBW=spec;>`
  - Use `;` as separator within spec, recommended to end with `;`
  - Specs without `A=` or `B=` → used in `Lora Loader (Block Weight)` node
  - Specs with `A=` or `B=` → parameters for `A` and `B` in loader node
  - Examples:
    - `<lora:chunli:1.0:1.0:LBW=B11:0,0,0,0,0,0,0,0,0,0,A,0,0,0,0,0,0;A=0.;>`
    - `<lora:chunli:1.0:1.0:LBW=0,0,0,0,0,0,0,0,0,0,A,B,0,0,0,0,0;A=0.5;B=0.2;>`
    - `<lora:chunli:1.0:1.0:LBW=SD-MIDD;>`
- **BREAK Syntax**: Separately encode prompts and connect using `Conditioning (Concat)`
- **Output**: Returns processed conditioning with all LoRAs applied

---

### 2.4 Detailer Integration

Special syntax for Detailer Wildcard nodes (region-specific prompt application).

**Ordering Control** (place at very beginning of prompt):
- `[ASC]` - Ascending order by (x, y) coordinates (left takes precedence, then top)
- `[DSC]` - Descending order by (x, y) coordinates
- `[ASC-SIZE]` - Ascending order by area size
- `[DSC-SIZE]` - Descending order by area size
- `[RND]` - Random order
- Example: `[ASC]\n1girl, blue eyes, smile [SEP]\n1boy, brown eyes [SEP]`

**Area Control**:
- `[SEP]` - Separator for different prompts per detection area (SEG)
- `[SKIP]` - Skip detailing for current SEG
- `[STOP]` - Stop detailing, including current SEG
- `[CONCAT]` - Concatenate wildcard conditioning with positive conditioning (instead of replacing)

**Label-Based Application**:
- `[LAB]` - Apply prompts based on labels (each label appears once)
- `[ALL]` - Prefix that applies to all labels
- Example:
  ```
  [LAB]
  [ALL] laugh, detailed eyes
  [Female] blue eyes
  [Male] brown eyes
  ```
  Female labels get: "laugh, detailed eyes, blue eyes"
  Male labels get: "laugh, detailed eyes, brown eyes"

**Complete Example**:
```
[DSC-SIZE]
sun glasses[SEP]
[SKIP][SEP]
blue glasses[SEP]
[STOP]
```
Result: Faces sorted by size descending, largest gets "sun glasses", second largest skipped, third gets "blue glasses", rest not detailed.

---

### 2.5 File Formats

**TXT Files**:
- **Format**: One option per line (comma-separated on single line = one item)
- **Comments**: Lines starting with `#` are comments
- **Encoding**: UTF-8
- **Loading**: Supports on-demand loading (loaded only when used)
- **Subfolder Support**: Use path in wildcard name (e.g., `custom_wildcards/obj/person.txt` → `__obj/person__`)
- **Example** (flower.txt):
  ```
  rose
  orchid
  iris
  carnation
  lily
  ```

**YAML Files** (V4.18.4+):
- **Format**: Nested hierarchical structure with multiple levels
- **Usage**: Keys become wildcard paths (e.g., `astronomy.Celestial-Bodies` → `__astronomy/Celestial-Bodies__`)
- **Loading**: Always pre-loaded at startup (keys exist in file content, not path)
- **Example**:
  ```yaml
  astronomy:
    Celestial-Bodies:
      - Star
      - Planet
  surface-swap:
    - swap the surfaces for
    - replace the surfaces with
  ```
- **Performance Note**: For large collections with on-demand loading, prefer TXT file structure over YAML

**Wildcard Directories**:
- Default directories: `ComfyUI-Impact-Pack/wildcards/` and `ComfyUI-Impact-Pack/custom_wildcards/`
- Recommendation: Use `custom_wildcards/` to avoid conflicts during updates
- Custom path: Configure via `impact-pack.ini` → `custom_wildcards` setting

---

### 2.6 System Features

**Progressive On-Demand Loading** ⭐:
- **Automatic Mode Detection**: System chooses optimal loading strategy based on collection size
- **Full Cache Mode** (total size < 50MB):
  - All wildcards loaded into memory at startup
  - Instant access with no load delays
  - UI Indicator: 🟢 `Select Wildcard 🟢 Full Cache`
  - Startup log: `Using full cache mode.`
- **On-Demand Mode** (total size ≥ 50MB):
  - Only metadata scanned at startup (< 1 minute for 10GB+)
  - Actual wildcard data loaded progressively as accessed
  - Low initial memory (< 100MB)
  - UI Indicator: 🔵 `Select Wildcard 🔵 On-Demand: X loaded`
  - Startup log: `Using on-demand loading mode (metadata scan only).`
- **Configuration**: Adjust threshold via `impact-pack.ini` → `wildcard_cache_limit_mb = 50`
- **File Type Behavior**:
  - TXT files: Full on-demand loading support
  - YAML files: Always pre-loaded (keys embedded in content)
- **Refresh Behavior**: Clears all cached data, re-scans directories, re-determines mode

**Depth-Agnostic Matching** ⭐:
- **Automatic Fallback**: When direct lookup fails, searches for pattern matches at any depth
- **Pattern Matching**: Finds keys that end with, start with, or contain the wildcard name
- **Multi-Source Combination**: Combines all matched wildcards into single selection pool
- **Zero Configuration**: Works automatically with any directory structure
- **Performance**: Results cached for subsequent access

**Wildcard Refresh API**:
- `GET /impact/wildcards/refresh` - Reload wildcards without restarting ComfyUI
- Clears all cached data (full cache and on-demand loaded)
- Re-scans wildcard directories
- Re-determines loading mode

**Other APIs**:
- `POST /impact/wildcards` - Process wildcard text with seed
- `GET /impact/wildcards/list` - List all available wildcards
- `GET /impact/wildcards/list/loaded` - Show currently loaded wildcards (on-demand mode)

**Deterministic Generation**:
- Seed-based random selection ensures reproducibility
- Same seed + same wildcard = same result
- Compatible with ImpactInt and Seed(rgthree) nodes

---

## 3. Requirements

### 3.1 Functional Requirements

**FR-1: Wildcard Processing**
- Support all documented syntax patterns
- Deterministic results with seed control
- Up to 100 levels of nested expansion
- Graceful error handling

**FR-2: Dynamic Prompts**
- Random, weighted, and multi-select
- Unlimited nesting depth
- Custom separators

**FR-3: Progressive Loading**
- Automatic mode detection
- On-demand loading for large collections
- Real-time tracking

**FR-4: Depth-Agnostic Matching**
- Automatic fallback pattern matching
- Combine all matched wildcards
- Support any directory structure

**FR-5: ComfyUI Integration**
- ImpactWildcardProcessor node
- ImpactWildcardEncode node with LoRA
- Detailer special syntax

---

### 3.2 Non-Functional Requirements

**NFR-1: Usability**
- Time to first success: < 5 minutes
- Zero configuration for basic use
- Clear error messages

**NFR-2: Reliability**
- 100% deterministic with same seed
- Graceful error handling
- No data loss on refresh

**NFR-3: Compatibility**
- Python 3.8+
- Windows, Linux, macOS
- Backward compatible with v1.x

**NFR-4: Scalability**
- Collections up to 100GB
- Up to 1M wildcard files
- Concurrent multi-user access

---

## 4. Configuration

**File**: `impact-pack.ini` (in ComfyUI-Impact-Pack directory)

```ini
[default]
# Custom wildcard directory (optional)
# Use this to specify additional wildcard directory path
custom_wildcards = /path/to/wildcards

# Cache size limit in MB (default: 50)
# Determines threshold for Full Cache vs On-Demand mode
wildcard_cache_limit_mb = 50
```

**Default Wildcard Directories**:
- `ComfyUI-Impact-Pack/wildcards/` - System wildcards (avoid modifying)
- `ComfyUI-Impact-Pack/custom_wildcards/` - User wildcards (recommended)
- Custom path via `custom_wildcards` setting (optional)

**Configuration Best Practices**:
- No configuration required for basic use
- Use `custom_wildcards/` to avoid conflicts during updates
- Adjust `wildcard_cache_limit_mb` based on system memory and collection size:
  - Lower limit → More likely to use on-demand mode (slower first access, lower memory)
  - Higher limit → More likely to use full cache mode (faster access, higher memory)
- For large collections (10GB+), consider organizing into subdirectories for better performance

---

## 5. User Workflows

### 5.1 Getting Started

**Goal**: First wildcard in < 5 minutes

1. Create file: `custom_wildcards/flower.txt`
2. Add content (one per line):
   ```
   rose
   orchid
   iris
   carnation
   lily
   ```
3. Use in ImpactWildcardProcessor: `a beautiful __flower__`
4. Set mode to Populate and run queue prompt
5. Result: Random selection like "a beautiful rose"

### 5.2 Reusable Prompt Templates

**Goal**: Save frequently used prompts

1. Create `custom_wildcards/ppos.txt` with:
   ```
   photorealistic:1.4, best quality:1.4
   ```
2. Use concise prompt: `__ppos__, beautiful nature`
3. Result: "photorealistic:1.4, best quality:1.4, beautiful nature"

### 5.3 Large Collections

**Goal**: Import 10GB+ seamlessly

1. Copy large wildcard collection to directory
2. Start ComfyUI (< 1 minute startup with on-demand mode)
3. Check UI indicator: 🔵 On-Demand mode active
4. Use wildcards immediately (loaded on first access)
5. Subsequent uses are cached for speed

### 5.4 LoRA + Wildcards

**Goal**: Dynamic character with LoRA

1. Create `custom_wildcards/characters.txt`:
   ```
   <lora:char1:1.0:1.0> young girl with blue dress
   <lora:char2:1.0:1.0> warrior with armor
   <lora:char3:1.0:1.0> mage with robe
   ```
2. Use ImpactWildcardEncode node
3. Prompt: `__characters__, {day|night} scene, detailed face`
4. Result: Random character with LoRA loaded + random time of day

### 5.5 Multi-Face Detailing

**Goal**: Different prompts for multiple detected faces

1. Create Detailer Wildcard prompt:
   ```
   [DSC-SIZE]
   blue eyes, smile[SEP]
   brown eyes, serious[SEP]
   green eyes, laugh
   ```
2. Result: Largest face gets "blue eyes, smile", second gets "brown eyes, serious", third gets "green eyes, laugh"

---

## 6. References

### User Documentation
- **[ImpactWildcard Tutorial](../../../ComfyUI-extension-tutorials/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md)** - Complete feature documentation

### Technical Documentation
- **[Design Document](WILDCARD_SYSTEM_DESIGN.md)** - Architecture details
- **[Testing Guide](WILDCARD_TESTING_GUIDE.md)** - Test procedures

---

## Appendix: Glossary

- **Wildcard**: Reusable text snippet (`__name__`)
- **Dynamic Prompt**: Inline options (`{a|b|c}`)
- **Pattern Matching**: Finding wildcards by partial match
- **Depth-Agnostic**: Works with any directory structure
- **On-Demand Loading**: Load data when accessed
- **LoRA**: Low-Rank Adaptation models
- **Detailer**: Node for region-specific processing

---

**Last Updated**: 2025-11-18
