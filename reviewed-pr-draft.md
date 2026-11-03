# Support constant wildcard inputs and preserve list seeds after #1269

## Intent and scope

Extend [#1269](https://github.com/ltdrdata/ComfyUI-Impact-Pack/pull/1269) for `ImpactWildcardProcessor` and `ImpactWildcardEncode`: accept direct string constants during prompt submission and keep the requested seed for text produced by lists. With #1269 alone, `[{red|green|blue}, plain]` can reuse the seed-2 result when seed 5 is requested, leaving the wildcard item incorrectly unchanged.

This PR depends on #1269. The [follow-up diff](https://github.com/ltdrdata/ComfyUI-Impact-Pack/compare/609e411e13c6da94fefb07982d63b90417122412...432fd24ef0e05c42e2f865fb05575b01567097fe) is commit `432fd24`; the branch also contains #1269's commits while that PR remains open.

Population in `on_prompt` supports literal text and direct `PrimitiveString` / `PrimitiveStringMultiline` values. An arbitrary runtime `wildcard_text` input keeps its existing populated text while other nodes and hooks continue. Runtime text connected to `populated_text` retains #1269's eligible scalar cache reuse. Widget names, order, links and serialized values remain compatible.

## Changes

- **Resolve direct constants before execution.** Populate with the requested seed, then normalize the execution seed to `0` only for static text. Direct static `populated_text` also normalizes immediately, allowing requests queued before the first execution to reuse cache. Saved/UI seeds and seed links are preserved.
- **Keep list selection seed-dependent.** Check the selected upstream outputs and their ancestors before reusing a learned seed. List-producing paths, including V3 outputs, keep the requested seed so the last mapped item cannot determine reuse for the whole list. All-static lists also retain normal seed-based execution.
- **Preserve native restoration.** Synchronize named and positional population metadata without changing widgets. Reuse core `is_link` and remove unreachable checks after integer conversion.
- **Make regression failures visible.** Restrict the Bypass xfail to the known Processor case. Replace stale loader diagnostics with isolated tests for YAML preload, progressive TXT access and reloads. Require both API modes and all seven valid results before reporting parity.

## Verification

Chrome builds and connects graphs through the UI, submits real requests, and loads native exports in both positional and named restoration modes. Checks compare outputs, execution history, cache/execution events, widget names/order/values and links. The trusted catalog fixtures are unchanged.

| Requirement | Scenario and observed behavior |
|---|---|
| Reuse cold constant requests | Queue two requests while a separate Sleep workflow occupies the worker. Both use execution seed `0`, preserve source links, and the second reuses wildcard/preview results. The [before/after comparison](impact-pack-pr1269-followup-pr-assets/cold-queue-comparison.json) records the prior `731/732` execution seeds. |
| Preserve mixed-list selection | Test both nodes, list orders and restoration modes: seeds `2 → 5` produce `red → blue`, while the static item stays unchanged. Integration also submits a request between mapped calls. |
| Preserve scalar reuse | Concatenated static text reuses its learned seed; changing it to choice syntax restores requested-seed selection. Changing/opaque providers and linked seeds stay excluded. |
| Preserve Controls and saved workflows | Exercise Stop, Mute, Bypass, Switch, Remote Int, widget feedback and bounded countdown requeues; reload execution metadata and the complete native widget catalog. The existing Processor Bypass limitation is reported below. |
| Reuse and invalidate actual sampling correctly | With installed `add_detail` LoRA and the tutorial LBW vector, unchanged text reuses KSampler; edited text executes its dependent path. Core cache signatures agree with server events and image hashes. |

<details>
<summary>Mixed-list regression: #1269 before and the combined follow-up after</summary>

The source list is `[{red|green|blue}, plain]`. Seed 2 first produces `[red, plain]`. For seed 5, #1269 rewrites the execution seed to 2 and reuses the old result; the follow-up keeps seed 5 and produces `[blue, plain]`.

[![Before: seed 5 incorrectly reuses seed 2](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/list-before.png)](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/list-before.png)

Sources: [submitted request](impact-pack-pr1269-followup-pr-assets/baseline-mixed-list/seed-5-request.json) · [history](impact-pack-pr1269-followup-pr-assets/baseline-mixed-list/seed-5-history.json) · [cache/execution events](impact-pack-pr1269-followup-pr-assets/baseline-mixed-list/seed-5-events.json) · [original capture](impact-pack-pr1269-followup-pr-assets/baseline-mixed-list/seed-5.png).

[![After: seed 5 produces blue and preserves the static item](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/list-after.png)](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/list-after.png)

Sources: [submitted request](impact-pack-pr1269-followup-pr-assets/focus-mixed-list/seed-5-request.json) · [history](impact-pack-pr1269-followup-pr-assets/focus-mixed-list/seed-5-history.json) · [cache/execution events](impact-pack-pr1269-followup-pr-assets/focus-mixed-list/seed-5-events.json) · [original capture](impact-pack-pr1269-followup-pr-assets/focus-mixed-list/seed-5.png).

</details>

<details>
<summary>Actual LoRA/LBW sampling: why 732 reuses everything and 733 reuses only three nodes</summary>

The native tutorial is adapted to a cottage landscape with installed `SD1.5/etc/add_detail.safetensors`, strengths `0.7/0.7`, and its original LBW vector. A core Text (Multiline) node feeds `wildcard_text`. This checks the constant-input path with real sampling; it does not claim original-tutorial image parity.

KSampler's seed stays `944516346096519`. The requested **wildcard** seed increments, while static populated text makes its **execution** seed `0` in every row. Text A includes the LoRA/LBW tag and cottage prompt. Text B is A plus `, red flowers beside the door`.

| Requested wildcard seed | Source and populated text | Actual cached nodes | Actual executed nodes | Why |
|---|---|---|---|---|
| 731 | A | None | All eight | First execution. |
| 732 | A, unchanged | All eight | None | All execution inputs and cache keys match 731. |
| 733 | B, edited | Checkpoint `#4`, Latent `#5`, Negative CLIP `#7` | Text `#16`, Encode `#10`, KSampler `#3`, Decode `#8`, Save `#9` | The positive text changed; only its dependent path has different cache keys. |
| 734 | B, positional reload | All eight | None | Reload preserves inputs and links from 733. |
| 735 | B, named reload | All eight | None | Named restoration preserves the same execution inputs. |

[![Unchanged constant text reuses all eight nodes](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/constant-lora-same.png)](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/constant-lora-same.png)

Sources: [request 732](impact-pack-pr1269-followup-pr-assets/constant-lora/same-request.json) · [history](impact-pack-pr1269-followup-pr-assets/constant-lora/same-history.json) · [cache/execution events](impact-pack-pr1269-followup-pr-assets/constant-lora/same-events.json) · [original capture](impact-pack-pr1269-followup-pr-assets/constant-lora/same.png).

[![A text edit executes the positive sampling path](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/constant-lora-cache.png)](impact-pack-pr1269-followup-pr-assets/annotated-screenshots/constant-lora-cache.png)

Sources: [request 733](impact-pack-pr1269-followup-pr-assets/constant-lora/changed-request.json) · [history](impact-pack-pr1269-followup-pr-assets/constant-lora/changed-history.json) · [cache/execution events](impact-pack-pr1269-followup-pr-assets/constant-lora/changed-events.json) · [original capture](impact-pack-pr1269-followup-pr-assets/constant-lora/changed.png).

The [input/cache-key comparison](impact-pack-pr1269-followup-pr-assets/constant-lora/node-cache-keys.json) identifies the two direct changes at 733: Text `#16` `value` and Encode `#10` `populated_text`. It recomputes ComfyUI's actual cache signatures and verifies all four transitions against server events. The [five observations](impact-pack-pr1269-followup-pr-assets/constant-lora/result.json) also record output pixel hashes: 731=732, 733 differs, and 733=734=735. The screenshots display the **next** requested wildcard seed after increment; their annotations identify the completed request.

</details>

## Results

**576 unique pytest cases passed, with 2 existing Processor Bypass xfails.** The [acceptance report](impact-pack-pr1269-followup-pr-assets/validation.json) combines the full repository run with the corrected tests and targeted reruns; it is not a claim that all 576 cases were rerun after the test cleanup. Production code and unaffected tests match the full run.

| Pytest coverage | Accepted result |
|---|---|
| Wildcard/configuration units | [211 passed](impact-pack-pr1269-followup-pr-assets/reverification/unit.xml) |
| Browser harness | [3 passed](impact-pack-pr1269-followup-pr-assets/browser-harness.xml) |
| Chrome cache and Control combinations, Classic | [97 passed; 2 existing xfails](impact-pack-pr1269-followup-pr-assets/test-review/cache-controls.xml) |
| Native catalog: 198 nodes, 719 widgets | [14 passed](impact-pack-pr1269-followup-pr-assets/catalog/catalog.xml) |
| Native wildcard tutorials | [8 passed](impact-pack-pr1269-followup-pr-assets/catalog/tutorial-native.xml) |
| Switch execution and native restoration | [55 passed](impact-pack-pr1269-followup-pr-assets/catalog/switches.xml) |
| Additional repository regression | [188 passed](impact-pack-pr1269-followup-pr-assets/full-regression/validation.json) |

The [latest revalidation](impact-pack-pr1269-followup-pr-assets/reverification/validation.json) passed 211 unit cases in both normal and loader-first order, 10 Chrome Control cases with the same 2 xfails, [20 hook/cache/list-mapper groups](impact-pack-pr1269-followup-pr-assets/reverification/integration.json), [7 actual full-cache/on-demand API comparisons](impact-pack-pr1269-followup-pr-assets/reverification/ondemand/pair.log), and the [progressive-loading diagnostic](impact-pack-pr1269-followup-pr-assets/reverification/progressive/results.json). [Fourteen failure probes](impact-pack-pr1269-followup-pr-assets/reverification/failure-detection.json) confirm that missing/error API results and broken Encode Bypass output cannot pass. Repetitions, including the [18 Classic reruns](impact-pack-pr1269-followup-pr-assets/verified-classic/cache.xml) and [32 LRU-64 cases](impact-pack-pr1269-followup-pr-assets/verified-lru/cache.xml), are excluded from the unique total.

Separate model checks passed [5 core CLIP/LoRA tensor comparisons](impact-pack-pr1269-followup-pr-assets/cache-tensors.json) and [5 actual LoRA/LBW sampling observations](impact-pack-pr1269-followup-pr-assets/constant-lora/result.json). The hook/cache matrix uses a CLIP stand-in for Encode; the model checks use installed models. AnimateDiff coverage in the repository total is assembly and native restoration.

**Existing Bypass limitation:** Processor with only `populated_text` connected loses `PreviewAny.source` when bypassed. Both restoration modes reproduce this on [Main + #1269 before this patch](impact-pack-pr1269-followup-pr-assets/baseline-controls/bypass.xml). Encode with the same connection, and both nodes with their first text input connected, pass Bypass/reactivation. Only those two baseline Processor cases are xfail.

**Validated revision:** Main `082523b` + #1269 `609e411` + follow-up `432fd24`. The [source manifest](impact-pack-pr1269-followup-pr-assets/source.json), [environment](impact-pack-pr1269-followup-pr-assets/environment.json) and [push verification](impact-pack-pr1269-followup-pr-assets/push-record.json) bind the evidence to that combined tree. Later Main changes, including #1270, require validation after #1269 merges. [Reproduction commands and evidence guide](impact-pack-pr1269-followup-pr-assets/pr-evidence.md).
