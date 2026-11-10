# Final draft-based review

Target: `fix/preview-bridge-after-pr1172`, separate Main + #1172 integration base `45e5321`, commit `a5350a6` in `retime-prepare/followup.patch` (identical to the reviewed working tree). This review covers the draft's claims against the complete PR diff; only three files changed in this pass. Exact hashes are in `final-state.json`.

| Finding | Action | Verification |
|---|---|---|
| Artificial concurrency in the registration test | Removed the ThreadPoolExecutor/PIL Barrier test and imports. The actual HTTP registration handler calls the function without yielding; the test invented a thread interleaving outside this path. | Real queued-edit tests still submit multiple registrations before execution and detect masks being overwritten. |
| Test helper duplicates built-in mock facilities | Replaced the separate request array/wrapper with Node's mock call count/arguments; removed unused ComfyApp and serialization stubs and redundant Promise wrappers. | All 81 JS cases pass with the same behavioral assertions. |
| Unedited backend requeue registered a new ID every time | Register the first feedback filename once, then keep that ID for the third queue, as the frontend does. | Both no-mask output and blocking pass through three calls; unchanged E2E also covers three queues. |
| Nested legacy fallback decisions repeated in both Bridges | Collapse only those two decisions to the equivalent condition. Keep existing path lookup behavior and Image/Latent mask semantics. | Same 200 backend cases pass before/after; 12 current Chrome cases pass. |
| Legacy compatibility claim lacked a direct saved/missing-file check | Add four cases: both Bridges × saved/missing raw clipspace reference, including real preview-file existence after fallback. | Disabling legacy registration fails the two saved cases; falsely accepting a missing file fails the two missing cases. |
| Draft repeated validation details and treated the previous source as current | Trim duplicated prose and distinguish current 81/200/12 results from the preceding full 64-case run. Keep original Red/Green images and source manifests intact. | Links, hashes, unchanged E2E file and trusted fixtures checked. |

No additional functional defect was confirmed in this review. Product changes in this pass are limited to the two equivalent fallback conditions; frontend runtime code, mask extraction and restoration policies did not change. Native positional/named reload passes, and INPUT_TYPES plus trusted catalog snapshots are unchanged.

Kept deliberately: the reconnect generation (one API listener, no per-node listeners), registration failure cleanup, late-result/restoration guards, legacy pre-queue widget initialization, device/type cache checks and native workflow tests. These protect observed lifecycle or compatibility boundaries. The filename parser is shared by Bridge and ImageReceiver; URLSearchParams, UUID and Node's mock API supply the existing standard primitives. Further helpers or a new abstraction for the two short mask-restoration branches would add indirection without eliminating a distinct responsibility.

Validation: JS **81 passed**; backend **200 passed** (51 Bridge), with the same tests passing on both sides of the cleanup; current focused Chrome **12 passed**, 169.19 seconds. All normal-code runs have zero failures/errors/skips. The full **64 passed** record belongs to the preceding behavior-fix source (`feedback-fixes/green-chrome`); it was not rerun for this equivalent-branch cleanup. Mutations were isolated under `cleanup-review/mutations`, never installed into the worktree or test server.

Historical review and result records are preserved in `cleanup-review/before-REREVIEW.md`, `cleanup-review/before-final-state.json` and earlier review directories. GPU, older frontend releases, diffusion sampling and recovery of deleted temporary files remain outside the verified scope.


Current integration plan: merge #1172 first, update Main, then rebase only the one follow-up commit from `45e5321` onto that Main. Dependency history remains separate. Both follow-up commit timestamps were renewed without changing the reviewed file tree.

Timestamp update: preparation `45e5321` and follow-up `a5350a6` both have AuthorDate/CommitDate `2026-11-10T19:43:40+09:00`. Their trees and the preparation merge parents are unchanged. The historical Red capture `6572207` has the same preparation tree; its recorded provenance is preserved.
