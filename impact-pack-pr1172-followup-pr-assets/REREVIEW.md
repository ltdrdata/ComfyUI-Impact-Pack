# Review after publication

Candidate `2b0808b` is one follow-up commit above the unchanged #1172 integration `45e5321`.

| Finding | Resolution | Verification |
|---|---|---|
| Latent Clear was allowed through while a full selection was blocked | Determine emptiness from effective mask presence and zero selected pixels. Keep white no-mask fallback separate from a valid full selection. | Incoming, reused and restored masks; block on/off; absent, cleared, partial and full masks. Native Clear/Invert/Save and three queue stages. |
| Raw clipspace API references lost restoration on input change | Reuse existing registration before restoration; preserve never and size checks. | Both Bridge types, saved partial/clear masks, missing files, matching and differing sizes. |
| Registration failure blocks submission | Retain intentional rejection and retry. Document the existing two-dot filename restriction separately. | Existing JS and actual browser failure/recovery tests pass. |
| Prerequisite PR remains open | Keep #1274 ready for review as requested. Document that #1172 must merge first, followed by rebase/retarget to Main. | Preparation history remains separate. |

Validation: JavaScript **81 passed**; backend **254 passed**, including **105 Bridge cases**; Chrome/API **68 passed**, including all **14 catalog cases**, **four real server restarts**, legacy SAM and API checks. No failures/errors/skips in Green. The two isolated browser groups contain 53 and 15 distinct cases, with no overlap.

The published `a5350a6` fails 20 of the same 105 backend cases (85 pass) and both new latent browser checks (both Image controls pass). Green passes those same checks. Current file hashes match both browser runtimes and test snapshots. Trusted catalog files and widget definitions remain unchanged.

Implementation remains limited to the existing raw-file registration boundary and the latent empty-mask predicate. No new helper, widget or node was introduced. The existing Clear test still preserves zero masks with blocking disabled; blocking assertions were added separately within its parameter set.

Actual captures, saved-mask hashes and three distinct execution IDs appear in `review-fix-v2/overlays/manifest.json`. Older SAM/Clear/shared/restart evidence retains its original recorded source provenance. Earlier review text is preserved in `review-fix-v2/before-rereview.md`.
