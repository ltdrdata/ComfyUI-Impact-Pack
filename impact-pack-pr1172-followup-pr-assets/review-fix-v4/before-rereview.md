# Review after input-path containment

Candidate `fc073de` remains one follow-up commit above the unchanged #1172 integration `45e5321`. PR #1274 stays ready for review; #1172 must merge first, followed by rebase/retarget.

| Finding | Resolution | Verification |
|---|---|---|
| Upstream zero-mask impact was insufficiently explained | The PR body explicitly states that incoming zero masks now block when blocking is enabled. | Incoming/reuse/restore × absent/zero/partial/full × block on/off; 24 cases plus two repeated no-mask cases. |
| Confusion between sampler semantics and Bridge blocking | Retain the published condition: no available mask or zero selection blocks; an explicit full mask passes. `block=never` preserves zero masks. No new block-policy code change. | The 26 focused cases pass; four native Clear/full-selection cases repeat same input, changed input and native reload. |
| Raw fallback could open outside input | Resolve candidate and input root, check containment before reading, and reject escapes before basename fallback. Use the actual relative directory in preview metadata. | 23 new cases cover valid roots/nesting/inside symlinks, legacy references, CWD shadows, malformed paths, traversal, external symlinks and both Bridge call paths. |

Validation: JavaScript **81 passed**; backend **277 passed**, including **128 Bridge cases**; Chrome/API **68 passed** (53 native Bridge/catalog + 15 real SAM/API). Green has no failures, errors or skips. All 14 trusted catalog cases and four real isolated-server restarts pass. Runtime and test hashes match the final source; widget definitions and trusted fixtures are unchanged.

The same 128 backend cases on published `2b0808b` produce **20 failures / 108 passes**. Failures are six file-selection/metadata cases, six forbidden-read cases and eight actual Bridge external-read cases. They are not import or setup failures. Current source passes all 128.

The temporary proposal to block full masks was withdrawn before any production edit or push. Its local test run is excluded from Red/Green evidence. The authoritative Red run uses the corrected unchanged mask expectations on `2b0808b`.

No new node, widget, dependency or helper module was added. The existing raw registration function owns containment. Existing frontend registration and registered temp/output IDs retain their behavior. Earlier screenshot comparisons remain valid historical evidence; all retained scenarios were rerun on the current source.
