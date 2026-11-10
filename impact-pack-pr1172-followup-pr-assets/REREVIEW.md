# Review after media-route validation

Candidate `30c7678` remains one follow-up commit above `45e5321`. #1274 stays ready for review and depends on #1172 merging first.

| Finding | Resolution | Verification |
|---|---|---|
| HTTP registration accepted absolute or escaping subfolders | Check the normalized complete filename against its input/output/temp root before registration or image reads. | Nine malicious subfolder cases change from HTTP 200 to 400; no read, ID or mask-cache entry. Six filename rejection controls remain passing. |
| Raw registration rejected media-directory symlinks | Match /view and upload: normalize with abspath, preserve aliases and allow links configured below the media root. | Four raw link cases preserve actual pixels, mask and logical metadata; twelve HTTP allowed-path cases retain masks in both Bridges. |
| Core policy was described too broadly | State that this matches /view and upload, not the stricter get_annotated_filepath helper. | Checked installed ComfyUI source and actual view/upload/registration/prompt execution. |

The same 155 backend cases give **13 failures / 142 passes** on `fc073de` and **155 passes** on current source. Failures are nine subfolder escapes and four raw symlink rejection/metadata cases. The same isolated-server probe gives an 8×8 white fallback for an uncached raw symlink on Red and the saved 64×64 mask on Green. It uses distinct node IDs to exclude registered-node mask-cache fallback.

Validation: JavaScript **81 passed**, backend **304 passed** (Bridge/registration 155), Chrome/API **68 passed** (53+15), plus **one real-server media-path probe** covering eight prompt executions, six linked-file view/upload/registration cases and six rejected external registrations. Green has no failures, errors or skips. Four real server restarts and all 14 trusted catalog cases pass. Source/test hashes, unchanged widget definitions and unchanged trusted fixtures are verified.

Only the existing raw registration function and HTTP registration boundary changed. No new helper module, node, widget or dependency was added. Mask blocking remains unchanged. Configured filesystem symlinks are intentionally trusted, including links to another disk; direct request paths outside a media root remain rejected. Existing screenshot comparisons retain their recorded provenance and their scenarios were rerun.
