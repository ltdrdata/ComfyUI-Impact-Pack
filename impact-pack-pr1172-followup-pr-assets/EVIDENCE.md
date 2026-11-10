# Current evidence and source boundaries

Exact current candidate: commit `a5350a6`, `final-state.json` and `retime-prepare/followup.patch`. The latest cleanup changes only two equivalent backend fallback decisions and test code. Runtime JS, E2E actions, widget definitions and catalog fixtures are unchanged.

| Check | Result | Record |
|---|---|---|
| Current JS | 81 passed | `cleanup-review/node.log` |
| Same backend tests before/after cleanup | 200 / 200 passed; 51 Bridge | `cleanup-review/before-backend.xml`, `cleanup-review/backend.xml` |
| Current focused Chrome | 12 passed, 169.19 seconds | `cleanup-review/chrome/browser.xml` |
| Preceding full Chrome/API | 64 passed, including actual server restarts | `feedback-fixes/green-chrome/browser.xml` |
| Disabled legacy registration mutation | Two intended saved-file failures | `cleanup-review/mutations/no-legacy-registration.xml` |
| Missing file falsely accepted mutation | Two intended fallback failures | `cleanup-review/mutations/missing-file-success.xml` |

The 12 current Chrome cases cover both native restoration modes, three unedited queues with and without blocking, and shared/different-file queues with held feedback. The latest runtime/test hashes match the current worktree. The full 64-case record is explicitly attributed to the preceding source, not presented as a full run of the cleanup.

The draft keeps eight paired SAM/Clear/shared/restart overlays (`red-green/manifest.json`) and the three-queue latent comparison (`red-green/requeue-manifest.json`). They were captured before this equivalent-branch cleanup; original pixels, prompt IDs and provenance were preserved. Their E2E test file is unchanged. Images are embedded in collapsed sections; links remain outside the images.

Red sources: Main+#1172 (`6572207`) for SAM/Clear, pre-queue-fix source for shared registrations, and pre-feedback-fix candidate for unedited latent/restart. Logs are under `feedback-fixes/red-pr1172`, `red-shared` and `red-chrome`. These baselines remain distinct.

`browser_checks.py` imports an isolated worktree copy, controls only its own server and uses headless Chrome. Real SAM and VAE evidence, retained-tab restart checks and all three-queue output records remain available in the preceding full-run directory. Older frontend, GPU, diffusion sampling and deleted-temp-file recovery remain unverified. See `REREVIEW.md` for this pass's decisions; historical records are preserved under `cleanup-review/before-*`.



The current branch keeps the #1172 integration parent separate from follow-up `a5350a6`. AuthorDate and CommitDate are `2026-11-10T19:43:40+09:00`. `squash-separated/rewrite.json` records the identical reviewed tree. After #1172 merges, rebase only this follow-up onto updated Main.

Timestamp update: preparation `45e5321` and follow-up `a5350a6` both have AuthorDate/CommitDate `2026-11-10T19:43:40+09:00`. Their trees and the preparation merge parents are unchanged. The historical Red capture `6572207` has the same preparation tree; its recorded provenance is preserved.
