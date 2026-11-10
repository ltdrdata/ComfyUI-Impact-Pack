# Current evidence and source boundaries

Current candidate: `2b0808b`. Exact files/results: `final-state.json` and `review-fix-v2/summary.json`.

| Check | Result | Record |
|---|---|---|
| JavaScript | 81 passed | `review-fix-v2/node.log` |
| Backend | 254 passed, including 105 Bridge cases | `review-fix-v2/backend-green.xml` |
| Native Bridge and full widget catalog | 53 passed | `review-fix-v2/chrome-green-core/browser.xml` |
| Real SAM and API cases | 15 passed | `review-fix-v2/chrome-green-models/browser.xml` |
| Same backend cases on published Red | 20 intended failures / 85 passing controls | `review-fix-v2/backend-red.xml` |
| New browser selection cases on published Red | 2 latent failures / 2 passing Image controls | `review-fix-v2/chrome-red/browser.xml` |

The two new paired captures compare the same first queue after saving Clear or a full selection, with matching saved-mask hashes. Green records three distinct queues: same input, changed input, native export/reload. Blocking is checked from execution history; a retained canvas preview is not counted as a current output. Original screenshots are unchanged and annotated separately.

Nine earlier comparisons remain historical captures with their original source manifests. The complete current 68-case Chrome/API run reruns their retained behavior checks, including four server restarts. Older frontend releases, Bridge GPU execution, diffusion sampling and deleted-file recovery remain outside this validation.

The #1172 integration parent remains separate. Merge #1172 first, then rebase only the follow-up onto updated Main and retarget #1274.
