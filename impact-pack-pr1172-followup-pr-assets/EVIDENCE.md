# Current evidence and source boundaries

Candidate `fc073de`. Exact source: `final-state.json` and `review-fix-v3/summary.json`.

| Check | Result | Record |
|---|---|---|
| JavaScript | 81 passed | `review-fix-v3/node.log` |
| Backend | 277 passed, including 128 Bridge cases | `review-fix-v3/backend-green.xml` |
| Same 128 cases on published Red | 20 failures / 108 passing controls | `review-fix-v3/backend-red.xml` |
| Native Bridge and full catalog | 53 passed | `review-fix-v3/chrome-green-core/browser.xml` |
| Real SAM and API | 15 passed | `review-fix-v3/chrome-green-models/browser.xml` |

The raw-file comparison uses the exact same test bytes on published `2b0808b` and current source. It observes file reads, selected pixels, restored masks and preview metadata. External paths are rejected before image contents are opened. Allowed paths still return the saved pixels and mask.

All eleven existing screenshot comparisons retain their original source provenance. The two selection overlays in `review-fix-v2/overlays` compare `a5350a6` with the selection fix; that mask policy remains unchanged. The other nine comparisons retain their earlier manifests. Current full Chrome checks rerun the retained scenarios, including native workflows, SAM, three-stage selections and four real restarts. New raw containment is backend behavior and has a concrete Red/Green table in the PR body.

Older frontend releases, Bridge GPU execution, diffusion sampling and deleted-file recovery remain outside validation. The separate #1172 preparation parent is unchanged.
