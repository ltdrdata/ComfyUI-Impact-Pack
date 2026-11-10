# Current evidence and source boundaries

Candidate `30c7678`. Exact source: `final-state.json` and `review-fix-v4/summary.json`.

| Check | Result | Record |
|---|---|---|
| JavaScript | 81 passed | `review-fix-v4/node.log` |
| Backend | 304 passed, including 155 Bridge/registration cases | `review-fix-v4/backend-green.xml` |
| Same 155 cases on `fc073de` | 13 failures / 142 passing controls | `review-fix-v4/backend-red.xml` |
| Native Bridge and full catalog | 53 passed | `review-fix-v4/chrome-green-core/browser.xml` |
| Real SAM and API | 15 passed | `review-fix-v4/chrome-green-models/browser.xml` |
| Same real-server media probe | Red fails at raw symlink restoration; Green passes | `review-fix-v4/media-route-red/observations.json`, `review-fix-v4/media-route-probe/observations.json` |

Backend HTTP cases use actual aiohttp HTTP requests and the actual registration handlers. Only unused PromptServer route/onprompt registration is substituted. The additional isolated ComfyUI probe exercises its real /view, /upload/image, registration and /prompt endpoints. Green records eight prompt IDs and saved MASK outputs. Raw prompts use fresh Bridge IDs, so their output cannot come from a registered node's last-mask cache.

Eleven existing annotated screenshots retain their original source provenance. New media-path changes are documented with actual HTTP statuses, image-read/cache effects, mask sizes and pixel values in the PR's comparison table. All retained browser scenarios were rerun on current source. Older frontend versions, Bridge GPU execution, diffusion sampling and deleted-file recovery remain outside validation.
