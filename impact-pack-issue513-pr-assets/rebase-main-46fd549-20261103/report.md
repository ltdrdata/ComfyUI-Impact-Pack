# Rebase on latest Main

`9cd74e5134cc441739c193e7a5ae6902ab5edb18` is one commit on `46fd5493ca0005b59ef178c40159c3d958b46c73`. Previous PR head: `057abbe915048910ab2e656f991af6a029747323`. The only conflict was the native workflow helper. Main's extra node arguments and type-sensitive positional/named saved-value checks are preserved together with the PR's promoted-widget checks and complete link-record comparison. Production JS is byte-identical to the earlier validated PR. README wording now respects Main's constant-text cache reuse.

## Current validation

Node **62**, Python **379** (211 wildcard/loader + 168 other regressions) passed. Chrome verified **299 scenarios: 297 passed, 2 pre-existing Main xfail** (86 existing cache/tutorial + 120 complete LRU + 79 new Main cache/control interactions + 14 trusted catalog checks). The catalog preserves 198 records and both trusted files are unchanged. Five integration/tensor scripts passed, including all 20 current cache integration groups and original SD1.5/SDXL tensor references. All owned servers stopped.

The first upstream run had four failures because Main's Bridge test expected fixed-mode seed increments. This PR intentionally stops those increments. Its test now requires a stable seed, cache reuse, preserved increment selection, and the existing Stop/reactivate behavior. All eight changed cases passed. The 71 unchanged cases (69 pass, 2 existing Processor Bypass xfail) plus those eight reruns cover all 79 upstream scenarios. This is not presented as one raw all-green 79-case run. [Updated assertions](control-expectations.patch) · [Main/candidate/Main RGR](bridge-rgr.json): two intended seed-increment failures, two passes, same two failures.

The same two Processor Bypass xfail were also rerun directly on both Main and candidate and remained xfail: this seed-control patch does not fix them. [Direct comparison](bypass-comparison.json) · [Follow-up note](bypass-follow-up.md). The follow-up remains a local investigation/draft item, not an externally opened issue or PR.

Fresh actual renders were recorded three times per revision against Main `46fd549`. Main advances wildcard seeds 513/514/515; the patch keeps 513 and caches queues 2–3. Independent sampler seeds 9001/9002/9003 produce distinct outputs with exact corresponding RGB parity. Queue 3 follows native reload. [Comparison](sampling/comparison.json) · [Source mapping](commit.json) · [Exact commands/results](summary.json) · [Rebase range diff](range-diff.txt).

The former full regression of `057abbe` on `56a34b9` is retained as [historical evidence](../full-regression-057abbe-20261103/report.md). Its counts and six legacy failed checks are not claimed as current-base outcomes; Main changed the loader tests. This pass reruns the affected boundaries and does not claim a fresh full SAM/mask/Switch inference run. Independent upstream GlobalSeed and the core promoted populate/before projection issue retain the PR's stated boundaries.
