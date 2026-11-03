from pathlib import Path
p=Path('/mnt/teratera/git/impact-pack-issue513-pr-draft.md');s=p.read_text()
s=s.replace('Head: `6783bc5`; one local commit.','Head: `4819ada`; one local commit.')
start=s.index('Environment: ComfyUI')
end=s.index('## Results',start)
s=s[:start]+'''Environment: ComfyUI `7c4390a6`, frontend 1.57.6, Chrome 156, Python 3.12.3 and
Torch 2.10.0. The full regression uses isolated classic, LRU and cache-none servers,
CPU integrations and RTX 4080 real-model checks. Exact SAM pixel comparisons use
CPU for both the server and independent predictor. Installed models only; no downloads.
The retained foodphoto evidence uses deterministic CUDA/PyTorch attention, **SD1.5**
at **512 × 512**. Its production source is identical to this commit.

'''+s[end:]
start=s.index('Full repository regression is in progress')
end=s.index('| Unchanged fixed input',start)
s=s[:start]+'''Full regression completed on the source committed as `4819ada`. The worktree is
clean, the PR still contains one commit, and all owned validation servers are stopped.

| Check | Result |
| --- | --- |
| Python unit suites | **366 passed**; browser harness **3 passed** |
| Node suites | **20 passed** |
| Trusted catalog | **14 passed**, covering **198 node records**, positional/named restoration |
| Wildcard Chrome, classic cache | All **63 test bodies passed**, including original LBW inference; one startup error was caught at teardown and the unchanged case passed **3 retries** |
| Wildcard Chrome, LRU cache | **37/38 passed initially**; one settings-menu timeout; unchanged case passed **3 retries**, covering all **38 scenarios** |
| Switch / Mask / LoRA source Chrome | **55 / 4 / 2 passed** |
| SAM masks, tutorials, legacy editor and API | **19 passed** after matching predictor/server devices and isolating legacy-editor extensions |
| Cache and tensor integrations | **8 cache groups, 5 real-model tensor groups, 3 tutorial tensor groups**, LoRA patch/conditioning tensors and input validation passed |
| Real render / native / API scripts | LoRA/core image parity, Detailer, Switch tutorial routes, Regional sampler, AnimateDiff native restoration, full-cache/on-demand wildcard UI and UTF-8 API passed |

The **12 new edge cases passed in both classic and LRU cache**. They cover policy
edits while fixed followed by native reload/resumption, intentional upstream seed
changes, and three prompts from one Run click. New batch RGR with the same frozen
test file was **4 failed → 4 passed → the same 4 failed**, with no errors or skips.
Ruff and whitespace checks passed. Production JavaScript and both trusted workflow
fixtures remain byte-for-byte identical to the earlier validated `aaec21a` revision.

This is **not an all-green raw repository run**. Five legacy failure items reproduce
on base `244e93e`: three Python assertions and two shell scenarios. Their causes are
outdated loaded-key expectations, two separate config-module instances, eagerly loaded
YAML data, and a missing fixture in a script that prints failure but exits zero.
They are recorded as pre-existing test defects, not changed in this seed-control PR.
The two browser incidents above, the initial SAM device mismatch and the legacy SAM
assertion triggered by Inspire's deprecated UI import are also retained. Assertions
and trusted references were not relaxed; corrected runs passed.

[Full results and exact commands](impact-pack-issue513-pr-assets/full-regression-20261103/summary.json)
· [Failure diagnosis and limits](impact-pack-issue513-pr-assets/full-regression-20261103/triage.md)
· [Batch RGR](impact-pack-issue513-pr-assets/full-regression-20261103/batch-rgr.json)
· [Tested source hashes](impact-pack-issue513-pr-assets/full-regression-20261103/candidate-source.json)
· [Final commit and source verification](impact-pack-issue513-pr-assets/full-regression-20261103/commit.json).

The preceding unit review also verified **8 failed / 3 passed → 11 passed → the
same 8 failed / 3 passed**. All five injected defects were detected; earlier tests
had missed dropped fixed-mode arguments, dropped return values and restoring
`increment` instead of the selected policy.
[Unit RGR and mutation results](impact-pack-issue513-pr-assets/draft-review-20261103/unit-review.json).
The six-image sequence below is retained from `aaec21a` with identical production
code; the full regression additionally rerendered the original LBW tutorial and
required exact reference pixels plus a cached second queue.

'''+s[end:]
s=s.replace('The full node catalog was rerun against its unchanged trusted fixtures. Original\nwildcard PNGs other than LBW cover widget restoration, not complete inference.\nAPI callers', 'The full node catalog was rerun against its unchanged trusted fixtures. Original\nwildcard PNGs other than LBW cover widget restoration, not complete inference.\nAnimateDiff coverage checks assembly and native loading, not video generation.\nAPI callers')
p.write_text(s)
