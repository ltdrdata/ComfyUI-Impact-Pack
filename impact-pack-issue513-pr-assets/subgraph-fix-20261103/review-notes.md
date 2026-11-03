# Review dispositions and validation boundaries

- Reviewer point 1: accepted. Fixed now pauses automatic seed changes even for choices in populated_text. Both mode/control tooltips and tests/README describe this; use populate for automatic variation. Manual/upstream edits remain authoritative.
- Point 2: accepted. The PR description is rewritten around intent, changes, tests and the remaining limitation. Three collapsed before/after queue blocks and their twelve actual output PNGs are retained because the author explicitly requested inline evidence with clickable links. Detailed journals live here instead of in the PR prose.
- Point 3: accepted. SAM device matching and Inspire isolation guidance is removed from this PR's diff. Those earlier execution notes remain in the historical full-regression triage report.
- Point 4: accepted. Transition tests observe the original HTTP prompt POST and assert seed 5/6 after timing or 6/7 before timing, plus visible seed 6/7. They do not rely on seeds 6 and 7 coincidentally producing green, or backend-normalized history seed 0.
- Point 5: addressed. Native promoted seeds, nested hosts and promoted modes now reuse core seed control and queue bookkeeping. Native link bindings are resolved per instance. Control-only stable projections preserve the core's first-queue identity when frontend projections are recreated. All original objects are returned outside the control phase. Selected policies restore before serialization and on error. No seed arithmetic/RNG is copied.

## Scope and remaining limitation

Inspire GlobalSeed is an independent upstream controller. Its explicit changes may still invalidate fixed nodes; this was accepted by the author. Manual edits and connected upstream input changes are also intentional cache invalidations. No backend parser/cache behavior, node registration, widget names/order or saved widget mapping changes.

## Diagnostic runs

All diagnostic logs are retained locally. Initial failures from the old implementation are distinct from harness failures. Four promoted-mode native comparisons failed before queueing because the frontend reordered two subgraph link records. Their full records, IDs and endpoints were identical; only list order differed. The test now compares link records by ID while retaining every field and the order of node/widget arrays. Trusted catalog fixtures remain unchanged.

One Node vm-module RGR invocation exited with SIGSEGV before producing individual test results. The log is retained as latest-unit-rgr-red-again-sigsegv.log. The unchanged complete RGR rerun produced 8 failed/17 passed, 25 passed, then the same 8 failed/17 passed. The crash is not counted as an intended Red.

An interrupted pre-normalization classic run is retained. A prematurely started replacement found the test port occupied; its logs are prefixed occupied-port-. It is excluded from validation. The driver now checks port availability before spawning. The complete final classic run uses its own verified server.

## Historical broad regression

The earlier full-regression-20261103 archive was captured on the previous direct-node implementation (4819ada, baseline 244e93e). Its broader model/SAM/Detailer checks and five baseline-reproducible legacy failures remain historical evidence. They are not described as newly rerun for the promoted-seed change. The final summary lists the actual current-source suites and model captures.

## Current model evidence

Fresh Chrome sampling is based on the rebased previous PR b1ee82d versus candidate JS SHA-256 34ca0477de4b37ae7df1e7cb6d3dbdfef089a1502fba6437ec8867b5cf9bc48b. Six distinct prompt IDs and unmodified screenshots cover three queues per revision; native Export/reload precedes the third. Candidate source is mapped to the final commit after hash verification. The installed SD1.5 checkpoint and foodphoto LoRA use 512x512 inference. No model downloads. Each corresponding independent teapot output and the initial pancake output have exact RGB parity across revisions.

Two of four corrected LRU cases initially stopped at a populate menu timeout immediately after a manual host seed edit. Their ten fixed queues, cache hits, native reload and manual seed invalidation all passed. This is retained in final-promoted-lru-retry.log. The test now uses the existing show_node helper to wait for the recreated native widget projections to redraw before interacting. No assertions or production code changed. All four promoted-mode scenarios are repeated on classic and LRU with this final test source.

The redraw-only focused runs produced classic 3 passed/1 dropdown timeout and LRU 4 passed. The unchanged complete classic 24-case run had already passed. To remove the flaky menu dependency rather than retry it, the final test uses the native mode combo left arrow (fixed immediately follows populate), explicitly waits for the actual promoted mode value to be populate, then retains the original submitted seed 6/7 or 5/6 assertions. All four focused scenarios are checked on each cache profile using this final interaction.

The single immediate arrow action removed the menu dependency but one LRU case still observed no mode change from its click. Final canvas interaction presses for 100ms and repeats the idempotent left-arrow UI action at most three times, recording each observed mode in mode-interaction.json. It must actually reach populate before any seed transition assertions execute. Queue/cache/seed assertions are never retried, skipped or relaxed. The final focused runs use the settled- prefix; intermediate arrow/dropdown runs remain diagnostic.
