# Promoted seed before-mode bookkeeping can restart when native widget projections change

Local issue draft only; not submitted upstream. The linked runtime comparison was captured on the pre-audit local source based on `3452e43`; it documents the separate frontend behavior, not a new execution of the final PR commit.

On frontend **1.57.6**, a native subgraph containing an Impact wildcard node in populate mode submits seed **5, 5** for two consecutive queues with increment selected and widget-control timing set to before. The expected progression is 5 for the initial skipped control, then 6. Both `ImpactWildcardProcessor` and `ImpactWildcardEncode` reproduce it.

This occurs with the prior extension implementation (`b1ee82d`) and with the local fixed-mode patch that returns native promoted widgets unchanged. The fixed-mode patch does not cause or correct this behavior.

1. Load either UI-exported native workflow linked below.
2. Select before timing in Settings → Widget control mode.
3. Keep the interior wildcard node in populate mode, seed 5 and increment selected.
4. Click Run twice and inspect the original prompt POSTs. Backend history can normalize the wildcard seed after population, so use the POST input to verify it.

[Processor native workflow](impact-pack-issue513-pr-assets/review2-20261103/core-probe-verified/candidate/ImpactWildcardProcessor/saved.json) · [Encode native workflow](impact-pack-issue513-pr-assets/review2-20261103/core-probe-verified/candidate/ImpactWildcardEncode/saved.json) · [Baseline/candidate comparison](impact-pack-issue513-pr-assets/review2-20261103/core-probe-verified/comparison.json) · [Probe and runtime logs](impact-pack-issue513-pr-assets/review2-20261103/core-probe-driver.log).

The installed bundle's `applyPromotedWidgetControl` keeps first-execution state in a WeakSet keyed by the host widget object. Its collector obtains that object through `getWidgetFromSlot`. Projection recreation therefore resets first-execution tracking; the frontend should define the intended stable binding/lifetime rather than requiring custom nodes to substitute Proxy objects. The exact projection recreation trigger still needs upstream isolation.

[Installed bundle version/hash](impact-pack-issue513-pr-assets/review2-20261103/frontend.json). The read-only frontend source context is `src/core/graph/subgraph/promotedWidgetControl.ts`; the installed bundle is authoritative. The reproducer requires Impact Pack and uses native UI construction/loading and actual Run actions. The reviewer also reported KSampler involvement; a pure KSampler reproduction was not independently executed here and is not claimed as verified.
