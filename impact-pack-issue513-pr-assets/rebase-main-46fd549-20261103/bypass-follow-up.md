# Follow-up candidate: preserve Processor output through Bypass

After completing PR #1271, investigate this separate Main failure. No follow-up issue or PR has been opened.

`test_control_bridge_mode_change_requeues[Bypass-populated-only-ImpactWildcardProcessor-positional]` and the corresponding named case reach native Bypass with only `populated_text` connected. The resulting API graph drops `PreviewAny.source`, leaving `inputs={}` and no preview output. Both Main and the seed-control candidate reproduce the two existing xfail; see [comparison](bypass-comparison.json).

Expected behavior: Bypass should forward the connected text to PreviewAny, then reactivate normally. Preserve existing widget names, widget order, serialized values and link identities. Keep named/positional Export/reload tests and the Mute/Bypass-connected-first-input controls.

Inspect the frontend's Bypass input/output resolution and its handling of an unconnected first compatible input before choosing the repository and implementation. A core fix may be appropriate; this note does not claim a confirmed root cause. Reordering widgets or inputs as a workaround is not the proposed fix. Reproduce Main failure, fix the actual resolution path, then require both current xfail cases to pass without the xfail condition.
