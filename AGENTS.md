# Workflow compatibility

- Existing widget names and order must not change. This is a fundamental project
  rule and applies to fixes, refactoring, and merged changes.
- Preserve the mapping between each widget and its serialized `widgets_values`
  entry. Do not rename, reorder, or insert widgets in a way that shifts existing
  saved values to different controls.
- Fix behavior without changing the existing widget layout or names. Review
  incoming merges for the same compatibility requirement.
- Relevant regression tests must check widget names, order, and saved values
  before and after loading a native workflow.

# Adding nodes to the validation catalog

- Before starting a node addition, run `tests/test_e2e_widget_catalog.py` with
  pytest against the existing trusted workflow and snapshot. Both positional
  and named restoration must pass before changing either fixture.
- Treat `tests/workflows/all-impact-nodes.widgets.json` as the fixed reference.
  A workflow loaded by modified or broken code is not a new source of truth.
  If existing checks fail, fix the code first; never regenerate, delete, or
  relax the snapshot to make them pass. Restore missing fixtures from a trusted
  revision instead of rebuilding them from the current runtime.
- Every new registered node type, including aliases and nodes without widgets,
  must be added exactly once to `tests/workflows/all-impact-nodes.json` in the
  same change as the node implementation.
- Extend the existing workflow through Chrome E2E node search and native Export.
  Do not construct or edit its JSON by hand or create nodes through graph APIs.
  Preserve existing nodes and snapshot records; append the new node's rendered
  widget name/value pairs in order, or an empty list when it has no widgets.
- The recorder must verify existing nodes against the unchanged snapshot before
  adding nodes, then verify the complete candidate in both restoration modes
  before replacing either fixture. Run the full catalog E2E after the update.
  See `docs/E2E_TEST_STRATEGY.md` for commands. This covers loading and widget
  compatibility; add behavior tests appropriate to the new node separately.
