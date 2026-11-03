# Follow-up to #1269: validation records

The latest accepted run is the [fresh full regression of pushed head `63d28f8`](full-head-63d28f8/README.md): 575 unique pytest passes and two existing Processor Bypass xfails. All accepted results were executed again; the report records exact sources, fixture provenance, runtime setup corrections and cleanup.

- [Machine-readable report](full-head-63d28f8/validation.json)
- [All test entry points](full-head-63d28f8/test-inventory.md)
- [Reproduction and evidence guide](pr-evidence.md)
- [Draft PR #1272](https://github.com/ltdrdata/ComfyUI-Impact-Pack/pull/1272)

The four English-annotated screenshots retain their original captures and [provenance](annotated-screenshots/manifest.json). Their behavior is rechecked by fresh browser captures, server events and core cache signatures linked from the current report. Source links remain outside the images and both PR evidence sections default to collapsed.

Earlier reports remain as historical evidence: [full repository run](full-regression/README.md), [test cleanup](test-review/README.md), [targeted revalidation](reverification/README.md), and [commit split](commit-split/validation.json). Their cumulative counts are not added to or reused in the latest total.

#1269 remains open. The current validation covers the pushed branch; a later rebase onto current Main requires validation before the draft becomes ready.
