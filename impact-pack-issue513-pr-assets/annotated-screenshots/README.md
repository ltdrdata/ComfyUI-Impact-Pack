# Annotated three-queue screenshots

Six original Chrome captures from the recorded baseline/candidate sampling run.
The presentation follows the #1190 constant-text draft: separate SVG borders and
numbers plus English HTML notes over the unchanged source image. Each HTML view
has a Show annotation layer checkbox and links to its original capture, history,
visible widgets and actual 512 × 512 outputs.

- Green: patched fixed-mode controls, unchanged sampler seed and pancake output.
- Red: baseline fixed-mode branch; wildcard seed advances and sampling runs again.
- Orange: independent sampler, which still increments and renders a different teapot
  on each of the three queues. Its visible seed is already the next submission seed.

[Renderer](render.py) · [Source hashes and annotations](manifest.json) · [Verification](review.json).

The recorded candidate is `aaec21a`; its production code is identical to `4819ada`.
These are presentation updates of existing captures, not additional inference runs.
Baseline queues 2 and 3 can render identical pancake pixels despite executing again;
cache events, submitted inputs and distinct prompt IDs establish that behavior.

**Queue 1**

Baseline · Prompt `38a49aa2` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-history.json) · [Toggle annotations / evidence links](./baseline-queue-1.html)

![Baseline · Queue 1/3 — seed advances after the first render: numbered seed, fixed render and independent sampler annotations](./baseline-queue-1.png)

Patched · Prompt `692172a4` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-history.json) · [Toggle annotations / evidence links](./candidate-queue-1.html)

![Patched · Queue 1/3 — initial render: numbered seed, fixed render and independent sampler annotations](./candidate-queue-1.png)

**Queue 2**

Baseline · Prompt `0f193ad2` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-history.json) · [Toggle annotations / evidence links](./baseline-queue-2.html)

![Baseline · Queue 2/3 — fixed branch runs again: numbered seed, fixed render and independent sampler annotations](./baseline-queue-2.png)

Patched · Prompt `d9c1d9d1` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-history.json) · [Toggle annotations / evidence links](./candidate-queue-2.html)

![Patched · Queue 2/3 — fixed branch cached: numbered seed, fixed render and independent sampler annotations](./candidate-queue-2.png)

**Queue 3 — after native Export/reload**

Baseline · Prompt `2303a5d1` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-history.json) · [Toggle annotations / evidence links](./baseline-queue-3.html)

![Baseline · Queue 3/3 — fixed branch runs again after native reload: numbered seed, fixed render and independent sampler annotations](./baseline-queue-3.png)

Patched · Prompt `d7bad13b` · [Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-history.json) · [Toggle annotations / evidence links](./candidate-queue-3.html)

![Patched · Queue 3/3 — fixed branch cached after native reload: numbered seed, fixed render and independent sampler annotations](./candidate-queue-3.png)
