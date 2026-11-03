# Three-queue screenshots and actual outputs

Expand a queue to compare baseline and patched captures. Each block contains
clickable links to the original screenshot, execution history and widget state,
plus the actual 512 × 512 pancake and teapot output PNGs. Click an image to open it
at full size. Evidence links are outside the raster overlay.

Green identifies the patched fixed branch; red identifies the baseline branch
that still reruns; orange identifies the independent incrementing sampler.
The recorded candidate `aaec21a` has identical production code to `4819ada`.
These are annotations of existing runs, with no additional inference.

[Renderer](render.py) · [Source hashes](manifest.json) · [Verification](review.json).

<details>
<summary>Queue 1/3 — initial render: baseline vs patched screenshots and actual output PNGs</summary>

**Baseline · Prompt `38a49aa2-678e-4db9-b819-274de7785050`**

[![Baseline · Queue 1/3 — seed advances after the first render](./baseline-queue-1.png)](./baseline-queue-1.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Baseline queue 1: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-fixed-output.png) | [![Baseline queue 1: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-1-independent-output.png)

**Patched · Prompt `692172a4-78fd-426b-8aac-0780cadf9838`**

[![Patched · Queue 1/3 — initial render](./candidate-queue-1.png)](./candidate-queue-1.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Patched queue 1: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-fixed-output.png) | [![Patched queue 1: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-1-independent-output.png)

</details>

<details>
<summary>Queue 2/3 — repeat: baseline vs patched screenshots and actual output PNGs</summary>

**Baseline · Prompt `0f193ad2-5548-4825-a285-9bbdd62f0f19`**

[![Baseline · Queue 2/3 — fixed branch runs again](./baseline-queue-2.png)](./baseline-queue-2.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Baseline queue 2: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-fixed-output.png) | [![Baseline queue 2: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-2-independent-output.png)

**Patched · Prompt `d9c1d9d1-3489-415a-8842-688fe8033a86`**

[![Patched · Queue 2/3 — fixed branch cached](./candidate-queue-2.png)](./candidate-queue-2.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Patched queue 2: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-fixed-output.png) | [![Patched queue 2: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-2-independent-output.png)

</details>

<details>
<summary>Queue 3/3 — after native Export/reload: baseline vs patched screenshots and actual output PNGs</summary>

**Baseline · Prompt `2303a5d1-f9f0-4236-a3e1-4ac2bd867408`**

[![Baseline · Queue 3/3 — fixed branch runs again after native reload](./baseline-queue-3.png)](./baseline-queue-3.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Baseline queue 3: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-fixed-output.png) | [![Baseline queue 3: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/baseline/queue-3-independent-output.png)

**Patched · Prompt `d7bad13b-a736-41f1-b6c7-7b6e0c0e878e`**

[![Patched · Queue 3/3 — fixed branch cached after native reload](./candidate-queue-3.png)](./candidate-queue-3.png)

[Original capture](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-workflow.png) · [Queue history](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-history.json) · [Visible widget values](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-widget-state.json)

| Actual fixed wildcard output — pancakes | Actual independent sampler output — teapot |
| --- | --- |
| [![Patched queue 3: pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-fixed-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-fixed-output.png) | [![Patched queue 3: teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-independent-output.png)](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-independent-output.png) |

[Open full-size pancake PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-fixed-output.png) · [Open full-size teapot PNG](../fixed-mode/seed-refresh/independent-sampler/candidate/queue-3-independent-output.png)

</details>
