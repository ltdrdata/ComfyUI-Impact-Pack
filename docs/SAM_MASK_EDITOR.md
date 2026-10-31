# SAM in the native MaskEditor

This integration requires a frontend exposing
`app.extensionManager.maskEditor.registerTool`. The frontend implementation is on
`feat/mask-editor-extension-api` (validated with commit `d20e674820`). Checking the API, rather
than a version number, determines which editor integration loads. Frontends
without it retain the existing legacy SAM dialog.

1. Run the image or latent Preview Bridge, then choose **Open in MaskEditor**.
2. Select **SAM Detector** in the toolbar. Left click adds an Include point;
   right click adds an Exclude point. **Point type** also supports exclusion
   with a pen or a single mouse button.
3. Adjust **Confidence**, then click **Detect**. Each successful detection is
   one native mask Undo step. **Undo point** removes only the last prompt point.
   **Clear points and mask** preserves the paint and base image layers.
4. Refine with the native brush or eraser and use the editor's **Save**. Run the
   workflow to pass the edited image/mask or latent noise mask downstream.

SAM controls, prompt state, transport, and inference belong to Impact Pack.
The extension uses only the public MaskEditor API; native Save owns PNG alpha
encoding, image uploads, widgets, previews, and workflow serialization. Tool
switching starts a fresh SAM session. Native Undo/Redo and image transforms clear
prompt points. A cancelled or outdated result cannot overwrite a later edit.

Detection uploads a copy of the current base-image pixels to
`POST /impact/sam/detect`. This includes rotations and mirroring and excludes
the separate paint layer and point markers. Paint remains intact when applying
the inferred mask. No input/output filename or global prepared image is used.

The endpoint uses the existing `sam_editor_model` and `sam_editor_cpu` settings
and the existing confidence-selection logic. SAMLoader and both editor APIs share
the model loader; the checkpoint filename determines the architecture, regardless
of its parent directory. Each request loads its model,
computes image embeddings and the mask, and releases the model afterward.
Inference runs outside the server event loop; legacy and snapshot operations
acquire their shared lock only in worker threads. Legacy prepare returns after
the image is ready, and release clears the prepared-image cache so the same image
can be prepared again. Already-running inference may finish after HTTP
cancellation, but its result is discarded and no snapshot model or image session
is retained.

PreviewBridgeLatent also passes a restored mask into the output `noise_mask`
when upstream samples refresh. The input latent and its samples are preserved;
an incoming `noise_mask` keeps priority over the saved mask.

Preview Bridge execution feedback includes the submitted image reference in
`source_image`. The browser accepts a filename update only for the current
registration and unchanged editor value. Older queued results cannot overwrite
a newer saved edit, including one already submitted to the queue. Repeated
executions of the same image can still advance to their latest output.

## Validation

Run the JavaScript contract tests with Node 20.19 or newer:

```bash
node --experimental-vm-modules --test tests/sam_mask_editor.test.mjs tests/preview_bridge.test.mjs
```

Backend validation and cleanup tests use the ComfyUI Python environment:

```bash
PYTHONPATH=/path/to/ComfyUI:$PWD/modules python -m pytest \
  tests/test_sam_editor.py tests/test_sam_server.py \
  tests/test_preview_bridge_latent.py tests/test_preview_bridge_result.py -q
```

The HTTP tests run actual handlers in isolated subprocesses with a model double.
They verify responsiveness during slow request bodies and concurrent inference,
prepare and release requests, JSON errors and recovery, and repeated preparation.
The latent tests verify restoration policy, incoming-mask priority, and unchanged
sample tensors. Both bridge types report their submitted reference on refresh
and reuse. Chrome delays real execution messages across two saved and queued
edits, then verifies the latest mask after native workflow reload.
Chrome keeps the resized restoration and discard cases to verify
propagation through a second bridge. Model inference itself is exercised by the
Chrome tests below.

For Chrome E2E, serve a production build of the feature frontend from an
isolated ComfyUI server with both this SAM integration and the Preview Bridge
integration installed. Vite's development server disables custom extensions.
Install the configured SAM model and `vae-ft-mse-840000-ema-pruned.safetensors`
for the latent test, then run:

```bash
IMPACT_E2E_BASE_URL=http://127.0.0.1:18188 \
IMPACT_E2E_SAM_IMAGE=/path/to/subject.png \
IMPACT_E2E_ARTIFACTS=/tmp/impact-sam-e2e \
python -m pytest tests/test_e2e_preview_bridge.py -k interactive_sam -v
```

These tests capture the real SAM response, compare it with the editor and
downstream MASK, check image/paint preservation, manual editing followed by
redetection, transformed image coordinates, latent samples and downstream
`noise_mask`, and native export/import in both widget restoration modes.
They also send an invalid image through the real HTTP endpoint and retry from
the UI, hold a real SAM response while editing/switching/reopening, and verify
that cancellation preserves the later mask and saved node state. Legacy HTTP
prepare/detect/release is compared with snapshot inference and repeated after
release. The frontend companion independently checks rendered landmarks after
zoom, pan, rotation, and mirroring.
SAM responses are never replaced with fixture masks. The tests skip only when
`IMPACT_E2E_SAM_IMAGE` is unset.
