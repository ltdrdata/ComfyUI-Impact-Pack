# SAM in the native MaskEditor

This integration requires a frontend exposing
`app.extensionManager.maskEditor.registerTool`. The frontend implementation is on
`feat/mask-editor-extension-api` (commit `63d786ec2d`). Checking the API, rather
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
and the existing confidence-selection logic. Each request loads its model,
computes image embeddings and the mask, and releases the model afterward.
Inference runs outside the server event loop; legacy and snapshot inference
share a lock. Already-running inference may finish after HTTP cancellation, but
its result is discarded and no model or image session is retained.

## Validation

Run the JavaScript contract tests with Node 20.19 or newer:

```bash
node --experimental-vm-modules --test tests/sam_mask_editor.test.mjs tests/preview_bridge.test.mjs
```

Backend validation and cleanup tests use the ComfyUI Python environment:

```bash
PYTHONPATH=/path/to/ComfyUI:$PWD/modules python -m pytest tests/test_sam_editor.py -q
```

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
SAM responses are never replaced with fixture masks. The tests skip only when
`IMPACT_E2E_SAM_IMAGE` is unset.
