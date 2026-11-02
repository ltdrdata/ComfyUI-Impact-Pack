export function createSamSession(context, infer, changed = () => {}) {
    let points = [];
    let confidence = 70;
    let pointMode = "include";
    let message = "Add points, then Detect.";
    let request = null;
    let disposed = false;

    function update() {
        changed(session.state);
    }

    function cancel() {
        request?.abort();
        request = null;
    }

    function drawPoints() {
        context.setOverlay(points.map((point, index) => ({
            id: String(index), x: point.x, y: point.y, radius: 4,
            color: point.positive ? "#3b82f6" : "#ef4444"
        })));
        message = `${points.length} point${points.length === 1 ? "" : "s"}.`;
        update();
    }

    const session = {
        onPointer(event) {
            if (disposed || event.phase !== "down" || !event.inBounds || ![0, 2].includes(event.button)) return;
            cancel();
            points = [...points, { x: event.x, y: event.y, positive: event.button === 0 && pointMode === "include" }];
            drawPoints();
        },
        onDocumentChange() {
            cancel();
            points = [];
            drawPoints();
            message = "Image or mask changed. Add points to detect again.";
            update();
        },
        dispose() {
            disposed = true;
            cancel();
        },
        async detect() {
            if (disposed || request || !points.length) return;
            const snapshot = context.readImage();
            if (!snapshot) return;
            const controller = new AbortController();
            request = controller;
            const signal = AbortSignal.any([context.signal, controller.signal]);
            message = "Detecting…";
            update();
            try {
                const mask = await infer(snapshot.image, {
                    positive_points: points.filter(point => point.positive).map(point => [point.x, point.y]),
                    negative_points: points.filter(point => !point.positive).map(point => [point.x, point.y]),
                    threshold: confidence / 100
                }, { signal });
                if (signal.aborted) return;
                const result = context.commitMask(mask, { expectedRevision: snapshot.revision });
                message = result.ok ? "Mask applied." : result.reason === "stale"
                    ? "Image changed. Detect again."
                    : `Cannot apply the mask (${result.reason}).`;
            } catch (error) {
                if (!signal.aborted) message = `SAM detection failed: ${error.message}`;
            } finally {
                if (request === controller) {
                    request = null;
                    if (!disposed) update();
                }
            }
        },
        undoPoint() {
            if (disposed) return;
            cancel();
            points = points.slice(0, -1);
            drawPoints();
        },
        clear() {
            if (disposed) return;
            cancel();
            points = [];
            drawPoints();
            const snapshot = context.readImage();
            if (!snapshot) return;
            const { width, height } = snapshot.image;
            const result = context.commitMask({ width, height, data: new Uint8Array(width * height) }, {
                expectedRevision: snapshot.revision
            });
            message = result.ok ? "Points and mask cleared." : `Cannot clear the mask (${result.reason}).`;
            update();
        },
        setConfidence(value) {
            cancel();
            confidence = value;
            message = "Confidence changed. Detect to update the mask.";
            update();
        },
        setPointMode(value) {
            pointMode = value;
            update();
        },
        get state() { return { points, confidence, pointMode, busy: request !== null, message }; }
    };
    return session;
}

export async function inferSam(api, image, prompts, { signal }) {
    const canvas = new OffscreenCanvas(image.width, image.height);
    const context = canvas.getContext("2d");
    context.putImageData(image, 0, 0);
    const data = new FormData();
    data.append("image", await canvas.convertToBlob({ type: "image/png" }), "snapshot.png");
    data.append("prompts", JSON.stringify(prompts));
    const response = await api.fetchApi("/impact/sam/detect", { method: "POST", body: data, signal });
    if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.error || `Server returned HTTP ${response.status}.`);
    }
    const bitmap = await createImageBitmap(await response.blob());
    try {
        if (bitmap.width !== image.width || bitmap.height !== image.height) {
            throw new Error("The detector returned a different image size.");
        }
        context.drawImage(bitmap, 0, 0);
        const rgba = context.getImageData(0, 0, image.width, image.height).data;
        const mask = new Uint8Array(image.width * image.height);
        for (let i = 0; i < mask.length; i++) mask[i] = rgba[i * 4];
        return { width: image.width, height: image.height, data: mask };
    } finally {
        bitmap.close();
    }
}

export function createSamTool(api) {
    return {
        id: "impact.sam",
        label: "SAM Detector",
        createSession(context) {
            let render = () => {};
            const session = createSamSession(context, (...args) => inferSam(api, ...args), state => render(state));
            return {
                onPointer: session.onPointer,
                onDocumentChange: session.onDocumentChange,
                dispose: session.dispose,
                mountPanel(container) {
                    const panel = document.createElement("div");
                    panel.style.cssText = "display:flex;flex-direction:column;gap:12px;padding:12px;color:var(--input-text);";
                    const title = document.createElement("strong");
                    title.textContent = "SAM Detector";
                    const help = document.createElement("p");
                    help.textContent = "Left click to add the selected point type. Right click to exclude. Use Detect to apply a mask, then the editor's Save.";
                    const modeLabel = document.createElement("label");
                    modeLabel.textContent = "Point type ";
                    const mode = document.createElement("select");
                    for (const [value, text] of [["include", "Include"], ["exclude", "Exclude"]]) {
                        mode.add(new Option(text, value));
                    }
                    mode.onchange = () => session.setPointMode(mode.value);
                    modeLabel.append(mode);
                    const confidenceLabel = document.createElement("label");
                    confidenceLabel.textContent = "Confidence ";
                    const confidence = document.createElement("input");
                    confidence.type = "range";
                    confidence.min = "0";
                    confidence.max = "100";
                    confidence.step = "1";
                    confidence.oninput = () => session.setConfidence(Number(confidence.value));
                    const confidenceValue = document.createElement("output");
                    confidenceLabel.append(confidence, confidenceValue);
                    const status = document.createElement("output");
                    status.setAttribute("aria-label", "SAM status");
                    status.setAttribute("aria-live", "polite");
                    const button = (label, action) => {
                        const element = document.createElement("button");
                        element.type = "button";
                        element.textContent = label;
                        element.style.cssText = "padding:6px;border:1px solid var(--border-color);border-radius:4px;background:var(--comfy-input-bg);cursor:pointer;";
                        element.onclick = action;
                        return element;
                    };
                    const detect = button("Detect", () => session.detect());
                    const undo = button("Undo point", () => session.undoPoint());
                    const clear = button("Clear points and mask", () => session.clear());
                    render = state => {
                        mode.value = state.pointMode;
                        confidence.value = String(state.confidence);
                        confidenceValue.textContent = `${state.confidence}%`;
                        detect.disabled = state.busy || !state.points.length;
                        undo.disabled = !state.points.length;
                        status.textContent = state.message;
                    };
                    render(session.state);
                    panel.append(title, help, modeLabel, confidenceLabel, detect, undo, clear, status);
                    container.append(panel);
                    return () => { render = () => {}; panel.remove(); };
                }
            };
        }
    };
}
