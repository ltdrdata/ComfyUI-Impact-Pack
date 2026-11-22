import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { getFileItem, showsImage } from "./image-path.js";

let connectionVersion = 0;
api.addEventListener("reconnected", () => { connectionVersion++; });

export function isPreviewBridge(node) {
    return node.comfyClass === "PreviewBridge" || node.comfyClass === "PreviewBridgeLatent";
}

function explainNoImage() {
    app.extensionManager.toast.add({
        severity: "warn",
        summary: "Preview Bridge",
        detail: "This Preview Bridge has no image to open. It gets one when the node itself is executed. If a run left it empty, the run did not reach it: check its input connection, a bypass, or a switch that routes around it.",
        life: 10000
    });
}

export function guardMaskEditorCommand() {
    const command = app.extensionManager.command.commands.find(c => c.id === "Comfy.MaskEditor.OpenMaskEditor");
    if (!command) return;
    const open = command.function;
    command.function = (...args) => {
        const selected = Object.values(app.canvas.selected_nodes ?? {});
        if (selected.length === 1 && isPreviewBridge(selected[0]) && !showsImage(selected[0])) return explainNoImage();
        return open(...args);
    };
}

function imageValue(item) {
    return `${item.subfolder ? item.subfolder + "/" : ""}${item.filename} [${item.type || "temp"}]`;
}

export function setupPreviewBridge(node) {
    const widget = node.widgets.find(w => w.name === "image");
    node.imgs = [new Image()];

    const getExtraMenuOptions = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(canvas, options) {
        const result = getExtraMenuOptions?.call(this, canvas, options);
        for (const option of options) {
            if (typeof option?.content !== "string" || !option.content.startsWith("Open in MaskEditor")) continue;
            const open = option.callback;
            option.callback = (...args) => showsImage(node) ? open(...args) : explainNoImage();
        }
        return result;
    };

    let shown = [];
    let selected = 0;
    Object.defineProperty(node, "imageIndex", {
        configurable: true,
        enumerable: true,
        get: () => selected,
        set: index => {
            selected = index;
            const item = shown[index ?? 0];
            if (item && shown.some(other => imageValue(other) === widget.value)) {
                widget.value = imageValue(item);
                node.properties.image = widget.value;
            }
        }
    });

    const option = node.widgets.find(w => w.name === "alpha_as_mask");
    let ranWith;
    if (option) {
        const callback = option.callback;
        option.callback = function(value, ...args) {
            const result = callback?.call(this, value, ...args);
            if (!shown.length || node.images?.[0]?.filename !== shown[0].filename) return result;
            const files = shown;
            const images = files.map(({ filename, subfolder, type }, index) => {
                const image = new Image();
                image.src = api.apiURL(value === ranWith ? `/view?${new URLSearchParams({ filename, subfolder, type })}`
                    : `/impact/view/pb_alpha_preview?${new URLSearchParams({ node_id: String(node.id), alpha_as_mask: value, index, rand: Math.random() })}`);
                return image;
            });
            Promise.all(images.map(image => image.decode())).then(() => {
                if (option.value !== value || shown !== files) return;
                node.imgs = images;
                node.graph?.setDirtyCanvas(true);
            }, () => {});
            return result;
        };
    }

    const onAdded = node.onAdded;
    node.onAdded = function(...args) {
        const result = onAdded?.apply(this, args);
        if (!widget.value) widget.value = `$${node.id}-0`;
        return result;
    };

    let registration;
    widget.serializeValue = async function() {
        if (typeof widget.value !== "string") widget.value = `$${node.id}-0`;
        const value = widget.value || `$${node.id}-0`;
        if (!registration || value !== registration.value || registration.connectionVersion !== connectionVersion) {
            const current = { value, displayedValue: value, connectionVersion };
            registration = current;
            current.promise = (async () => {
                if (typeof value === "string" && value.startsWith("$")) return value;
                const item = getFileItem("input", value);
                if (!item?.filename) throw new Error("Preview Bridge: no image to register");
                const params = new URLSearchParams({ node_id: String(node.id), ...item });
                const response = await api.fetchApi(`/impact/set/pb_id_image?${params}`, { cache: "no-store" });
                if (!response.ok) throw new Error(`Preview Bridge: image registration failed (${response.status})`);
                return await response.text();
            })().then(id => {
                current.id = id;
                return id;
            }).catch(error => {
                if (registration === current) registration = undefined;
                throw error;
            });
        }
        return await registration.promise;
    };

    const onExecuted = node.onExecuted;
    node.onExecuted = function(message, ...args) {
        const result = onExecuted?.call(this, message, ...args);
        if (message?.alpha_in_mask?.[0]) {
            app.extensionManager.toast.add({
                severity: "warn",
                summary: "Preview Bridge",
                detail: "alpha_as_mask was switched off. The mask saved in the mask editor is kept as saved, including the part of the image's transparent area it covers. Clear it in the mask editor to start over.",
                life: 10000
            });
        }
        shown = message?.images ?? [];
        ranWith = message?.alpha_as_mask?.[0];
        const item = shown[0];
        if (item && registration?.id && registration.connectionVersion === connectionVersion
            && message.source_image?.[0] === registration.id
            && widget.value === registration.displayedValue) {
            widget.value = imageValue(item);
            registration.displayedValue = widget.value;
            node.properties.image = widget.value;
        }
        return result;
    };
}

export async function restorePreviewBridge(node) {
    const widget = node.widgets.find(w => w.name === "image");
    const value = widget.value;
    let item = getFileItem("input", value);
    if (typeof value === "string" && value.startsWith("$")) {
        const response = await api.fetchApi(`/impact/get/pb_id_image?${new URLSearchParams({ id: value })}`, { cache: "no-store" });
        if (!response.ok) return;
        item = await response.json();
    }
    if (!item || widget.value !== value || !node.graph) return;
    const image = new Image();
    image.src = api.apiURL(`/view?${new URLSearchParams(item)}`);
    node.images = [item];
    node.imgs = [image];
    node.graph.setDirtyCanvas(true);
}
