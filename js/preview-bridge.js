import { api } from "../../scripts/api.js";
import { getFileItem } from "./image-path.js";

// Registrations belong to the server process; a retained tab outlives that map.
let connectionVersion = 0;
api.addEventListener("reconnected", () => { connectionVersion++; });

export function isPreviewBridge(node) {
    return node.comfyClass === "PreviewBridge" || node.comfyClass === "PreviewBridgeLatent";
}

function imageValue(item) {
    return `${item.subfolder ? item.subfolder + "/" : ""}${item.filename} [${item.type || "temp"}]`;
}

export function setupPreviewBridge(node) {
    const widget = node.widgets.find(w => w.name === "image");
    node.imgs = [new Image()];
    node.imageIndex = 0;

    // Preserve the legacy initial widget value in workflows saved before a queue.
    const onAdded = node.onAdded;
    node.onAdded = function(...args) {
        const result = onAdded?.apply(this, args);
        if (!widget.value) widget.value = `$${node.id}-0`;
        return result;
    };

    // Keep the native image widget and its Vue value-store binding intact.
    // Only the API prompt needs an Impact ID; native workflows retain the
    // editor's filename in the existing image widget's serialized position.
    let registration;
    widget.serializeValue = async function() {
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
        const item = message?.images?.[0];
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
        // Workflows saved before the Vue integration contain opaque IDs.
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
