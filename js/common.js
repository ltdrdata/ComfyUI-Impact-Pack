import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

let original_show = app.ui.dialog.show;

export function customAlert(message) {
	try {
		app.extensionManager.toast.addAlert(message);
	}
	catch {
		alert(message);
	}
}

export function isBeforeFrontendVersion(compareVersion) {
    try {
        const frontendVersion = window['__COMFYUI_FRONTEND_VERSION__'];
        if (typeof frontendVersion !== 'string') {
            return false;
        }

        function parseVersion(versionString) {
            const parts = versionString.split('.').map(Number);
            return parts.length === 3 && parts.every(part => !isNaN(part)) ? parts : null;
        }

        const currentVersion = parseVersion(frontendVersion);
        const comparisonVersion = parseVersion(compareVersion);

        if (!currentVersion || !comparisonVersion) {
            return false;
        }

        for (let i = 0; i < 3; i++) {
            if (currentVersion[i] > comparisonVersion[i]) {
                return false;
            } else if (currentVersion[i] < comparisonVersion[i]) {
                return true;
            }
        }

        return false;
    } catch {
        return true;
    }
}

function dialog_show_wrapper(html) {
	if (typeof html === "string") {
		if(html.includes("IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE")) {
			return;
		}

		this.textElement.innerHTML = html;
	} else {
		this.textElement.replaceChildren(html);
	}
	this.element.style.display = "flex";
}

app.ui.dialog.show = dialog_show_wrapper;


function nodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		const w = node.widgets.find((w) => event.detail.widget_name === w.name);
		if(w) {
			w.value = event.detail.value;
		}
	}
}

api.addEventListener("impact-node-feedback", nodeFeedbackHandler);


function setMuteState(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		if(event.detail.is_active)
			node.mode = 0;
		else
			node.mode = 2;
	}
}

api.addEventListener("impact-node-mute-state", setMuteState);


async function bridgeContinue(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		const mutes = new Set(event.detail.mutes);
		const actives = new Set(event.detail.actives);
		const bypasses = new Set(event.detail.bypasses);

		for(let i in app.graph._nodes_by_id) {
			let this_node = app.graph._nodes_by_id[i];
			if(mutes.has(i)) {
				this_node.mode = 2;
			}
			else if(actives.has(i)) {
				this_node.mode = 0;
			}
			else if(bypasses.has(i)) {
				this_node.mode = 4;
			}
		}

		await app.queuePrompt(0, 1);
	}
}

api.addEventListener("impact-bridge-continue", bridgeContinue);


function addQueue(event) {
	app.queuePrompt(0, 1);
}

api.addEventListener("impact-add-queue", addQueue);


function refreshPreview(event) {
	let node_id = event.detail.node_id;
	let item = event.detail.item;
	let img = new Image();
	img.src = `/view?filename=${item.filename}&subfolder=${item.subfolder}&type=${item.type}&no-cache=${Date.now()}`;
	let node = app.graph._nodes_by_id[node_id];
	if(node)
		node.imgs = [img];
}

api.addEventListener("impact-preview", refreshPreview);


// ============================================================================
// MaskRectArea Shared Utilities
// ============================================================================

/**
 * Reads a numeric value from a connected link by inspecting the origin node widget.
 * More reliable than getInputData() in ComfyUI's frontend execution model.
 *
 * @param {LGraphNode} node - LiteGraph node instance
 * @param {string} inputName - Name of the input to read
 * @returns {number|null} The numeric value or null if not available
 */
export function readLinkedNumber(node, inputName) {
    try {
        if (!node || !node.graph || !Array.isArray(node.inputs)) {
            return null;
        }
        const inp = node.inputs.find(i => i && i.name === inputName);
        if (!inp || inp.link == null) {
            return null;
        }

        const link = node.graph.links && node.graph.links[inp.link];
        if (!link) {
            return null;
        }

        const originNode = node.graph.getNodeById
            ? node.graph.getNodeById(link.origin_id)
            : null;
        if (!originNode || !Array.isArray(originNode.widgets) || originNode.widgets.length === 0) {
            return null;
        }

        const w = originNode.widgets.find(ww => ww && ww.name === "value")
            || originNode.widgets[0];
        const v = w ? w.value : null;

        return (typeof v === "number") ? v : null;
    } catch (e) {
        return null;
    }
}

/**
 * Generates a color based on percentage using HSL color space.
 *
 * @param {number} percent - Value between 0 and 1
 * @param {string} alpha - Hex alpha value (e.g., "ff", "80")
 * @returns {string} Hex color string with alpha (e.g., "#ff8040ff")
 */
export function getDrawColor(percent, alpha) {
    let h = 360 * percent;
    let s = 50;
    let l = 50;
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
        const k = (n + h / 30) % 12;
        const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        return Math.round(255 * color).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}${alpha}`;
}

/**
 * Computes and adjusts canvas size for preview widgets.
 *
 * @param {LGraphNode} node - LiteGraph node instance
 * @param {[number, number]} size - [width, height] array
 * @param {number} minHeight - Minimum canvas height (REQUIRED)
 * @param {number} minWidth - Minimum canvas width (REQUIRED)
 * @returns {void}
 */
export function computeCanvasSize(node, size, minHeight, minWidth) {
    // Validate required parameters
    if (typeof minHeight !== 'number' || typeof minWidth !== 'number') {
        console.warn('[computeCanvasSize] minHeight and minWidth are required parameters');
        return;
    }

    // Null safety check for widgets array
    if (!node.widgets?.length || node.widgets[0].last_y == null) {
        return;
    }

    // LiteGraph global availability check
    const NODE_WIDGET_HEIGHT = (typeof LiteGraph !== 'undefined' && LiteGraph.NODE_WIDGET_HEIGHT)
        ? LiteGraph.NODE_WIDGET_HEIGHT
        : 20;

    let y = node.widgets[0].last_y + 5;
    let freeSpace = size[1] - y;

    // Compute the height of all non-customCanvas widgets
    let widgetHeight = 0;
    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        if (w.type !== "customCanvas") {
            if (w.computeSize) {
                widgetHeight += w.computeSize()[1] + 4;
            } else {
                widgetHeight += NODE_WIDGET_HEIGHT + 5;
            }
        }
    }

    // Ensure there is enough vertical space
    freeSpace -= widgetHeight;

    // Clamp minimum canvas height
    if (freeSpace < minHeight) {
        freeSpace = minHeight;
    }

    // Allow both grow and shrink to fit content
    const targetHeight = y + widgetHeight + freeSpace;
    if (node.size[1] !== targetHeight) {
        node.size[1] = targetHeight;
        node.graph.setDirtyCanvas(true);
    }

    // Ensure the node width meets the minimum width requirement
    if (node.size[0] < minWidth) {
        node.size[0] = minWidth;
        node.graph.setDirtyCanvas(true);
    }

    // Position each of the widgets
    for (const w of node.widgets) {
        w.y = y;
        if (w.type === "customCanvas") {
            y += freeSpace;
        } else if (w.computeSize) {
            y += w.computeSize()[1] + 4;
        } else {
            y += NODE_WIDGET_HEIGHT + 4;
        }
    }

    node.canvasHeight = freeSpace;
}
