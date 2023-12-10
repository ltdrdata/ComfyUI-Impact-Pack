import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

let original_show = app.ui.dialog.show;

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
