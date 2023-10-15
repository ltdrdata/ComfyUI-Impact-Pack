import { api } from "../../scripts/api.js";

function nodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.id];
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
	let node = nodes[event.detail.id];
	if(node) {
		if(event.detail.is_active)
			node.mode = 0;
		else
			node.mode = 2;
	}
}

api.addEventListener("impact-node-mute-state", setMuteState);


function addQueue(event) {
	app.queuePrompt(0, 1);
}

api.addEventListener("impact-add-queue", addQueue);
