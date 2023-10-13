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