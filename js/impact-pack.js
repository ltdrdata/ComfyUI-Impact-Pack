import { ComfyApp, app } from "/scripts/app.js";
import { ComfyDialog, $el } from "/scripts/ui.js";
import { api } from "/scripts/api.js";

// temporary implementation (copying from https://github.com/pythongosssss/ComfyUI-WD14-Tagger)
// I think this should be included into master!!
class ImpactProgressBadge {
	constructor() {
		if (!window.__progress_badge__) {
			window.__progress_badge__ = Symbol("__impact_progress_badge__");
		}
		this.symbol = window.__progress_badge__;
	}

	getState(node) {
		return node[this.symbol] || {};
	}

	setState(node, state) {
		node[this.symbol] = state;
		app.canvas.setDirty(true);
	}

	addStatusHandler(nodeType) {
		if (nodeType[this.symbol]?.statusTagHandler) {
			return;
		}
		if (!nodeType[this.symbol]) {
			nodeType[this.symbol] = {};
		}
		nodeType[this.symbol] = {
			statusTagHandler: true,
		};

		api.addEventListener("impact/update_status", ({ detail }) => {
			let { node, progress, text } = detail;
			const n = app.graph.getNodeById(+(node || app.runningNodeId));
			if (!n) return;
			const state = this.getState(n);
			state.status = Object.assign(state.status || {}, { progress: text ? progress : null, text: text || null });
			this.setState(n, state);
		});

		const self = this;
		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			const r = onDrawForeground?.apply?.(this, arguments);
			const state = self.getState(this);
			if (!state?.status?.text) {
				return r;
			}

			const { fgColor, bgColor, text, progress, progressColor } = { ...state.status };

			ctx.save();
			ctx.font = "12px sans-serif";
			const sz = ctx.measureText(text);
			ctx.fillStyle = bgColor || "dodgerblue";
			ctx.beginPath();
			ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
			ctx.fill();

			if (progress) {
				ctx.fillStyle = progressColor || "green";
				ctx.beginPath();
				ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, (sz.width + 12) * progress, 20, 5);
				ctx.fill();
			}

			ctx.fillStyle = fgColor || "#fff";
			ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
			ctx.restore();
			return r;
		};
	}
}

const input_tracking = {};
const input_dirty = {};
const output_tracking = {};

function progressExecuteHandler(event) {
	if(event.detail.output.aux){
		const id = event.detail.node;
		if(input_tracking.hasOwnProperty(id)) {
			if(input_tracking.hasOwnProperty(id) && input_tracking[id][0] != event.detail.output.aux[0]) {
				input_dirty[id] = true;
			}
			else{

			}
		}

		input_tracking[id] = event.detail.output.aux;
	}
}

function imgSendHandler(event) {
	if(event.detail.images.length > 0){
		let data = event.detail.images[0];
		let filename = `${data.filename} [${data.type}]`;

		let nodes = app.graph._nodes;
		for(let i in nodes) {
			if(nodes[i].type == 'ImageReceiver') {
				if(nodes[i].widgets[1].value == event.detail.link_id) {
					nodes[i].widgets[0].value = filename;
					let img = new Image();
					img.src = `/view?filename=${data.filename}&type=${data.type}&subfolder=${data.subfolder}`+app.getPreviewFormatParam();
					nodes[i].imgs = [img];
					nodes[i].size[1] = Math.max(200, nodes[i].size[1]);
				}
			}
		}
	}
}

var progressEventRegistered = false;
var imgSendEventRegistered = false;
const impactProgressBadge = new ImpactProgressBadge();

app.registerExtension({
	name: "Comfy.Impack",
	loadedGraphNode(node, app) {
		if (node.comfyClass == "PreviewBridge" || node.comfyClass == "MaskPainter") {
			if (!progressEventRegistered) {
				api.addEventListener("executed", progressExecuteHandler);
				progressEventRegistered = true;
			}

			input_dirty[node.id + ""] = true;
		}

		if (node.comfyClass == "ImageSender") {
			if (!imgSendEventRegistered) {
				api.addEventListener("img-send", imgSendHandler);
				imgSendEventRegistered = true;
			}
		}
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name == "IterativeLatentUpscale" || nodeData.name == "IterativeImageUpscale") {
			impactProgressBadge.addStatusHandler(nodeType);
		}
	},

	nodeCreated(node, app) {
		if(node.comfyClass == "MaskPainter") {
			node.addWidget("button", "Edit mask", null, () => {
				ComfyApp.copyToClipspace(node);
				ComfyApp.clipspace_return_node = node;
				ComfyApp.open_maskeditor();
			});
		}
		if (node.comfyClass == "PreviewBridge" || node.comfyClass == "MaskPainter") {
			node.widgets[0].value = '#placeholder';

			Object.defineProperty(node, "images", {
				set: function(value) {
					node._images = value;
				},
				get: function() {
					const id = node.id+"";
					if(node.widgets[0].value != '#placeholder') {
						var need_invalidate = false;

						if(input_dirty.hasOwnProperty(id) && input_dirty[id]) {
							node.widgets[0].value = {...input_tracking[id][1]};
							input_dirty[id] = false;
							need_invalidate = true
						}

						node.widgets[0].value['image_hash'] = app.nodeOutputs[id]['aux'][0];
						node.widgets[0].value['forward_filename'] = app.nodeOutputs[id]['aux'][1][0]['filename'];
						node.widgets[0].value['forward_subfolder'] = app.nodeOutputs[id]['aux'][1][0]['subfolder'];
						node.widgets[0].value['forward_type'] = app.nodeOutputs[id]['aux'][1][0]['type'];
						app.nodeOutputs[id].images = [node.widgets[0].value];

						if(need_invalidate) {
							Promise.all(
								app.nodeOutputs[id].images.map((src) => {
									return new Promise((r) => {
										const img = new Image();
										img.onload = () => r(img);
										img.onerror = () => r(null);
										img.src = "/view?" + new URLSearchParams(src[0]).toString();
										console.log(`new img => ${img.src}`);
									});
								})
							).then((imgs) => {
								this.imgs = imgs.filter(Boolean);
								this.setSizeForImage?.();
								app.graph.setDirtyCanvas(true);
							});
						}

						return app.nodeOutputs[id].images;
					}
					else {
						return node._images;
					}
				}
			});
		}
	}
});