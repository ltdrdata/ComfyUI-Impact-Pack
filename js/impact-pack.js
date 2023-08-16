import { ComfyApp, app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

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
					if(data.subfolder)
						nodes[i].widgets[0].value = `${data.subfolder}/${data.filename} [${data.type}]`;
					else
						nodes[i].widgets[0].value = `${data.filename} [${data.type}]`;

					let img = new Image();
					img.src = `/view?filename=${data.filename}&type=${data.type}&subfolder=${data.subfolder}`+app.getPreviewFormatParam();
					nodes[i].imgs = [img];
					nodes[i].size[1] = Math.max(200, nodes[i].size[1]);
				}
			}
		}
	}
}


function latentSendHandler(event) {
	if(event.detail.images.length > 0){
		let data = event.detail.images[0];
		let filename = `${data.filename} [${data.type}]`;

		let nodes = app.graph._nodes;
		for(let i in nodes) {
			if(nodes[i].type == 'LatentReceiver') {
				if(nodes[i].widgets[1].value == event.detail.link_id) {
					if(data.subfolder)
						nodes[i].widgets[0].value = `${data.subfolder}/${data.filename} [${data.type}]`;
					else
						nodes[i].widgets[0].value = `${data.filename} [${data.type}]`;

					let img = new Image();
					img.src = `/view?filename=${data.filename}&type=${data.type}&subfolder=${data.subfolder}`+app.getPreviewFormatParam();
					nodes[i].imgs = [img];
					nodes[i].size[1] = Math.max(200, nodes[i].size[1]);
				}
			}
		}
	}
}


function valueSendHandler(event) {
    let nodes = app.graph._nodes;
    for(let i in nodes) {
        if(nodes[i].type == 'ImpactValueReceiver') {
            if(nodes[i].widgets[2].value == event.detail.link_id) {
                nodes[i].widgets[1].value = event.detail.value;

                let typ = typeof event.detail.value;
                if(typ != "number") {
                    nodes[i].widgets[0].value = typeof event.detail.value;
                }
                else if(Number.isInteger(event.detail.value)) {
                    nodes[i].widgets[0].value = "INT";
                }
                else {
                    nodes[i].widgets[0].value = "FLOAT";
                }
            }
        }
    }
}


const impactProgressBadge = new ImpactProgressBadge();

api.addEventListener("stop-iteration", () => {
    document.getElementById("autoQueueCheckbox").checked = false;
});
api.addEventListener("value-send", valueSendHandler);
api.addEventListener("img-send", imgSendHandler);
api.addEventListener("latent-send", latentSendHandler);
api.addEventListener("executed", progressExecuteHandler);

app.registerExtension({
	name: "Comfy.Impack",
	loadedGraphNode(node, app) {
		if (node.comfyClass == "PreviewBridge" || node.comfyClass == "MaskPainter") {
			input_dirty[node.id + ""] = true;
		}
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name == "IterativeLatentUpscale" || nodeData.name == "IterativeImageUpscale") {
			impactProgressBadge.addStatusHandler(nodeType);
		}

        if (nodeData.name === 'ImpactMakeImageList') {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (!connected && this.inputs.length > 1) {
                    if (this.widgets) {
                        const w = this.widgets.find((w) => w.name === this.inputs[index].name)
                        if (w) {
                            w.onRemoved?.()
                            this.widgets.length = this.widgets.length - 1
                        }
                    }
                    this.removeInput(index);
                }

                for (let i = 0; i < this.inputs.length; i++) {
                    this.inputs[i].label = `image${i + 1}`
                    this.inputs[i].name = `image${i + 1}`
                }

                if (this.inputs[this.inputs.length - 1].link != undefined) {
                    this.addInput(`image${this.inputs.length + 1}`, 'IMAGE');
                }
            }
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

		switch(node.comfyClass) {
		    case "ToDetailerPipe":
		    case "BasicPipeToDetailerPipe":
		    case "EditDetailerPipe":
		    case "FaceDetailer":
		    case "DetailerForEach":
		    case "DetailerForEachDebug":
		    case "DetailerForEachPipe":
		    case "DetailerForEachDebugPipe":
		        {
                    for(let i in node.widgets) {
                        let widget = node.widgets[i];
                        if(widget.type === "customtext") {
                            widget.dynamicPrompts = false;
                            widget.inputEl.placeholder = "wildcard spec: if kept empty, this option will be ignored";
                            widget.serializeValue = () => {
                                return node.widgets[i].value;
                            };
                        }
			        }
			    }
		        break;
		}

		if(node.comfyClass == "ImpactSEGSLabelFilter") {
			Object.defineProperty(node.widgets[0], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(node.widgets[1].value.trim() != "" && !node.widgets[1].value.trim().endsWith(","))
                                node.widgets[1].value += ", "

                            node.widgets[1].value += value;
                            node.widgets_values[1] = node.widgets[1].value;
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}

		if(node.comfyClass == "ImpactWildcardEncode") {
			node._value = "Select the LoRA to add to the text";

			Object.defineProperty(node.widgets[3], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Select the LoRA to add to the text") {
	                            let lora_name = value;
	                            if (lora_name.endsWith('.safetensors')) {
	                                lora_name = lora_name.slice(0, -12);
	                            }

	                            node.widgets[0].value += `<lora:${lora_name}>`;
	                            node.widgets_values[0] = node.widgets[0].value;
                            }
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[3].serializeValue = () => { return "Select the LoRA to add to the text"; }
		}

		if(node.comfyClass == "ImpactWildcardProcessor" || node.comfyClass == "ImpactWildcardEncode") {
			node.widgets[0].inputEl.placeholder = "Wildcard Prompt (User input)";
			node.widgets[1].inputEl.placeholder = "Populated Prompt (Will be generated automatically)";
			node.widgets[1].inputEl.disabled = true;
			node.widgets[0].dynamicPrompts = false;
			node.widgets[1].dynamicPrompts = false;

            let populate_getter = node.widgets[1].__lookupGetter__('value');
            let populate_setter = node.widgets[1].__lookupSetter__('value');

			let force_serializeValue = async (n,i) =>
				{
					if(!node.widgets[2].value) {
						return node.widgets[1].value;
					}
					else {
				        let wildcard_text = await node.widgets[0].serializeValue();

						let response = await api.fetchApi(`/impact/wildcards`, {
																method: 'POST',
																headers: { 'Content-Type': 'application/json' },
																body: JSON.stringify({text: wildcard_text})
															});

						let populated = await response.json();

						n.widgets_values[2] = false;
						n.widgets_values[1] = populated.text;
						populate_setter.call(node.widgets[1], populated.text);

						return populated.text;
					}
				};

			// mode combo
			Object.defineProperty(node.widgets[2], "value", {
				set: (value) => {
						node._mode_value = value == true || value == "Populate";
						node.widgets[1].inputEl.disabled = value == true || value == "Populate";
					},
				get: () => {
						if(node._mode_value != undefined)
							return node._mode_value;
						else
							return true;
					 }
			});

            // to avoid conflict with presetText.js of pythongosssss
			Object.defineProperty(node.widgets[1], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(!stackTrace.includes('serializeValue'))
				            populate_setter.call(node.widgets[1], value);
					},
				get: () => {
				        return populate_getter.call(node.widgets[1]);
					 }
			});

            node.widgets[0].serializeValue = (n,i) => {
                if(node.inputs) {
	                let link_id = node.inputs.find(x => x.name=="wildcard_text")?.link;
	                if(link_id != undefined) {
	                    let link = app.graph.links[link_id];
	                    let input_widget = app.graph._nodes_by_id[link.origin_id].widgets[link.origin_slot];
	                    if(input_widget.type == "customtext") {
	                        return input_widget.value;
	                    }
	                }
	                else {
	                    return node.widgets[0].value;
	                }
                }
                else {
                    return node.widgets[0].value;
                }
            };
            node.widgets[1].serializeValue = force_serializeValue;
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
							this._images = app.nodeOutputs[id].images;
						}

						let filename = app.nodeOutputs[id]['aux'][1][0]['filename'];
						let subfolder = app.nodeOutputs[id]['aux'][1][0]['subfolder'];
						let type = app.nodeOutputs[id]['aux'][1][0]['type'];

						let item =
							{
								image_hash: app.nodeOutputs[id]['aux'][0],
								forward_filename: app.nodeOutputs[id]['aux'][1][0]['filename'],
								forward_subfolder: app.nodeOutputs[id]['aux'][1][0]['subfolder'],
								forward_type: app.nodeOutputs[id]['aux'][1][0]['type']
							};

						app.nodeOutputs[id].images = [{
								...node._images[0],
								...item
							}];

						node.widgets[0].value =
							{
								...node._images[0],
								...item
							};

						if(need_invalidate) {
							Promise.all(
								app.nodeOutputs[id].images.map((src) => {
									return new Promise((r) => {
										const img = new Image();
										img.onload = () => r(img);
										img.onerror = () => r(null);
										img.src = "/view?" + new URLSearchParams(src).toString();
									});
								})
							).then((imgs) => {
								this.imgs = imgs.filter(Boolean);
								this.setSizeForImage?.();
								app.graph.setDirtyCanvas(true);
							});

							app.nodeOutputs[id].images[0] = { ...node.widgets[0].value };
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
