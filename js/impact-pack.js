import { ComfyApp, app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

let wildcards_list = [];
async function load_wildcards() {
	let res = await api.fetchApi('/impact/wildcards/list');
	let data = await res.json();
	wildcards_list = data.data;
}

load_wildcards();

export function get_wildcards_list() {
	return wildcards_list;
}

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
				let is_linked = false;

				if(nodes[i].widgets[1].type == 'converted-widget') {
					for(let j in nodes[i].inputs) {
						let input = nodes[i].inputs[j];
						if(input.name === 'link_id') {
							if(input.link) {
								let src_node = app.graph._nodes_by_id[app.graph.links[input.link].origin_id];
								if(src_node.type == 'ImpactInt' || src_node.type == 'PrimitiveNode') {
									is_linked = true;
								}
							}
							break;
						}
					}
				}
				else if(nodes[i].widgets[1].value == event.detail.link_id) {
					is_linked = true;
				}

				if(is_linked) {
					if(data.subfolder)
						nodes[i].widgets[0].value = `${data.subfolder}/${data.filename} [${data.type}]`;
					else
						nodes[i].widgets[0].value = `${data.filename} [${data.type}]`;

					let img = new Image();
					img.onload = (event) => {
						nodes[i].imgs = [img];
						nodes[i].size[1] = Math.max(200, nodes[i].size[1]);
						app.canvas.setDirty(true);
					};
					img.src = `/view?filename=${data.filename}&type=${data.type}&subfolder=${data.subfolder}`+app.getPreviewFormatParam();
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
				if(typ == 'string') {
					nodes[i].widgets[0].value = "STRING";
				}
				else if(typ == "boolean") {
					nodes[i].widgets[0].value = "BOOLEAN";
				}
				else if(typ != "number") {
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
		if (node.comfyClass == "MaskPainter") {
			input_dirty[node.id + ""] = true;
		}
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name == "IterativeLatentUpscale" || nodeData.name == "IterativeImageUpscale"
			|| nodeData.name == "RegionalSampler"|| nodeData.name == "RegionalSamplerAdvanced") {
			impactProgressBadge.addStatusHandler(nodeType);
		}

		if(nodeData.name == "ImpactControlBridge") {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info || this.inputs[0].type != '*')
					return;

				// assign type
				let slot_type = '*';

				if(type == 2) {
					slot_type = link_info.type;
				}
				else {
					const node = app.graph.getNodeById(link_info.origin_id);
					slot_type = node.outputs[link_info.origin_slot].type;
				}

				this.inputs[0].type = slot_type;
				this.outputs[0].type = slot_type;
				this.outputs[0].label = slot_type;
			}
		}

		if(nodeData.name == "ImpactConditionalBranch" || nodeData.name == "ImpactConditionalBranchSelMode") {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info || this.inputs[0].type != '*')
					return;

				if(index >= 2)
					return;

				// assign type
				let slot_type = '*';

				if(type == 2) {
					slot_type = link_info.type;
				}
				else {
					const node = app.graph.getNodeById(link_info.origin_id);
					slot_type = node.outputs[link_info.origin_slot].type;
				}

				this.inputs[0].type = slot_type;
				this.inputs[1].type = slot_type;
				this.outputs[0].type = slot_type;
				this.outputs[0].label = slot_type;
			}
		}

		if(nodeData.name == "ImpactCompare") {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info || this.inputs[0].type != '*' || type == 2)
					return;

				// assign type
				const node = app.graph.getNodeById(link_info.origin_id);
				let slot_type = node.outputs[link_info.origin_slot].type;

				this.inputs[0].type = slot_type;
				this.inputs[1].type = slot_type;
			}
		}

		if(nodeData.name === 'ImpactInversedSwitch') {
			nodeData.output = ['*'];
			nodeData.output_is_list = [false];
			nodeData.output_name = ['output1'];

			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info)
					return;

				if(type == 2) {
					// connect output
					if(connected){
						if(app.graph._nodes_by_id[link_info.target_id].type == 'Reroute') {
							app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
						}

						if(this.outputs[0].type == '*'){
							if(link_info.type == '*') {
								app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
							}
							else {
								// propagate type
								this.outputs[0].type = link_info.type;
								this.outputs[0].name = link_info.type;

								for(let i in this.inputs) {
									if(this.inputs[i].name != 'select')
										this.inputs[i].type = link_info.type;
								}
							}
						}
					}
				}
				else {
					if(app.graph._nodes_by_id[link_info.origin_id].type == 'Reroute')
						this.disconnectInput(link_info.target_slot);

					// connect input
					if(this.inputs[0].type == '*'){
						const node = app.graph.getNodeById(link_info.origin_id);
						let origin_type = node.outputs[link_info.origin_slot].type;

						if(origin_type == '*') {
							this.disconnectInput(link_info.target_slot);
							return;
						}

						for(let i in this.inputs) {
							if(this.inputs[i].name != 'select')
								this.inputs[i].type = origin_type;
						}

						this.outputs[0].type = origin_type;
						this.outputs[0].name = origin_type;
					}

					return;
				}

				if (!connected && this.outputs.length > 1) {
					const stackTrace = new Error().stack;

					if(
						!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
						!stackTrace.includes('LGraphNode.connect') && // for mouse device
						!stackTrace.includes('loadGraphData')) {
							if(this.outputs[link_info.origin_slot].links.length == 0)
								this.removeOutput(link_info.origin_slot);
					}
				}

				let slot_i = 1;
				for (let i = 0; i < this.outputs.length; i++) {
					this.outputs[i].name = `output${slot_i}`
					slot_i++;
				}

				let last_slot = this.outputs[this.outputs.length - 1];
				if (last_slot.slot_index == link_info.origin_slot) {
					this.addOutput(`output${slot_i}`, this.outputs[0].type);
				}

				let select_slot = this.inputs.find(x => x.name == "select");
				if(this.widgets) {
					this.widgets[0].options.max = select_slot?this.outputs.length-1:this.outputs.length;
					this.widgets[0].value = Math.min(this.widgets[0].value, this.widgets[0].options.max);
					if(this.widgets[0].options.max > 0 && this.widgets[0].value == 0)
						this.widgets[0].value = 1;
				}
			}
		}

		if (nodeData.name === 'ImpactMakeImageList' || nodeData.name === 'ImpactMakeImageBatch' ||
			nodeData.name === 'CombineRegionalPrompts' ||
			nodeData.name === 'ImpactCombineConditionings' || nodeData.name === 'ImpactConcatConditionings' ||
			nodeData.name === 'ImpactSEGSConcat' ||
			nodeData.name === 'ImpactSwitch' || nodeData.name === 'LatentSwitch' || nodeData.name == 'SEGSSwitch') {
			var input_name = "input";

			switch(nodeData.name) {
			case 'ImpactMakeImageList':
			case 'ImpactMakeImageBatch':
				input_name = "image";
				break;

			case 'ImpactSEGSConcat':
				input_name = "segs";
				break;

			case 'CombineRegionalPrompts':
				input_name = "regional_prompts";
				break;

			case 'ImpactCombineConditionings':
			case 'ImpactConcatConditionings':
				input_name = "conditioning";
				break;

			case 'LatentSwitch':
				input_name = "input";
				break;

			case 'SEGSSwitch':
				input_name = "input";
				break;

			case 'ImpactSwitch':
				input_name = "input";
			}

			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info)
					return;

				if(type == 2) {
					// connect output
					if(connected && index == 0){
						if(nodeData.name == 'ImpactSwitch' && app.graph._nodes_by_id[link_info.target_id]?.type == 'Reroute') {
							app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
						}

						if(this.outputs[0].type == '*'){
							if(link_info.type == '*') {
								app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
							}
							else {
								// propagate type
								this.outputs[0].type = link_info.type;
								this.outputs[0].label = link_info.type;
								this.outputs[0].name = link_info.type;

								for(let i in this.inputs) {
									let input_i = this.inputs[i];
									if(input_i.name != 'select' && input_i.name != 'sel_mode')
										input_i.type = link_info.type;
								}
							}
						}
					}

					return;
				}
				else {
					if(nodeData.name == 'ImpactSwitch' && app.graph._nodes_by_id[link_info.origin_id].type == 'Reroute')
						this.disconnectInput(link_info.target_slot);

					// connect input
					if(this.inputs[index].name == 'select' || this.inputs[index].name == 'sel_mode')
						return;

					if(this.inputs[0].type == '*'){
						const node = app.graph.getNodeById(link_info.origin_id);
						let origin_type = node.outputs[link_info.origin_slot].type;

						if(origin_type == '*') {
							this.disconnectInput(link_info.target_slot);
							return;
						}

						for(let i in this.inputs) {
							let input_i = this.inputs[i];
							if(input_i.name != 'select' && input_i.name != 'sel_mode')
								input_i.type = origin_type;
						}

						this.outputs[0].type = origin_type;
						this.outputs[0].label = origin_type;
						this.outputs[0].name = origin_type;
					}
				}

				let select_slot = this.inputs.find(x => x.name == "select");
				let mode_slot = this.inputs.find(x => x.name == "sel_mode");

				let converted_count = 0;
				converted_count += select_slot?1:0;
				converted_count += mode_slot?1:0;

				if (!connected && (this.inputs.length > 1+converted_count)) {
					const stackTrace = new Error().stack;

					if(
						!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
						!stackTrace.includes('LGraphNode.connect') && // for mouse device
						!stackTrace.includes('loadGraphData') &&
						this.inputs[index].name != 'select') {
						this.removeInput(index);
					}
				}

				let slot_i = 1;
				for (let i = 0; i < this.inputs.length; i++) {
					let input_i = this.inputs[i];
					if(input_i.name != 'select'&& input_i.name != 'sel_mode') {
						input_i.name = `${input_name}${slot_i}`
						slot_i++;
					}
				}

				let last_slot = this.inputs[this.inputs.length - 1];
				if (
					(last_slot.name == 'select' && last_slot.name != 'sel_mode' && this.inputs[this.inputs.length - 2].link != undefined)
					|| (last_slot.name != 'select' && last_slot.name != 'sel_mode' && last_slot.link != undefined)) {
						this.addInput(`${input_name}${slot_i}`, this.outputs[0].type);
				}

				if(this.widgets) {
					this.widgets[0].options.max = select_slot?this.inputs.length-1:this.inputs.length;
					this.widgets[0].value = Math.min(this.widgets[0].value, this.widgets[0].options.max);
					if(this.widgets[0].options.max > 0 && this.widgets[0].value == 0)
						this.widgets[0].value = 1;
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
			case "ToDetailerPipeSDXL":
			case "BasicPipeToDetailerPipe":
			case "BasicPipeToDetailerPipeSDXL":
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

		if(node.comfyClass == "ImpactSEGSLabelFilter" || node.comfyClass == "SEGSLabelFilterDetailerHookProvider") {
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

		if(node.comfyClass == "UltralyticsDetectorProvider") {
			let model_name_widget = node.widgets.find((w) => w.name === "model_name");
			let orig_draw = node.onDrawForeground;
			node.onDrawForeground = function (ctx) {
				const r = orig_draw?.apply?.(this, arguments);

				let is_seg = model_name_widget.value.startsWith('segm/') || model_name_widget.value.includes('-seg');
				if(!is_seg) {
					var slot_pos = new Float32Array(2);
					var pos = node.getConnectionPos(false, 1, slot_pos);

					pos[0] -= node.pos[0] - 10;
					pos[1] -= node.pos[1];

					ctx.beginPath();
					ctx.strokeStyle = "red";
					ctx.lineWidth = 4;
					ctx.moveTo(pos[0] - 5, pos[1] - 5);
					ctx.lineTo(pos[0] + 5, pos[1] + 5);
					ctx.moveTo(pos[0] + 5, pos[1] - 5);
					ctx.lineTo(pos[0] - 5, pos[1] + 5);
					ctx.stroke();
				}
			}
		}

		if(
		node.comfyClass == "ImpactWildcardEncode" || node.comfyClass == "ImpactWildcardProcessor"
		|| node.comfyClass == "ToDetailerPipe" || node.comfyClass == "ToDetailerPipeSDXL"
		|| node.comfyClass == "EditDetailerPipe" || node.comfyClass == "EditDetailerPipeSDXL"
		|| node.comfyClass == "BasicPipeToDetailerPipe" || node.comfyClass == "BasicPipeToDetailerPipeSDXL") {
			node._value = "Select the LoRA to add to the text";
			node._wvalue = "Select the Wildcard to add to the text";

			var tbox_id = 0;
			var combo_id = 3;
			var has_lora = true;

			switch(node.comfyClass){
				case "ImpactWildcardEncode":
					tbox_id = 0;
					combo_id = 3;
					break;

				case "ImpactWildcardProcessor":
					tbox_id = 0;
					combo_id = 4;
					has_lora = false;
					break;

				case "ToDetailerPipe":
				case "ToDetailerPipeSDXL":
				case "EditDetailerPipe":
				case "EditDetailerPipeSDXL":
				case "BasicPipeToDetailerPipe":
				case "BasicPipeToDetailerPipeSDXL":
					tbox_id = 0;
					combo_id = 1;
					break;
			}

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
						const stackTrace = new Error().stack;
						if(stackTrace.includes('inner_value_change')) {
							if(value != "Select the Wildcard to add to the text") {
								if(node.widgets[tbox_id].value != '')
									node.widgets[tbox_id].value += ', '

								node.widgets[tbox_id].value += value;
							}
						}
					},
				get: () => { return "Select the Wildcard to add to the text"; }
			});

			Object.defineProperty(node.widgets[combo_id+1].options, "values", {
				set: (x) => {},
				get: () => {
					return wildcards_list;
				}
			});

			if(has_lora) {
				Object.defineProperty(node.widgets[combo_id], "value", {
					set: (value) => {
							const stackTrace = new Error().stack;
							if(stackTrace.includes('inner_value_change')) {
								if(value != "Select the LoRA to add to the text") {
									let lora_name = value;
									if (lora_name.endsWith('.safetensors')) {
										lora_name = lora_name.slice(0, -12);
									}

									node.widgets[tbox_id].value += `<lora:${lora_name}>`;
									if(node.widgets_values) {
										node.widgets_values[tbox_id] = node.widgets[tbox_id].value;
									}
								}
							}

							node._value = value;
						},

					get: () => { return "Select the LoRA to add to the text"; }
				});
			}

			// Preventing validation errors from occurring in any situation.
			if(has_lora) {
				node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			}
			node.widgets[combo_id+1].serializeValue = () => { return "Select the Wildcard to add to the text"; }
		}

		if(node.comfyClass == "ImpactWildcardProcessor" || node.comfyClass == "ImpactWildcardEncode") {
			node.widgets[0].inputEl.placeholder = "Wildcard Prompt (User input)";
			node.widgets[1].inputEl.placeholder = "Populated Prompt (Will be generated automatically)";
			node.widgets[1].inputEl.disabled = true;

			const populated_text_widget = node.widgets.find((w) => w.name == 'populated_text');
			const mode_widget = node.widgets.find((w) => w.name == 'mode');

			// mode combo
			Object.defineProperty(mode_widget, "value", {
				set: (value) => {
						node._mode_value = value == true || value == "Populate";
						populated_text_widget.inputEl.disabled = value == true || value == "Populate";
					},
				get: () => {
						if(node._mode_value != undefined)
							return node._mode_value;
						else
							return true;
					 }
			});
		}

		if (node.comfyClass == "MaskPainter") {
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

						if(node._images) {
							app.nodeOutputs[id].images = [{
									...node._images[0],
									...item
								}];

							node.widgets[0].value =
								{
									...node._images[0],
									...item
								};
						}
						else {
							app.nodeOutputs[id].images = [{
									...item
								}];

							node.widgets[0].value =
								{
									...item
								};
						}

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
