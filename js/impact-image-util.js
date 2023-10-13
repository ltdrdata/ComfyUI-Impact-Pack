import { ComfyApp, app } from "../../scripts/app.js";

function load_image(str) {
	let base64String = canvas.toDataURL('image/png');
	let img = new Image();
	img.src = base64String;
}

app.registerExtension({
	name: "Comfy.Impact.img",

	nodeCreated(node, app) {
		if(node.comfyClass == "ImageReceiver") {
			let w = node.widgets.find(obj => obj.name === 'image_data');
			let stw_widget = node.widgets.find(obj => obj.name === 'save_to_workflow');
			w._value = "";

			Object.defineProperty(w, 'value', {
				set(v) {
					if(v != '[IMAGE DATA]')
						w._value = v;
				},
				get() {
					const stackTrace = new Error().stack;
					if(!stackTrace.includes('draw') && !stackTrace.includes('graphToPrompt') && stackTrace.includes('app.js')) {
						return "[IMAGE DATA]";
					}
					else {
						if(stw_widget.value)
							return w._value;
						else
							return "";
					}
				}
			});

			Object.defineProperty(node, 'imgs', {
				set(v) {
					this._img = v;
					var canvas = document.createElement('canvas');
					canvas.width = v[0].width;
					canvas.height = v[0].height;

					var context = canvas.getContext('2d');
					context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

					var base64Image = canvas.toDataURL('image/png');
					w.value = base64Image;
				},
				get() {
					if(this._img == undefined && w.value != '') {
						this._img = [new Image()];
						if(stw_widget.value && w.value != '[IMAGE DATA]')
							this._img[0].src = w.value;
					}

					return this._img;
				}
			});
		}
    }
})