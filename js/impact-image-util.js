import { ComfyApp, app } from "../../scripts/app.js";

function load_image(str) {
	let base64String = canvas.toDataURL('image/png');
	let img = new Image();
	img.src = base64String;
}

function getFileItem(baseType, path) {
    let pathType = baseType;

    if (path === "[output]") {
        pathType = "output";
        path = path.slice(0, -9);
    } else if (path === "[input]") {
        pathType = "input";
        path = path.slice(0, -8);
    } else if (path === "[temp]") {
        pathType = "temp";
        path = path.slice(0, -7);
    }

    const subfolder = path.substring(0, path.lastIndexOf('/'));
    const filename = path.substring(path.lastIndexOf('/') + 1);

    return {
        filename: filename,
        subfolder: subfolder,
        type: pathType
    };
}

app.registerExtension({
	name: "Comfy.Impact.img",

	nodeCreated(node, app) {
		if(node.comfyClass == "PreviewBridge") {
			let w = node.widgets.find(obj => obj.name === 'image');
			Object.defineProperty(w, 'value', {
				async set(v) {
					w._value = v;
					let image = new Image();

					try {
						let item = getFileItem('temp', v);
						let params = `?filename=${item.filename}&type=${item.type}&subfolder=${item.subfolder}`;

						let res = await api.fetchApi('/view/validate'+params, { cache: "no-store" });
						if(res.status == 200) {
							image.src = 'view'+params;
						}
						else
							w._value = undefined;
					}
					catch {
						w._value = undefined;
					}
					node.imgs = [image];
				},
				get() {
					if(w._value == undefined) {
						return undefined;
					}
					return w._value;
				}
			});
		}

		if(node.comfyClass == "ImageReceiver") {
			let path_widget = node.widgets.find(obj => obj.name === 'image');
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

			let set_img_act = (v) => {
				node._img = v;
				var canvas = document.createElement('canvas');
				canvas.width = v[0].width;
				canvas.height = v[0].height;

				var context = canvas.getContext('2d');
				context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

				var base64Image = canvas.toDataURL('image/png');
				w.value = base64Image;
			};

			Object.defineProperty(node, 'imgs', {
				set(v) {
					if (!v[0].complete) {
						let orig_onload = v[0].onload;
						v[0].onload = function(v2) {
							if(orig_onload)
								orig_onload();
							set_img_act(v);
						};
					}
					else {
						set_img_act(v);
					}
				},
				get() {
					if(this._img == undefined && w.value != '') {
						this._img = [new Image()];
						if(stw_widget.value && w.value != '[IMAGE DATA]')
							this._img[0].src = w.value;
					}
					else if(this._img == undefined && path_widget.value) {
						let image = new Image();
						image.src = path_widget.value;

						try {
							let item = getFileItem('temp', path_widget.value);
							let params = `?filename=${item.filename}&type=${item.type}&subfolder=${item.subfolder}`;

							let res = api.fetchApi('/view/validate'+params, { cache: "no-store" }).then(response => response);
							if(res.status == 200) {
								image.src = 'view'+params;
							}

							this._img = [new Image()]; // placeholder
							image.onload = function(v) {
								set_img_act([image]);
							};
						}
						catch {

						}
					}
					return this._img;
				}
			});
		}
    }
})