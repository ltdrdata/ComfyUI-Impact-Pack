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
				set(v) {
					w._value = v;
					let image = new Image();

					let item = getFileItem('temp', v);

					let v2 = v.replace(/\[temp\]$/, '')
					image.src = `view?filename=${item.filename}&type=${item.type}&subfolder=${item.subfolder}`;
					node.imgs = [image];
				},
				get() {
					return w._value;
				}
			});
		}

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
					let act = () => {
						this._img = v;
						var canvas = document.createElement('canvas');
						canvas.width = v[0].width;
						canvas.height = v[0].height;

						var context = canvas.getContext('2d');
						context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

						var base64Image = canvas.toDataURL('image/png');
						w.value = base64Image;
					};

					if (!v[0].complete) {
						let orig_onload = v[0].onload;
						v[0].onload = function() {
							if(orig_onload)
								orig_onload();
							act();
						};
					}
					else {
						act();
					}
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