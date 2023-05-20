import { app } from "/scripts/app.js";
import { ComfyDialog, $el } from "/scripts/ui.js";
import { api } from "/scripts/api.js";

// Helper function to convert a data URL to a Blob object
function dataURLToBlob(dataURL) {
  const parts = dataURL.split(';base64,');
  const contentType = parts[0].split(':')[1];
  const byteString = atob(parts[1]);
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);
  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }
  return new Blob([arrayBuffer], { type: contentType });
}

async function invalidateImage(node, formData) {
  const filepath = node.images[0];

  await fetch('/upload/temp', {
    method: 'POST',
    body: formData
  }).then(response => {
  }).catch(error => {
    console.error('Error:', error);
  });

  const img = new Image();
  img.onload = () => {
    node.imgs = [img];
    app.graph.setDirtyCanvas(true);
  };

  img.src = `view?filename=${filepath.filename}&type=${filepath.type}`;;
}

class ImpactInpaintDialog extends ComfyDialog {
  constructor() {
    super();
		this.element = $el("div.comfy-modal", { parent: document.body }, 
    [
			$el("div.comfy-modal-content", 
        [
          ...this.createButtons()]),
		]);
	}

	createButtons() {
		return [
			$el("button", {
				type: "button",
				textContent: "Save",
				onclick: () => {          
          const backupCtx = this.backupCanvas.getContext('2d', {transparent: true});
          backupCtx.clearRect(0,0,this.backupCanvas.width,this.backupCanvas.height);
          backupCtx.drawImage(this.maskCanvas, 
              0, 0, this.maskCanvas.width, this.maskCanvas.height, 
              0, 0, this.backupCanvas.width, this.backupCanvas.height);
              
          // paste mask data into alpha channel
          const backupData = backupCtx.getImageData(0, 0, this.backupCanvas.width, this.backupCanvas.height);

          for (let i = 0; i < backupData.data.length; i += 4) {
            if(backupData.data[i+3] == 255)
              backupData.data[i+3] = 0;
            else
              backupData.data[i+3] = 255;
              
            backupData.data[i] = 0;
            backupData.data[i+1] = 0;
            backupData.data[i+2] = 0;
          }

          backupCtx.globalCompositeOperation = 'source-over';
          backupCtx.putImageData(backupData, 0, 0);

          const dataURL = this.backupCanvas.toDataURL();
          const blob = dataURLToBlob(dataURL);

          const formData = new FormData();
          const filename = "impact-mask-" + performance.now() + ".png";

          const item =
            {
              "filename": filename,
              "subfolder": "",
              "type": "temp",
            };

          this.node.images[0] = item;
          this.node.widgets[1].value = item;

          formData.append('image', blob, filename);
          invalidateImage(this.node, formData);
          this.close();
        }
			}),
			$el("button", {
				type: "button",
				textContent: "Cancel",
				onclick: () => this.close(),
			}),
			$el("button", {
				type: "button",
				textContent: "Clear",
				onclick: () => { 
          this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        },
			}),
		];
	}

	show() {
    const imgCanvas = document.createElement('canvas');
    const maskCanvas = document.createElement('canvas');
    const backupCanvas = document.createElement('canvas');
    imgCanvas.id = "imageCanvas";
    maskCanvas.id = "maskCanvas";
    backupCanvas.id = "backupCanvas";

    this.element.appendChild(imgCanvas);
    this.element.appendChild(maskCanvas);

    this.node.widgets[1].value = null;

    this.element.style.display = "block";
    imgCanvas.style.position = "relative";
    imgCanvas.style.top = "200";
    imgCanvas.style.left = "0";

    maskCanvas.style.position = "absolute";
    
    const imgCtx = imgCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');
    const backupCtx = backupCanvas.getContext('2d');

    this.maskCanvas = maskCanvas;
    this.maskCtx = maskCtx;
    this.backupCanvas = backupCanvas;
    
    window.addEventListener("resize", () => {
      // repositioning 
      imgCanvas.width = window.innerWidth - 250;
      imgCanvas.height = window.innerHeight - 300;
      
      // redraw image
      let drawWidth = image.width;
      let drawHeight = image.height;
      if (image.width > imgCanvas.width) {
        drawWidth = imgCanvas.width;
        drawHeight = (drawWidth / image.width) * image.height;
      }
      if (drawHeight > imgCanvas.height) {
        drawHeight = imgCanvas.height;
        drawWidth = (drawHeight / image.height) * image.width;
      }

      imgCtx.drawImage(image, 0, 0, drawWidth, drawHeight);

      // update mask
      backupCtx.drawImage(maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height, 0, 0, backupCanvas.width, backupCanvas.height);
      
      maskCanvas.width = drawWidth; 
      maskCanvas.height = drawHeight;
      maskCanvas.style.top = imgCanvas.offsetTop + "px";
      maskCanvas.style.left = imgCanvas.offsetLeft + "px";

      maskCtx.drawImage(backupCanvas, 0, 0, backupCanvas.width, backupCanvas.height, 0, 0, maskCanvas.width, maskCanvas.height);
    });
    

    // image load
    const image = new Image();
    image.onload = function() {
        backupCanvas.width = image.width;
        backupCanvas.height = image.height; 
        window.dispatchEvent(new Event('resize'));
    };
    
    const filepath = this.node.images[0];
    image.src = this.node.imgs[0].src;
    this.image = image;

    
    // event handler for user drawing ------
    let brush_size = 10;

    function mouse_down(event) {
      if (event.buttons === 1) { 
        const maskRect = maskCanvas.getBoundingClientRect();
        const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
        const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

        maskCtx.beginPath();
        maskCtx.fillStyle = "rgb(0,0,0)";
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
        maskCtx.fill();
      }
    }

    function mouse_move(event) {
      if (event.buttons === 1) { 
        event.preventDefault();
        const maskRect = maskCanvas.getBoundingClientRect();
        const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
        const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

        maskCtx.beginPath();
        maskCtx.fillStyle = "rgb(0,0,0)";
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
        maskCtx.fill();
      }
      else if(event.buttons === 2) {
        event.preventDefault();
        const maskRect = maskCanvas.getBoundingClientRect();
        const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
        const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

        maskCtx.beginPath();
        maskCtx.globalCompositeOperation = "destination-out";
        maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
        maskCtx.fill();
      }
    }

    function touch_move(event) {
      event.preventDefault();
      const maskRect = maskCanvas.getBoundingClientRect();
      const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
      const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

      maskCtx.beginPath();
      maskCtx.fillStyle = "rgb(0,0,0)";
      maskCtx.globalCompositeOperation = "source-over";
      maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
      maskCtx.fill();
    }

    function handleWheelEvent(event) {

      if(event.deltaY < 0)
        brush_size = Math.min(brush_size+2, 100);
      else
        brush_size = Math.max(brush_size-2, 1);
    }

    maskCanvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });
    maskCanvas.addEventListener('wheel', handleWheelEvent);
    maskCanvas.addEventListener('mousedown', mouse_down);
    maskCanvas.addEventListener('mousemove', mouse_move);
    maskCanvas.addEventListener('touchmove', touch_move);
	}
}

const input_tracking = {};
const input_dirty = {};
const output_tracking = {};

function executeHandler(event) {
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

var eventRegistered = false;

app.registerExtension({
	name: "Comfy.Impack",
	loadedGraphNode(node, app) {
		if (node.comfyClass == "PreviewBridge") {
			if (!eventRegistered) {
				api.addEventListener("executed", executeHandler);
				eventRegistered = true;
			}

			input_dirty[node.id + ""] = false;
		}
	},
	nodeCreated(node, app) {
		if(node.comfyClass == "MaskPainter") {
			node.addWidget("button", "Edit mask", null, () => {
				this.dlg = new ImpactInpaintDialog(app);
				this.dlg.node = node;

				if('images' in node) {
					this.dlg.show();
				}
			});

			node.addWidget("hidden", "mask_image", null, null);
		}
		else if (node.comfyClass == "PreviewBridge") {
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