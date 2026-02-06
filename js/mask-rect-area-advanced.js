import { app } from "../../scripts/app.js";
import { readLinkedNumber, getDrawColor, computeCanvasSize } from "./common.js";
function showPreviewCanvas(node, app) {

    const widget = {
        type: "customCanvas",
        name: "mask-rect-area-canvas",
        get value() {
            return this.canvas.value;
        },
        set value(x) {
            this.canvas.value = x;
        },
        draw: function (ctx, node, widgetWidth, widgetY) {

            // If we are initially offscreen when created we wont have received a resize event
            // Calculate it here instead
            if (!node.canvasHeight) {
                computeCanvasSize(node, node.size, 220, 240);
            }

            const visible = true;
            const t = ctx.getTransform();
            const margin = 12;
            const border = 2;
            const widgetHeight = node.canvasHeight;

            // Keep preview in sync when inputs are driven by links.
            syncLinkedInputsToPropertiesAdvanced(node);

            const width = Math.max(1, Math.round(node.properties["width"]));
            const height = Math.max(1, Math.round(node.properties["height"]));
            const scale = Math.min(
                    (widgetWidth - margin * 3) / width,
                    (widgetHeight - margin * 3) / height
                    );
            const blurRadius = node.properties["blur_radius"] || 0;
            const index = 0;

            Object.assign(this.canvas.style, {
                left: `${t.e}px`,
                top: `${t.f + (widgetY * t.d)}px`,
                width: `${widgetWidth * t.a}px`,
                height: `${widgetHeight * t.d}px`,
                position: "absolute",
                zIndex: 1,
                fontSize: `${t.d * 10.0}px`,
                pointerEvents: "none"
            });

            this.canvas.hidden = !visible;

            let backgroundWidth = width * scale;
            let backgroundHeight = height * scale;

            let xOffset = margin;
            if (backgroundWidth < widgetWidth) {
                xOffset += (widgetWidth - backgroundWidth) / 2 - margin;
            }
            let yOffset = (margin / 2);
            if (backgroundHeight < widgetHeight) {
                yOffset += (widgetHeight - backgroundHeight) / 2 - margin;
            }

            let widgetX = xOffset;
            widgetY = widgetY + yOffset;

            // Draw the background border
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(widgetX - border, widgetY - border, backgroundWidth + border * 2, backgroundHeight + border * 2)

            // Draw the main background area 
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR;
            ctx.fillRect(widgetX, widgetY, backgroundWidth, backgroundHeight);

            // Draw the conditioning zone
            let [x, y, w, h] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + x, widgetY + y, w, h);
            ctx.beginPath();
            ctx.lineWidth = 1;

            // Draw grid lines
            for (let x = 0; x <= width / 64; x += 1) {
                ctx.moveTo(widgetX + x * 64 * scale, widgetY);
                ctx.lineTo(widgetX + x * 64 * scale, widgetY + backgroundHeight);
            }

            for (let y = 0; y <= height / 64; y += 1) {
                ctx.moveTo(widgetX, widgetY + y * 64 * scale);
                ctx.lineTo(widgetX + backgroundWidth, widgetY + y * 64 * scale);
            }

            ctx.strokeStyle = "#66666650";
            ctx.stroke();
            ctx.closePath();

            // Draw current zone
            let [sx, sy, sw, sh] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + sx, widgetY + sy, sw, sh);

            ctx.fillStyle = getDrawColor(0, "40");
            ctx.fillRect(widgetX + sx + border, widgetY + sy + border, sw - border * 2, sh - border * 2);

            // Draw white border around the current zone
            ctx.strokeStyle = globalThis.LiteGraph.NODE_SELECTED_TITLE_COLOR;
            ctx.lineWidth = 2;
            ctx.strokeRect(widgetX + sx, widgetY + sy, sw, sh);

            // Display
            ctx.beginPath();

            ctx.arc(LiteGraph.NODE_SLOT_HEIGHT * 0.5, LiteGraph.NODE_SLOT_HEIGHT * (index + 0.5) + 4, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.strokeStyle = "white";
            ctx.stroke();

            ctx.lineWidth = 1;
            ctx.closePath();

            // Draw progress bar canvas
            if (backgroundWidth < widgetWidth) {
                xOffset += (widgetWidth - backgroundWidth) / 2 - margin;
            }

            // Adjust X and Y coordinates
            const barHeight = 8;
            let widgetYBar = widgetY + backgroundHeight + margin;

            // Draw the border around the progress bar
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(
                    widgetX - border,
                    widgetYBar - border,
                    backgroundWidth + border * 2,
                    barHeight + border * 2
                    );

            // Draw the main bar area (background)
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR;
            ctx.fillRect(
                    widgetX,
                    widgetYBar,
                    backgroundWidth,
                    barHeight
                    );

            // Draw progress bar grid
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "#66666650";

            // Calculate the number of grid lines based on the bar size
            const numLines = Math.floor(backgroundWidth / 64);

            // Draw grid lines
            for (let x = 0; x <= width / 64; x += 1) {
                ctx.moveTo(widgetX + x * 64 * scale, widgetYBar);
                ctx.lineTo(widgetX + x * 64 * scale, widgetYBar + barHeight);
            }
            ctx.stroke();
            ctx.closePath();

            // Draw progress (based on blur_radius)
            const progress = Math.min(blurRadius / 255, 1);
            ctx.fillStyle = "rgba(0, 120, 255, 0.5)";

            ctx.fillRect(
                    widgetX,
                    widgetYBar,
                    backgroundWidth * progress,
                    barHeight
                    );
        }
    };

    widget.canvas = document.createElement("canvas");
    widget.canvas.className = "mask-rect-area-canvas";
    widget.parent = node;

    widget.computeLayoutSize = function (node) {
        return {
            minHeight: 200,
            maxHeight: 300
        };
    };

    document.body.appendChild(widget.canvas);
    node.addCustomWidget(widget);

    app.canvas.onDrawBackground = function () {
        // Draw node isnt fired once the node is off the screen
        // if it goes off screen quickly, the input may not be removed
        // this shifts it off screen so it can be moved back if the node is visible.
        for (let n in app.graph._nodes) {
            n = app.graph._nodes[n];
            for (let w in n.widgets) {
                let wid = n.widgets[w];
                if (Object.hasOwn(wid, "canvas")) {
                    wid.canvas.style.left = -8000 + "px";
                    wid.canvas.style.position = "absolute";
                }
            }
        }
    };

    node.onResize = function (size) {
        computeCanvasSize(node, size, 220, 240);
    };

    return {minWidth: 200, minHeight: 200, widget};
}

app.registerExtension({
    name: "drltdata.MaskRectAreaAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MaskRectAreaAdvanced") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.setProperty("width", 512);
            this.setProperty("height", 512);
            this.setProperty("x", 0);
            this.setProperty("y", 0);
            this.setProperty("w", 256);
            this.setProperty("h", 256);
            this.setProperty("blur_radius", 0);

            this.selected = false;
            this.index = 3;
            this.serialize_widgets = true;

            // If the node already provides widgets from Python/ComfyUI, do NOT recreate them
            const hasExisting = Array.isArray(this.widgets) && this.widgets.some(w => w && w.name === "x");

            // Helper: attach callbacks to existing widgets to keep node.properties in sync (canvas preview).
            const hookWidget = (node, widgetName, propName, opts) => {
                if (!Array.isArray(node.widgets)) {
                    return;
                }
                const w = node.widgets.find(ww => ww && ww.name === widgetName);
                if (!w) {
                    return;
                }

                const min = (opts && typeof opts.min === "number") ? opts.min : undefined;
                const max = (opts && typeof opts.max === "number") ? opts.max : undefined;
                const step = (opts && typeof opts.step === "number") ? opts.step : undefined;

                if (node.properties && Object.prototype.hasOwnProperty.call(node.properties, propName)) {
                    w.value = node.properties[propName];
                } else {
                    node.properties[propName] = w.value;
                }

                const prevCb = w.callback;
                w.callback = function (v, ...args) {
                    let val = v;
                    if (typeof val === "number") {
                        if (typeof step === "number" && step > 0) {
                            const s = step / 10;
                            val = Math.round(val / s) * s;
                        } else {
                            val = Math.round(val);
                        }
                        if (typeof min === "number") {
                            val = Math.max(min, val);
                        }
                        if (typeof max === "number") {
                            val = Math.min(max, val);
                        }
                    }
                    this.value = val;
                    node.properties[propName] = val;
                    if (prevCb) {
                        return prevCb.call(this, val, ...args);
                    }
                };
            };

            if (hasExisting) {
                hookWidget(this, "x", "x", {"step": 10});
                hookWidget(this, "y", "y", {"step": 10});
                hookWidget(this, "width", "w", {"step": 10});
                hookWidget(this, "height", "h", {"step": 10});
                hookWidget(this, "image_width", "width", {"step": 10});
                hookWidget(this, "image_height", "height", {"step": 10});
                hookWidget(this, "blur_radius", "blur_radius", {"min": 0, "max": 255, "step": 10});
            } else {
                CUSTOM_INT(this, "x", 0, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["x"] = this.value;
                });
                CUSTOM_INT(this, "y", 0, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["y"] = this.value;
                });
                CUSTOM_INT(this, "width", 256, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["w"] = this.value;
                });
                CUSTOM_INT(this, "height", 256, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["h"] = this.value;
                });
                CUSTOM_INT(this, "image_width", 512, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["width"] = this.value;
                });
                CUSTOM_INT(this, "image_height", 512, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["height"] = this.value;
                });
                CUSTOM_INT(this, "blur_radius", 0, function (v, _, node) {
                    this.value = Math.round(v) || 0;
                    node.properties["blur_radius"] = this.value;
                },
                        {"min": 0, "max": 255, "step": 10}
                );
            }

            showPreviewCanvas(this, app);

            this.onSelected = function () {
                this.selected = true;
            };
            this.onDeselected = function () {
                this.selected = false;
            };

            return r;
        };
    }
});

// Calculate the drawing area using individual properties.
function getDrawArea(node, backgroundWidth, backgroundHeight) {
    let x = node.properties["x"] * backgroundWidth / node.properties["width"];
    let y = node.properties["y"] * backgroundHeight / node.properties["height"];
    let w = node.properties["w"] * backgroundWidth / node.properties["width"];
    let h = node.properties["h"] * backgroundHeight / node.properties["height"];

    if (x > backgroundWidth) {
        x = backgroundWidth;
    }
    if (y > backgroundHeight) {
        y = backgroundHeight;
    }

    if (x + w > backgroundWidth) {
        w = Math.max(0, backgroundWidth - x);
    }

    if (y + h > backgroundHeight) {
        h = Math.max(0, backgroundHeight - y);
    }

    return [x, y, w, h];
}

function CUSTOM_INT(node, inputName, val, func, config = {}) {
    return {
        widget: node.addWidget(
                "number",
                inputName,
                val,
                func,
                Object.assign({}, {min: 0, max: 4096, step: 640, precision: 0}, config)
                )
    };
}

function syncLinkedInputsToPropertiesAdvanced(node) {
    let changed = false;

    const vx = readLinkedNumber(node, "x");
    if (vx != null) {
        const nv = Math.max(0, Math.round(vx));
        if (node.properties["x"] !== nv) {
            node.properties["x"] = nv;
            changed = true;
        }
    }

    const vy = readLinkedNumber(node, "y");
    if (vy != null) {
        const nv = Math.max(0, Math.round(vy));
        if (node.properties["y"] !== nv) {
            node.properties["y"] = nv;
            changed = true;
        }
    }

    // Input "width" is the rectangle width in px -> property "w"
    const vw = readLinkedNumber(node, "width");
    if (vw != null) {
        const nv = Math.max(0, Math.round(vw));
        if (node.properties["w"] !== nv) {
            node.properties["w"] = nv;
            changed = true;
        }
    }

    // Input "height" is the rectangle height in px -> property "h"
    const vh = readLinkedNumber(node, "height");
    if (vh != null) {
        const nv = Math.max(0, Math.round(vh));
        if (node.properties["h"] !== nv) {
            node.properties["h"] = nv;
            changed = true;
        }
    }

    // Image size (must be >=1 to avoid division by zero in getDrawArea)
    const viw = readLinkedNumber(node, "image_width");
    if (viw != null) {
        const nv = Math.max(1, Math.round(viw));
        if (node.properties["width"] !== nv) {
            node.properties["width"] = nv;
            changed = true;
        }
    }

    const vih = readLinkedNumber(node, "image_height");
    if (vih != null) {
        const nv = Math.max(1, Math.round(vih));
        if (node.properties["height"] !== nv) {
            node.properties["height"] = nv;
            changed = true;
        }
    }

    const vbr = readLinkedNumber(node, "blur_radius");
    if (vbr != null) {
        const nv = Math.max(0, Math.min(255, Math.round(vbr)));
        if (node.properties["blur_radius"] !== nv) {
            node.properties["blur_radius"] = nv;
            changed = true;
        }
    }

    return changed;
}

