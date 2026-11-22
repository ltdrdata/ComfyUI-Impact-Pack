import { app } from "../../scripts/app.js";
import { readLinkedNumber, getDrawColor, computeCanvasSize } from "./common.js";

const MASK_RECT_AREA_BACKGROUND_HOOK = "__impactMaskRectAreaOnDrawBackgroundHook";

function moveMaskRectAreaWidgetsOffscreen() {
    for (const node of app.graph?._nodes || []) {
        for (const widget of node.widgets || []) {
            if (Object.hasOwn(widget, "canvas")) {
                widget.canvas.style.left = "-8000px";
                widget.canvas.style.position = "absolute";
            }
        }
    }
}

function ensureMaskRectAreaBackgroundHook() {
    if (app.canvas[MASK_RECT_AREA_BACKGROUND_HOOK]) {
        return;
    }

    const onDrawBackground = app.canvas.onDrawBackground;
    app.canvas.onDrawBackground = function (...args) {
        onDrawBackground?.apply(this, args);
        moveMaskRectAreaWidgetsOffscreen();
    };
    app.canvas[MASK_RECT_AREA_BACKGROUND_HOOK] = true;
}
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

            if (!node.canvasHeight) {
                computeCanvasSize(node, node.size, 200, 200);
            }

            const visible = true;
            const t = ctx.getTransform();
            const margin = 12;
            const border = 2;
            const widgetHeight = node.canvasHeight;
            const width = 512;
            const height = 512;
            const scale = Math.min((widgetWidth - margin * 3) / width, (widgetHeight - margin * 3) / height);
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

            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(widgetX - border, widgetY - border, backgroundWidth + border * 2, backgroundHeight + border * 2);

            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR;
            ctx.fillRect(widgetX, widgetY, backgroundWidth, backgroundHeight);

            syncLinkedInputsToProperties(node);

            let [x, y, w, h] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + x, widgetY + y, w, h);
            ctx.beginPath();
            ctx.lineWidth = 1;

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

            let [sx, sy, sw, sh] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + sx, widgetY + sy, sw, sh);

            ctx.fillStyle = getDrawColor(0, "40");
            ctx.fillRect(widgetX + sx + border, widgetY + sy + border, sw - border * 2, sh - border * 2);

            ctx.strokeStyle = globalThis.LiteGraph.NODE_SELECTED_TITLE_COLOR;
            ctx.lineWidth = 2;
            ctx.strokeRect(widgetX + sx, widgetY + sy, sw, sh);

            ctx.beginPath();

            ctx.arc(LiteGraph.NODE_SLOT_HEIGHT * 0.5, LiteGraph.NODE_SLOT_HEIGHT * (index + 0.5) + 4, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.strokeStyle = "white";
            ctx.stroke();
            ctx.lineWidth = 1;
            ctx.closePath();

            if (backgroundWidth < widgetWidth) {
                xOffset += (widgetWidth - backgroundWidth) / 2 - margin;
            }

            const barHeight = 8;
            let widgetYBar = widgetY + backgroundHeight + margin;

            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(
                    widgetX - border,
                    widgetYBar - border,
                    backgroundWidth + border * 2,
                    barHeight + border * 2
                    );

            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR;
            ctx.fillRect(
                    widgetX,
                    widgetYBar,
                    backgroundWidth,
                    barHeight
                    );

            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "#66666650";

            const numLines = Math.floor(backgroundWidth / 64);

            for (let x = 0; x <= width / 64; x += 1) {
                ctx.moveTo(widgetX + x * 64 * scale, widgetYBar);
                ctx.lineTo(widgetX + x * 64 * scale, widgetYBar + barHeight);
            }
            ctx.stroke();
            ctx.closePath();

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

    ensureMaskRectAreaBackgroundHook();

    node.onResize = function (size) {
        computeCanvasSize(node, size, 200, 200);
    };

    return {minWidth: 200, minHeight: 200, widget};
}

app.registerExtension({
    name: 'drltdata.MaskRectArea',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MaskRectArea") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.setProperty("width", 512);
            this.setProperty("height", 512);
            this.setProperty("x", 0);
            this.setProperty("y", 0);
            this.setProperty("w", 50);
            this.setProperty("h", 50);
            this.setProperty("blur_radius", 0);

            this.selected = false;
            this.index = 3;
            this.serialize_widgets = true;

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

                if (node.properties && Object.prototype.hasOwnProperty.call(node.properties, propName)) {
                    w.value = node.properties[propName];
                } else {
                    node.properties[propName] = w.value;
                }

                const prevCb = w.callback;
                w.callback = function (v, ...args) {
                    let val = v;

                    if (typeof val === "number") {
                        val = Math.round(val);

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

            hookWidget(this, "x", "x", {"min": 0, "max": 100});
            hookWidget(this, "y", "y", {"min": 0, "max": 100});
            hookWidget(this, "width", "w", {"min": 0, "max": 100});
            hookWidget(this, "height", "h", {"min": 0, "max": 100});
            hookWidget(this, "blur_radius", "blur_radius", {"min": 0, "max": 255});

            showPreviewCanvas(this, app);

            const prevOnExecute = this.onExecute;
            this.onExecute = function () {
                const rr = prevOnExecute ? prevOnExecute.apply(this, arguments) : undefined;

                const readLinkedInt = (inputName) => {
                    if (!Array.isArray(this.inputs)) {
                        return null;
                    }
                    const inp = this.inputs.find(i => i && i.name === inputName);
                    if (!inp || !inp.link) {
                        return null;
                    }
                    try {
                        const v = this.getInputData(inputName);
                        return (typeof v === "number") ? v : null;
                    } catch (e) {
                        return null;
                    }
                };

                let changed = false;

                const vx = readLinkedInt("x");
                if (vx != null) {
                    const nv = Math.max(0, Math.min(100, Math.round(vx)));
                    if (this.properties["x"] !== nv) {
                        this.properties["x"] = nv;
                        changed = true;
                    }
                }

                const vy = readLinkedInt("y");
                if (vy != null) {
                    const nv = Math.max(0, Math.min(100, Math.round(vy)));
                    if (this.properties["y"] !== nv) {
                        this.properties["y"] = nv;
                        changed = true;
                    }
                }

                const vw = readLinkedInt("width");
                if (vw != null) {
                    const nv = Math.max(0, Math.min(100, Math.round(vw)));
                    if (this.properties["w"] !== nv) {
                        this.properties["w"] = nv;
                        changed = true;
                    }
                }

                const vh = readLinkedInt("height");
                if (vh != null) {
                    const nv = Math.max(0, Math.min(100, Math.round(vh)));
                    if (this.properties["h"] !== nv) {
                        this.properties["h"] = nv;
                        changed = true;
                    }
                }

                const vbr = readLinkedInt("blur_radius");
                if (vbr != null) {
                    const nv = Math.max(0, Math.min(255, Math.round(vbr)));
                    if (this.properties["blur_radius"] !== nv) {
                        this.properties["blur_radius"] = nv;
                        changed = true;
                    }
                }

                if (changed) {
                    this.setDirtyCanvas(true, true);
                    if (this.graph) {
                        this.graph.setDirtyCanvas(true, true);
                    }
                }

                return rr;
            };

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


function getDrawArea(node, backgroundWidth, backgroundHeight) {
    let x = (node.properties["x"] / 100) * backgroundWidth;
    let y = (node.properties["y"] / 100) * backgroundHeight;
    let w = (node.properties["w"] / 100) * backgroundWidth;
    let h = (node.properties["h"] / 100) * backgroundHeight;

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

function syncLinkedInputsToProperties(node) {
    let changed = false;

    const vx = readLinkedNumber(node, "x");
    if (vx != null) {
        const nv = Math.max(0, Math.min(100, Math.round(vx)));
        if (node.properties["x"] !== nv) {
            node.properties["x"] = nv;
            changed = true;
        }
    }

    const vy = readLinkedNumber(node, "y");
    if (vy != null) {
        const nv = Math.max(0, Math.min(100, Math.round(vy)));
        if (node.properties["y"] !== nv) {
            node.properties["y"] = nv;
            changed = true;
        }
    }

    const vw = readLinkedNumber(node, "width");
    if (vw != null) {
        const nv = Math.max(0, Math.min(100, Math.round(vw)));
        if (node.properties["w"] !== nv) {
            node.properties["w"] = nv;
            changed = true;
        }
    }

    const vh = readLinkedNumber(node, "height");
    if (vh != null) {
        const nv = Math.max(0, Math.min(100, Math.round(vh)));
        if (node.properties["h"] !== nv) {
            node.properties["h"] = nv;
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
