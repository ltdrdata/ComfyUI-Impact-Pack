import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

// The server keeps the None of that run in its cache, so a later prompt can fail on it without the switch running again.
const gaveNone = new Map();

let marks = [];
let layer;
const frames = document.createElement("style");
const looks = document.createElement("style");
looks.textContent = `
    .impact-none-layer { position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 10; }
    .impact-none-bubble { position: absolute; display: flex; align-items: center; gap: 8px; padding: 6px 7px 6px 12px; border-radius: 12px;
        background: #f5a623; color: #000; font: 600 13px sans-serif; white-space: nowrap; pointer-events: auto; }
    .impact-none-bubble::after { content: ""; position: absolute; left: 18px; top: 100%; border: 10px solid transparent; border-bottom-width: 0; border-top-color: #f5a623; }
    .impact-none-bubble button { width: 18px; height: 18px; border: 0; border-radius: 50%; padding: 0; cursor: pointer; background: #0002 center / 10px no-repeat
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'%3E%3Cpath d='M1.5 1.5l7 7m0-7l-7 7' stroke='%23000' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E"); }
    .impact-none-bubble button:hover { background-color: #0005; }`;
document.head.append(looks, frames);

function place() {
    const area = layer.getBoundingClientRect();
    const framed = [];
    for (const { node, bubble } of marks) {
        const element = node.graph === app.canvas.graph ? document.querySelector(`[data-node-id="${node.id}"]`) : null;
        bubble.hidden = !element;
        if (!element) continue;
        const box = element.getBoundingClientRect();
        bubble.style.left = `${box.left - area.left}px`;
        bubble.style.bottom = `${area.bottom - box.top + 14}px`;
        framed.push(`[data-node-id="${node.id}"] { outline: 3px solid #f5a623; outline-offset: 3px; }`);
    }
    if (frames.textContent !== framed.join("")) frames.textContent = framed.join("");
}

let following = false;
function follow() {
    place();
    following = marks.length > 0;
    if (following) requestAnimationFrame(follow);
}

function unmark(gone) {
    for (const { bubble } of gone) bubble.remove();
    marks = marks.filter(mark => !gone.includes(mark));
    place();
}

function mark(node, text) {
    const bubble = document.createElement("div");
    bubble.className = "impact-none-bubble";
    bubble.textContent = text;
    const close = document.createElement("button");
    close.title = close.ariaLabel = "Remove this mark";
    const made = { node, bubble };
    close.addEventListener("click", () => unmark([made]));
    bubble.append(close);
    layer.append(bubble);
    marks.push(made);
}

function nodeOf(executionId) {
    let graph = app.rootGraph ?? app.graph;
    let node;
    for (const id of String(executionId).split(":")) {
        node = graph?.getNodeById(id);
        graph = node?.subgraph;
    }
    return node;
}

app.registerExtension({
    name: "Comfy.Impact.SwitchNone",
    setup() {
        layer = document.createElement("div");
        layer.className = "impact-none-layer";
        app.canvas.canvas.parentElement.append(layer);
    }
});

api.addEventListener("execution_start", () => unmark(marks));
api.addEventListener("executing", ({ detail }) => gaveNone.delete(String(detail)));
api.addEventListener("impact-switch-none", ({ detail }) => gaveNone.set(String(detail.node_id), detail.select));

api.addEventListener("execution_error", async ({ detail }) => {
    const empty = Object.entries(detail.current_inputs ?? {}).filter(([, values]) => values?.length === 1 && values[0] === null).map(([name]) => name);
    if (!empty.length) return;
    const { output } = await app.graphToPrompt();
    for (const name of empty) {
        const source = output[detail.node_id]?.inputs?.[name];
        if (!Array.isArray(source) || output[source[0]]?.class_type !== "ImpactSwitch") continue;
        const inputs = output[source[0]].inputs;
        const select = gaveNone.get(String(source[0])) ?? (Number.isInteger(inputs.select) ? inputs.select : undefined);
        if (select === undefined) continue;
        const given = inputs[`input${select}`];
        const found = Array.isArray(given)
            ? [[source[0], `passed on the None of input${select}`], [given[0], "gave None to the Switch (Any)"]]
            : [[source[0], `returned None: select is ${select}, input${select} has no link`]];
        for (const [id, text] of found) {
            const node = nodeOf(id);
            if (node) mark(node, text);
        }
    }
    if (!following) follow();
});
