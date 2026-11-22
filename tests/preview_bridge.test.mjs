import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

async function extension(t) {
    const fetch = t.mock.fn(async () => new Response("$41-1"));
    const api = Object.assign(new EventTarget(), {
        fetchApi: fetch,
        apiURL(url) { return url; }
    });
    let registered;
    const decoded = t.mock.fn(async () => {});
    const toast = t.mock.fn();
    const command = { id: "Comfy.MaskEditor.OpenMaskEditor", function: t.mock.fn() };
    const app = { registerExtension(value) { registered = value; }, canvas: { selected_nodes: {} },
        extensionManager: { toast: { add: toast }, command: { commands: [command] } } };
    const context = createContext({ Image: class { src = ""; decode() { return decoded(this.src); } }, URLSearchParams, Promise, Math,
        document: { createElement: () => ({ getContext: () => ({ drawImage() {} }), toDataURL: () => "data:image/png;base64,saved" }) } });
    const modules = new Map();
    async function load(path) {
        if (!modules.has(path)) {
            const values = path === "../../scripts/api.js" ? { api }
                : path === "../../scripts/app.js" ? { app } : null;
            modules.set(path, values
                ? new SyntheticModule(Object.keys(values), function() {
                    for (const [key, value] of Object.entries(values)) this.setExport(key, value);
                }, { context })
                : new SourceTextModule(await readFile(new URL(`../js/${path}`, import.meta.url), "utf8"), { context }));
        }
        return modules.get(path);
    }
    const module = await load("./impact-image-util.js");
    await module.link(load);
    await module.evaluate();
    return { registered, fetch, api, toast, decoded, app, command };
}

async function bridge(t, configurable = true) {
    const { registered, fetch, api, toast, decoded } = await extension(t);
    let storedValue = "";
    const widget = { name: "image" };
    Object.defineProperty(widget, "value", {
        get() { return storedValue; }, set(value) { storedValue = value; }, configurable,
    });
    const descriptor = Object.getOwnPropertyDescriptor(widget, "value");
    const node = { id: 41, comfyClass: "PreviewBridge", widgets: [widget], properties: {}, graph: { setDirtyCanvas() {} } };
    registered.nodeCreated(node);
    node.onAdded?.();
    return { node, widget, descriptor, fetch, api, toast, decoded, restore: () => registered.loadedGraphNode?.(node), storeWrite(value) { storedValue = value; } };
}

test("ImageReceiver retains PR1172's protection for a non-configurable value", async t => {
    const { registered } = await extension(t);
    const imageData = { name: "image_data" };
    Object.defineProperty(imageData, "value", { value: "saved-image-data", writable: true });
    const widgets = [{ name: "image", value: "saved.png [temp]" }, imageData,
        { name: "save_to_workflow", value: true }];
    const before = widgets.map(({ name, value }) => ({ name, value }));
    registered.nodeCreated({ comfyClass: "ImageReceiver", widgets });
    assert.deepEqual(widgets.map(({ name, value }) => ({ name, value })), before);
    assert.equal(Object.getOwnPropertyDescriptor(imageData, "value").configurable, false);
});

test("ImageReceiver keeps its existing image capture and save switch", async t => {
    const { registered } = await extension(t);
    const widgets = [{ name: "image", value: "source.png [temp]" },
        { name: "image_data", value: "" }, { name: "save_to_workflow", value: false }];
    const node = { comfyClass: "ImageReceiver", widgets };
    registered.nodeCreated(node);
    node.imgs = [{ complete: true, width: 2, height: 2 }];
    assert.equal(widgets[1].value, "");
    widgets[2].value = true;
    assert.equal(widgets[1].value, "data:image/png;base64,saved");
    widgets[2].value = false;
    assert.equal(widgets[1].value, "");
    assert.deepEqual(widgets.map(widget => widget.name), ["image", "image_data", "save_to_workflow"]);
});

for (const configurable of [true, false]) {
    test(`Vue store-only saves preserve the native binding (configurable=${configurable})`, async t => {
        const { node, widget, descriptor, fetch, storeWrite } = await bridge(t, configurable);
        assert.deepEqual(Object.getOwnPropertyDescriptor(widget, "value"), descriptor);
        assert.equal(widget.value, "$41-0");
        const filename = "masks/a & b+#한글.png [input]";
        storeWrite(filename);
        assert.equal(fetch.mock.callCount(), 0);
        assert.equal(await widget.serializeValue(), "$41-1");
        assert.equal(widget.value, filename);
        assert.equal(await widget.serializeValue(), "$41-1");
        assert.equal(fetch.mock.callCount(), 1, "Unchanged files should retain their prompt ID");
        const params = new URL(fetch.mock.calls[0].arguments[0], "http://localhost").searchParams;
        assert.equal(params.get("node_id"), "41");
        assert.equal(params.get("filename"), "a & b+#한글.png");
        assert.equal(params.get("subfolder"), "masks");
        assert.equal(params.get("type"), "input");
        node.onExecuted({ source_image: ["$41-1"], images: [{ filename: "next.png", subfolder: "PreviewBridge", type: "temp" }] });
        assert.equal(widget.value, "PreviewBridge/next.png [temp]", "Editor must reopen the current upstream image");
    });
}

for (const [value, item] of [
    ["plain.png", { filename: "plain.png", subfolder: "", type: "input" }],
    ["clipspace-mask.png [temp]", { filename: "clipspace-mask.png", subfolder: "", type: "temp" }],
    ["nested/edited.png [temp]", { filename: "edited.png", subfolder: "nested", type: "temp" }],
    ["folder\\edited.png [INPUT]", { filename: "edited.png", subfolder: "folder", type: "input" }],
    ["folder/edited.png[OUTPUT] ", { filename: "edited.png", subfolder: "folder", type: "output" }]
]) {
    test(`registration and restoration use the native image reference: ${value}`, async t => {
        const { node, widget, fetch, restore, storeWrite } = await bridge(t);
        storeWrite(value);
        await restore();
        assert.deepEqual(JSON.parse(JSON.stringify(node.images)), [item]);
        assert.equal(widget.value, value);
        assert.equal(fetch.mock.callCount(), 0, "Filename restoration must not register");
        assert.deepEqual(Object.fromEntries(new URL(node.imgs[0].src, "http://localhost").searchParams), item);
        await widget.serializeValue();
        assert.deepEqual(Object.fromEntries(new URL(fetch.mock.calls[0].arguments[0], "http://localhost").searchParams), { node_id: "41", ...item });
    });
}

test("queue serialization waits for registration and leaves a newer edit intact", async t => {
    const { node, widget, fetch, storeWrite } = await bridge(t);
    let finish;
    fetch.mock.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    storeWrite("first.png [input]");
    let queued = false;
    const submission = widget.serializeValue().then(value => { queued = true; return value; });
    await Promise.resolve();
    assert.equal(queued, false);
    storeWrite("second.png [input]");
    finish(new Response("$41-2"));
    assert.equal(await submission, "$41-2");
    node.onExecuted({ source_image: ["$41-2"], images: [{ filename: "first.png", subfolder: "", type: "input" }] });
    assert.equal(widget.value, "second.png [input]");
});

test("an earlier queued result cannot overwrite an edit submitted afterward", async t => {
    const { node, widget, fetch, storeWrite } = await bridge(t);
    fetch.mock.mockImplementation(async url => new Response(
        new URL(url, "http://localhost").searchParams.get("filename") === "first.png" ? "$41-first" : "$41-second"));
    storeWrite("first.png [input]");
    const first = await widget.serializeValue();
    storeWrite("second.png [input]");
    const second = await widget.serializeValue();
    node.onExecuted({ source_image: [first], images: [{ filename: "first.png", subfolder: "", type: "input" }] });
    assert.equal(widget.value, "second.png [input]");
    node.onExecuted({ source_image: [second], images: [{ filename: "second-output.png", subfolder: "", type: "temp" }] });
    assert.equal(widget.value, "second-output.png [temp]");
    assert.equal(node.properties.image, "second-output.png [temp]");
});

test("queued refreshes of the same image advance the editor to the latest output", async t => {
    const { node, widget, storeWrite, fetch } = await bridge(t);
    storeWrite("source.png [input]");
    const first = await widget.serializeValue();
    const second = await widget.serializeValue();
    assert.equal(fetch.mock.callCount(), 1);
    node.onExecuted({ source_image: [first], images: [{ filename: "first-output.png", subfolder: "", type: "temp" }] });
    node.onExecuted({ source_image: [second], images: [{ filename: "second-output.png", subfolder: "", type: "temp" }] });
    assert.equal(widget.value, "second-output.png [temp]");
    await widget.serializeValue();
    assert.equal(new URL(fetch.mock.calls[1].arguments[0], "http://localhost").searchParams.get("filename"), "second-output.png");
});

test("out-of-order registrations keep each prompt tied to its own image", async t => {
    const { node, widget, fetch, storeWrite } = await bridge(t);
    let finishFirst;
    fetch.mock.mockImplementationOnce(() => new Promise(resolve => { finishFirst = resolve; }));
    storeWrite("first.png [input]");
    const first = widget.serializeValue();
    assert.equal(typeof finishFirst, "function", "Serialization must start registration before submitting the prompt");
    storeWrite("second.png [input]");
    fetch.mock.mockImplementationOnce(async () => new Response("$41-second"));
    assert.equal(await widget.serializeValue(), "$41-second");
    finishFirst(new Response("$41-first"));
    assert.equal(await first, "$41-first");
    node.onExecuted({ source_image: ["$41-first"], images: [{ filename: "first-output.png", subfolder: "", type: "temp" }] });
    assert.equal(widget.value, "second.png [input]");
    node.onExecuted({ source_image: ["$41-second"], images: [{ filename: "second-output.png", subfolder: "", type: "temp" }] });
    assert.equal(widget.value, "second-output.png [temp]");
});

test("failed registration rejects submission and can be retried", async t => {
    const { widget, fetch, storeWrite } = await bridge(t);
    storeWrite("edited.png [input]");
    fetch.mock.mockImplementationOnce(async () => new Response("", { status: 400 }));
    await assert.rejects(widget.serializeValue(), /image registration failed \(400\)/);
    assert.equal(widget.value, "edited.png [input]");
    assert.equal(await widget.serializeValue(), "$41-1");
    assert.equal(fetch.mock.callCount(), 2);
});

test("native workflow restoration preserves legacy IDs and restores their preview", async t => {
    const { node, widget, fetch, restore, storeWrite } = await bridge(t);
    const item = { filename: "old.png", subfolder: "PreviewBridge", type: "temp" };
    storeWrite("$41-previous");
    fetch.mock.mockImplementation(async () => Response.json(item));
    await restore();
    assert.equal(widget.value, "$41-previous");
    assert.equal(await widget.serializeValue(), "$41-previous");
    assert.equal(fetch.mock.callCount(), 1);
    assert.deepEqual(JSON.parse(JSON.stringify(node.images)), [item]);
    assert.equal(new URL(node.imgs[0].src, "http://localhost").searchParams.get("filename"), "old.png");
});

test("reconnection registers the unchanged filename again and ignores old feedback", async t => {
    const { node, widget, api, fetch, storeWrite } = await bridge(t);
    storeWrite("clipspace/saved.png [input]");
    const first = await widget.serializeValue();
    api.dispatchEvent(new Event("reconnected"));
    node.onExecuted({ source_image: [first], images: [{ filename: "stale.png", type: "temp" }] });
    assert.equal(widget.value, "clipspace/saved.png [input]");
    fetch.mock.mockImplementationOnce(async () => new Response("$41-reconnected"));
    const second = await widget.serializeValue();
    assert.equal(second, "$41-reconnected");
    assert.equal(fetch.mock.callCount(), 2);
    assert.equal(await widget.serializeValue(), second, "Ordinary queues still reuse the new registration");
    assert.equal(fetch.mock.callCount(), 2);
    node.onExecuted({ source_image: [first], images: [{ filename: "stale.png", type: "temp" }] });
    assert.equal(widget.value, "clipspace/saved.png [input]");
});

for (const change of ["new edit", "removed node"]) {
    test(`pending legacy restoration does not overwrite a ${change}`, async t => {
        const { node, fetch, restore, storeWrite } = await bridge(t);
        storeWrite("$41-old");
        let finish;
        fetch.mock.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
        const restoring = restore();
        if (change === "new edit") storeWrite("new.png [input]");
        else node.graph = null;
        const images = node.imgs;
        finish(Response.json({ filename: "old.png", type: "temp", subfolder: "" }));
        await restoring;
        assert.equal(node.imgs, images);
        assert.equal(node.images, undefined);
    });
}

test("an image saved as an object by the first versions of the bridge is a bridge without an image", async t => {
    const { node, widget, fetch, restore, storeWrite } = await bridge(t);
    storeWrite({ filename: "clipspace-mask-1.png", subfolder: "clipspace", type: "input", image_hash: 1,
        forward_filename: "ComfyUI_temp_00001_.png", forward_subfolder: "", forward_type: "temp" });
    await restore();
    assert.equal(await widget.serializeValue(), "$41-0");
    assert.equal(fetch.mock.callCount(), 0);
    node.onExecuted({ source_image: ["$41-0"], images: [{ filename: "next.png", subfolder: "PreviewBridge", type: "temp" }] });
    assert.equal(widget.value, "PreviewBridge/next.png [temp]", "The run gives the bridge its image, as it does for a new bridge");
});

test("a malformed filename rejects the queue before sending registration", async t => {
    const { widget, fetch, storeWrite } = await bridge(t);
    storeWrite("folder/[input]");
    await assert.rejects(widget.serializeValue(), /no image to register/);
    assert.equal(fetch.mock.callCount(), 0);
});

test("a mask kept over transparent pixels after alpha_as_mask is switched off raises one warning", async t => {
    const { node, toast } = await bridge(t);
    const item = { filename: "kept.png", subfolder: "", type: "input" };
    node.onExecuted({ source_image: ["$41-1"], images: [item], alpha_in_mask: [false] });
    node.onExecuted({ source_image: ["$41-1"], images: [item] });
    assert.equal(toast.mock.callCount(), 0);
    node.onExecuted({ source_image: ["$41-1"], images: [item], alpha_in_mask: [true] });
    assert.equal(toast.mock.callCount(), 1);
    const [{ severity, detail }] = toast.mock.calls[0].arguments;
    assert.equal(severity, "warn");
    assert.match(detail, /alpha_as_mask was switched off/);
});

test("switching alpha_as_mask shows the picture of the other setting at once", async t => {
    const { registered, decoded } = await extension(t);
    const original = t.mock.fn(() => "the widget's own result");
    const option = { name: "alpha_as_mask", value: false, callback: original };
    let redrawn = 0;
    const node = { id: 41, comfyClass: "PreviewBridge", widgets: [{ name: "image", value: "" }, option], properties: {}, graph: { setDirtyCanvas() { redrawn++; } } };
    registered.nodeCreated(node);
    const sources = () => node.imgs.map(image => new URL(image.src, "http://localhost"));
    const switched = async value => { option.value = value; option.callback(value); await new Promise(resolve => setTimeout(resolve)); };
    const before = node.imgs;
    await switched(true);
    assert.equal(node.imgs, before);
    assert.equal(option.callback(true, "canvas", node), "the widget's own result");
    assert.deepEqual(original.mock.calls.at(-1).arguments, [true, "canvas", node]);
    const images = ["a", "b"].map(name => ({ filename: `${name}.png`, subfolder: "PreviewBridge", type: "temp", metadata: {} }));
    node.images = images;
    node.onExecuted({ images, alpha_as_mask: [false] });
    await switched(true);
    assert.deepEqual(sources().map(url => [url.pathname, url.searchParams.get("node_id"), url.searchParams.get("alpha_as_mask"), url.searchParams.get("index")]),
        [["/impact/view/pb_alpha_preview", "41", "true", "0"], ["/impact/view/pb_alpha_preview", "41", "true", "1"]]);
    assert.equal(redrawn, 1);
    await switched(false);
    assert.deepEqual(sources().map(url => [url.pathname, url.searchParams.get("filename"), url.searchParams.get("subfolder"), url.searchParams.get("type")]),
        [["/view", "a.png", "PreviewBridge", "temp"], ["/view", "b.png", "PreviewBridge", "temp"]]);
    const shownNow = node.imgs;
    decoded.mock.mockImplementation(async () => { throw new Error("404"); });
    await switched(true);
    assert.equal(node.imgs, shownNow);
    decoded.mock.mockImplementation(async () => {});
    node.images = [{ filename: "clipspace-painted-masked-1.png", subfolder: "clipspace", type: "input" }];
    await switched(false);
    await switched(true);
    assert.equal(node.imgs, shownNow);
    node.images = images;
    const release = [];
    decoded.mock.mockImplementation(() => new Promise(resolve => { release.push(resolve); }));
    option.value = true; option.callback(true);
    option.value = false;
    assert.equal(release.length, 2);
    release.splice(0).forEach(resolve => resolve());
    await new Promise(resolve => setTimeout(resolve));
    assert.equal(node.imgs, shownNow);
    option.value = true; option.callback(true);
    node.onExecuted({ images: [...images], alpha_as_mask: [true] });
    release.splice(0).forEach(resolve => resolve());
    await new Promise(resolve => setTimeout(resolve));
    assert.equal(node.imgs, shownNow);
});

test("the image selected on the node is the one of a batch that the mask editor opens", async t => {
    const { node, widget, fetch, storeWrite } = await bridge(t);
    const images = ["a", "b", "c"].map(name => ({ filename: `${name}.png`, subfolder: "PreviewBridge", type: "temp" }));
    storeWrite("source.png [input]");
    node.onExecuted({ source_image: [await widget.serializeValue()], images });
    assert.equal(widget.value, "PreviewBridge/a.png [temp]");
    node.imageIndex = 2;
    assert.equal(node.imageIndex, 2);
    assert.equal(widget.value, "PreviewBridge/c.png [temp]");
    assert.equal(node.properties.image, "PreviewBridge/c.png [temp]");
    node.imageIndex = null;
    assert.equal(widget.value, "PreviewBridge/a.png [temp]");
    node.imageIndex = 1;
    await widget.serializeValue();
    assert.equal(new URL(fetch.mock.calls.at(-1).arguments[0], "http://localhost").searchParams.get("filename"), "b.png");
    storeWrite("clipspace/clipspace-painted-masked-1.png [input]");
    node.imageIndex = 0;
    assert.equal(widget.value, "clipspace/clipspace-painted-masked-1.png [input]");
});

test("a bridge that shows no image explains it where the mask editor would fail", async t => {
    const { registered, toast, app, command } = await extension(t);
    const openCommand = command.function;
    const openEntry = t.mock.fn();
    const copy = () => {};
    const node = { id: 41, comfyClass: "PreviewBridge", widgets: [{ name: "image", value: "" }], properties: {}, graph: { setDirtyCanvas() {} },
        getExtraMenuOptions(canvas, options) {
            options.push(null, { content: "Open in MaskEditor | Image Canvas", callback: openEntry }, { content: "Copy (Clipspace)", callback: copy });
        } };
    registered.nodeCreated(node);
    registered.setup();
    app.canvas.selected_nodes = { 41: node };
    const entries = () => { const options = []; node.getExtraMenuOptions(null, options); return options; };

    entries()[1].callback();
    command.function();
    assert.equal(toast.mock.callCount(), 2);
    assert.match(toast.mock.calls[0].arguments[0].detail, /It gets one when the node itself is executed/);
    assert.equal(openEntry.mock.callCount() + openCommand.mock.callCount(), 0);
    assert.equal(entries()[2].callback, copy, "other menu entries are left as they are");

    node.imgs[0].src = "/view?filename=a.png";
    entries()[1].callback("event");
    command.function();
    assert.equal(toast.mock.callCount(), 2);
    assert.equal(openEntry.mock.callCount(), 1);
    assert.equal(openEntry.mock.calls[0].arguments[0], "event");
    assert.equal(openCommand.mock.callCount(), 1);

    Object.assign(node.imgs[0], { complete: true, naturalWidth: 0 });
    entries()[1].callback();
    assert.equal(toast.mock.callCount(), 3);
    assert.equal(openEntry.mock.callCount(), 1);

    app.canvas.selected_nodes = { 7: { id: 7, comfyClass: "LoadImage" } };
    command.function();
    app.canvas.selected_nodes = { 7: { id: 7, comfyClass: "LoadImage" }, 41: node };
    command.function();
    assert.equal(openCommand.mock.callCount(), 3);
    assert.equal(toast.mock.callCount(), 3);
});
