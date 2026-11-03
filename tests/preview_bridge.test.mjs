import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";
import { getFileItem } from "../js/image-path.js";

async function bridge(t) {
    const requests = [];
    const fetch = t.mock.fn(async () => new Response("$41-1"));
    const api = {
        fetchApi(url, options) { requests.push(url); return fetch(url, options); },
        apiURL(url) { return url; }
    };
    const context = createContext({ Image: class {}, URLSearchParams });
    const module = new SourceTextModule(await readFile(new URL("../js/preview-bridge.js", import.meta.url), "utf8"), { context });
    await module.link(async path => path.endsWith("/api.js")
        ? new SyntheticModule(["api"], function() { this.setExport("api", api); }, { context })
        : new SourceTextModule(await readFile(new URL(`../js/${path}`, import.meta.url), "utf8"), { context }));
    await module.evaluate();
    let storedValue = "";
    const widget = { name: "image", get value() { return storedValue; }, set value(value) { storedValue = value; } };
    const descriptor = Object.getOwnPropertyDescriptor(widget, "value");
    const node = { id: 41, widgets: [widget], properties: {}, graph: { setDirtyCanvas() {} } };
    module.namespace.setupPreviewBridge(node);
    node.onAdded();
    return { node, widget, descriptor, requests, fetch, restore: () => module.namespace.restorePreviewBridge(node), storeWrite(value) { storedValue = value; } };
}

test("ImageReceiver retains its temp default and accepts an explicit input reference", () => {
    assert.deepEqual(getFileItem("temp", "folder/source.png"), { filename: "source.png", subfolder: "folder", type: "temp" });
    assert.deepEqual(getFileItem("temp", "folder/source.png [input]"), { filename: "source.png", subfolder: "folder", type: "input" });
});

test("Vue store-only saves keep the filename and register before queueing", async t => {
    const { node, widget, descriptor, requests, storeWrite } = await bridge(t);
    assert.equal(Object.getOwnPropertyDescriptor(widget, "value").get, descriptor.get);
    assert.equal(Object.getOwnPropertyDescriptor(widget, "value").set, descriptor.set);
    assert.equal(widget.value, "$41-0");
    // Frontend 1.53.7 writes the store without invoking a widget callback.
    const filename = "masks/a & b+#한글.png [input]";
    storeWrite(filename);
    assert.equal(requests.length, 0);
    assert.equal(await widget.serializeValue(), "$41-1");
    assert.equal(widget.value, filename);
    assert.equal(await widget.serializeValue(), "$41-1");
    assert.equal(requests.length, 1, "Unchanged files should retain their prompt ID");
    const params = new URL(requests[0], "http://localhost").searchParams;
    assert.equal(params.get("node_id"), "41");
    assert.equal(params.get("filename"), "a & b+#한글.png");
    assert.equal(params.get("subfolder"), "masks");
    assert.equal(params.get("type"), "input");
    node.onExecuted({ source_image: ["$41-1"], images: [{ filename: "next.png", subfolder: "PreviewBridge", type: "temp" }] });
    assert.equal(widget.value, "PreviewBridge/next.png [temp]", "Editor must reopen the current upstream image");
});

for (const [value, item] of [
    ["plain.png", { filename: "plain.png", subfolder: "", type: "input" }],
    ["folder\\edited.png [INPUT]", { filename: "edited.png", subfolder: "folder", type: "input" }],
    ["folder/edited.png[OUTPUT] ", { filename: "edited.png", subfolder: "folder", type: "output" }]
]) {
    test(`registration and restoration use the native image reference: ${value}`, async t => {
        const { node, widget, requests, restore, storeWrite } = await bridge(t);
        storeWrite(value);
        await widget.serializeValue();
        assert.deepEqual(Object.fromEntries(new URL(requests[0], "http://localhost").searchParams), { node_id: "41", ...item });
        await restore();
        assert.deepEqual(JSON.parse(JSON.stringify(node.images)), [item]);
        assert.equal(widget.value, value);
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
    const { node, widget, storeWrite, requests } = await bridge(t);
    storeWrite("source.png [input]");
    const first = await widget.serializeValue();
    const second = await widget.serializeValue();
    assert.equal(requests.length, 1);
    node.onExecuted({ source_image: [first], images: [{ filename: "first-output.png", subfolder: "", type: "temp" }] });
    node.onExecuted({ source_image: [second], images: [{ filename: "second-output.png", subfolder: "", type: "temp" }] });
    assert.equal(widget.value, "second-output.png [temp]");
    await widget.serializeValue();
    assert.equal(new URL(requests[1], "http://localhost").searchParams.get("filename"), "second-output.png");
});

test("out-of-order registrations keep each prompt tied to its own image", async t => {
    const { node, widget, fetch, storeWrite } = await bridge(t);
    let finishFirst;
    fetch.mock.mockImplementationOnce(() => new Promise(resolve => { finishFirst = resolve; }));
    storeWrite("first.png [input]");
    const first = widget.serializeValue();
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
    const { widget, fetch, requests, storeWrite } = await bridge(t);
    storeWrite("edited.png [input]");
    fetch.mock.mockImplementationOnce(async () => new Response("", { status: 400 }));
    await assert.rejects(widget.serializeValue(), /image registration failed \(400\)/);
    assert.equal(widget.value, "edited.png [input]");
    assert.equal(await widget.serializeValue(), "$41-1");
    assert.equal(requests.length, 2);
});

test("native workflow restoration preserves legacy IDs and restores their preview", async t => {
    const { node, widget, fetch, restore, storeWrite, requests } = await bridge(t);
    const item = { filename: "old.png", subfolder: "PreviewBridge", type: "temp" };
    storeWrite("$41-previous");
    fetch.mock.mockImplementation(async () => Response.json(item));
    await restore();
    assert.equal(widget.value, "$41-previous");
    assert.equal(await widget.serializeValue(), "$41-previous");
    assert.equal(requests.length, 1);
    assert.deepEqual(JSON.parse(JSON.stringify(node.images)), [item]);
    assert.equal(new URL(node.imgs[0].src, "http://localhost").searchParams.get("filename"), "old.png");
});

test("native workflow restoration preserves Vue filenames without registration", async t => {
    const { node, widget, restore, storeWrite, requests } = await bridge(t);
    storeWrite("folder/edited.png [input]");
    await restore();
    assert.equal(widget.value, "folder/edited.png [input]");
    assert.equal(requests.length, 0);
    assert.equal(node.images[0].subfolder, "folder");
    assert.equal(node.images[0].type, "input");
});
