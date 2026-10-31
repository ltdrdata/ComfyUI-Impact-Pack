import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

async function bridge(t, comfyClass = "PreviewBridge") {
    const requests = [];
    const fetch = t.mock.fn(async () => new Response("$41-1"));
    const api = {
        fetchApi(url, options) { requests.push(url); return fetch(url, options); },
        apiURL(url) { return url; }
    };
    const context = createContext({ Image: class {}, URLSearchParams });
    const module = new SourceTextModule(await readFile(new URL("../js/preview-bridge.js", import.meta.url), "utf8"), { context });
    await module.link(() => new SyntheticModule(["api"], function() { this.setExport("api", api); }, { context }));
    await module.evaluate();
    let storedValue = "";
    const widget = { name: "image", get value() { return storedValue; }, set value(value) { storedValue = value; } };
    const descriptor = Object.getOwnPropertyDescriptor(widget, "value");
    const node = { id: 41, comfyClass, widgets: [widget], properties: {}, graph: { setDirtyCanvas() {} } };
    module.namespace.setupPreviewBridge(node);
    node.onAdded();
    return { node, widget, descriptor, requests, fetch, restore: () => module.namespace.restorePreviewBridge(node), storeWrite(value) { storedValue = value; } };
}

for (const nodeType of ["PreviewBridge", "PreviewBridgeLatent"]) {
    test(`${nodeType}: Vue store-only saves keep the filename and register before queueing`, async t => {
        const { node, widget, descriptor, requests, storeWrite } = await bridge(t, nodeType);
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
        node.onExecuted({ images: [{ filename: "next.png", subfolder: "PreviewBridge", type: "temp" }] });
        assert.equal(widget.value, "PreviewBridge/next.png [temp]", "Editor must reopen the current upstream image");
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
    node.onExecuted({ images: [{ filename: "first.png", subfolder: "", type: "input" }] });
    assert.equal(widget.value, "second.png [input]");
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
