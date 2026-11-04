import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

async function createBridge(t, comfyClass) {
	let extension;
	const requests = [];
	const ids = ["$41-1", "$41-2"];
	const app = { registerExtension(value) { extension = value; } };
	const api = {
		async fetchApi(path) {
			requests.push(path);
			return { status: 200, text: async () => ids.shift() };
		}
	};
	const context = createContext({ Image: class { src = ""; } });
	const imports = {
		"../../scripts/app.js": { app, ComfyApp: {} },
		"../../scripts/api.js": { api }
	};
	const source = await readFile(new URL("../js/impact-image-util.js", import.meta.url), "utf8");
	const module = new SourceTextModule(source, { context });
	await module.link((specifier) => {
		const exports = imports[specifier];
		return new SyntheticModule(Object.keys(exports), function () {
			for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
		}, { context });
	});
	await module.evaluate();
	const callback = t.mock.fn();
	const widget = { name: "image", value: "", callback };
	const node = { id: 41, comfyClass, widgets: [widget] };
	extension.nodeCreated(node, app);
	return { node, widget, callback, requests };
}

function callbackOnly(widget, value, node) {
	widget.callback(value, node);
}

for (const { name, comfyClass, update, callbackCount } of [
	{ name: "PreviewBridge callback-only saves", comfyClass: "PreviewBridge", update: callbackOnly, callbackCount: 2 },
	{ name: "PreviewBridgeLatent callback-only saves", comfyClass: "PreviewBridgeLatent", update: callbackOnly, callbackCount: 2 },
	{
		name: "setter followed by callback does not register twice",
		comfyClass: "PreviewBridge",
		update(widget, value, node) { widget.value = value; widget.callback(value, node); },
		callbackCount: 2
	},
	{
		name: "legacy direct assignment still registers masks",
		comfyClass: "PreviewBridge",
		update(widget, value) { widget.value = value; },
		callbackCount: 0
	}
]) {
	test(name, async (t) => {
		const { node, widget, callback, requests } = await createBridge(t, comfyClass);

		update(widget, "clipspace/first-mask.png [input]", node);
		assert.equal(requests.length, 1);
		await setImmediate();
		assert.equal(widget.value, "$41-1");

		update(widget, "clipspace/second-mask.png [input]", node);
		await setImmediate();
		assert.equal(widget.value, "$41-2");
		assert.deepEqual(requests, [
			"/impact/set/pb_id_image?node_id=41&filename=first-mask.png&type=input&subfolder=clipspace",
			"/impact/set/pb_id_image?node_id=41&filename=second-mask.png&type=input&subfolder=clipspace"
		]);
		assert.deepEqual(callback.mock.calls.map(call => call.arguments), [
			["clipspace/first-mask.png [input]", node],
			["clipspace/second-mask.png [input]", node]
		].slice(0, callbackCount));
		assert.deepEqual(callback.mock.calls.map(call => call.this), Array(callbackCount).fill(widget));
	});
}
