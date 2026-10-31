import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";
import { createSamSession, createSamTool } from "../js/sam-mask-editor.js";

function editor() {
    let revision = 1;
    let overlay = [];
    const commits = [];
    const controller = new AbortController();
    const image = { width: 3, height: 2, data: new Uint8ClampedArray(24).fill(255) };
    return {
        context: {
            signal: controller.signal,
            readImage: () => ({ image, revision }),
            setOverlay: points => { overlay = points; },
            commitMask(mask, options) {
                if (options.expectedRevision !== revision) return { ok: false, reason: "stale" };
                commits.push(mask);
                return { ok: true, revision: ++revision };
            }
        },
        commits,
        overlay: () => overlay,
        changeImage: () => revision++,
        abort: () => controller.abort()
    };
}

const mask = { width: 3, height: 2, data: new Uint8Array([0, 255, 0, 0, 255, 0]) };
const point = (x, y, button = 0) => ({ phase: "down", x, y, button, inBounds: true });

test("includes and excludes image points, confidence, and one mask commit", async () => {
    const doc = editor();
    let request;
    const session = createSamSession(doc.context, async (...args) => { request = args; return mask; });
    session.onPointer(point(1, 1));
    session.onPointer(point(2, 0, 2));
    session.setConfidence(85);
    await session.detect();
    assert.deepEqual(request[1], { positive_points: [[1, 1]], negative_points: [[2, 0]], threshold: 0.85 });
    assert.equal(request[0].width, 3);
    assert.equal(doc.commits.length, 1);
    assert.deepEqual(doc.commits[0], mask);
    assert.equal(doc.overlay().length, 2);
    assert.equal(session.state.message, "Mask applied.");
});

test("point undo leaves the mask intact; Clear removes both points and mask", async () => {
    const doc = editor();
    const session = createSamSession(doc.context, async () => mask);
    session.onPointer(point(1, 1));
    session.onPointer(point(2, 0, 2));
    await session.detect();
    session.undoPoint();
    assert.equal(session.state.points.length, 1);
    assert.equal(doc.commits.length, 1);
    session.clear();
    assert.equal(session.state.points.length, 0);
    assert.deepEqual(doc.overlay(), []);
    assert.deepEqual(doc.commits.at(-1).data, new Uint8Array(6));
});

for (const action of ["dispose", "clear", "onDocumentChange", "undoPoint"]) {
    test(`${action} cancels an in-flight result, even when inference ignores cancellation`, async () => {
        const doc = editor();
        let finish;
        const pending = new Promise(resolve => { finish = resolve; });
        let signal;
        const session = createSamSession(doc.context, async (_, __, options) => {
            signal = options.signal;
            return pending;
        });
        session.onPointer(point(1, 1));
        const request = session.detect();
        assert.equal(session.state.busy, true);
        session[action]();
        const commits = doc.commits.length;
        finish(mask);
        await request;
        assert.equal(signal.aborted, true);
        assert.equal(session.state.busy, false);
        assert.equal(doc.commits.length, commits);
    });
}

test("document revision checks refuse stale masks", async () => {
    const doc = editor();
    const session = createSamSession(doc.context, async () => { doc.changeImage(); return mask; });
    session.onPointer(point(1, 1));
    await session.detect();
    assert.equal(doc.commits.length, 0);
    assert.match(session.state.message, /changed/i);
});

test("network failure is visible and can be retried", async () => {
    const doc = editor();
    let attempts = 0;
    const session = createSamSession(doc.context, async () => {
        if (!attempts++) throw new Error("SAM model missing");
        return mask;
    });
    session.onPointer(point(1, 1));
    await session.detect();
    assert.match(session.state.message, /SAM model missing/);
    assert.equal(session.state.busy, false);
    await session.detect();
    assert.equal(doc.commits.length, 1);
});

test("ignores navigation, hover and outside points; supports pen exclusion mode", () => {
    const session = createSamSession(editor().context, async () => mask);
    session.onPointer({ ...point(1, 1), phase: "move" });
    session.onPointer({ ...point(1, 1), inBounds: false });
    session.onPointer(point(1, 1, 1));
    assert.equal(session.state.points.length, 0);
    session.setPointMode("exclude");
    session.onPointer(point(1, 1));
    assert.equal(session.state.points[0].positive, false);
});

for (const available of [true, false]) {
    test(`registration uses the public API when available: ${available}`, async () => {
        const tools = [];
        const extensions = [];
        const legacy = [];
        const context = createContext({ console });
        const app = {
            extensionManager: available ? { maskEditor: {
                registerTool(tool) { tools.push(tool); return { ok: true }; }
            } } : {},
            registerExtension(extension) { extensions.push(extension); }
        };
        const module = new SourceTextModule(await readFile(new URL("../js/impact-sam-editor.js", import.meta.url), "utf8"), {
            context,
            async importModuleDynamically(path) {
                legacy.push(path);
                const fallback = new SyntheticModule([], () => {}, { context });
                await fallback.link(() => {});
                await fallback.evaluate();
                return fallback;
            }
        });
        await module.link(path => {
            const [name, value] = path.endsWith("/app.js") ? ["app", app]
                : path.endsWith("/api.js") ? ["api", {}] : ["createSamTool", createSamTool];
            return new SyntheticModule([name], function() { this.setExport(name, value); }, { context });
        });
        await module.evaluate();
        if (available) {
            assert.equal(extensions.length, 1);
            extensions[0].setup();
            assert.deepEqual(tools.map(tool => tool.id), ["impact.sam"]);
            assert.deepEqual(legacy, []);
        } else {
            assert.deepEqual(extensions, []);
            assert.deepEqual(legacy, ["./legacy-sam-editor.mjs"]);
        }
    });
}
