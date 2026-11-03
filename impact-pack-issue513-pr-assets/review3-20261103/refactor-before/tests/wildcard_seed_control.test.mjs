import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/impact-pack.js', import.meta.url), 'utf8');
const catalog = JSON.parse(readFileSync(new URL('./workflows/all-impact-nodes.widgets.json', import.meta.url)));

async function setup(type, beforeQueued = true, failure, serializationFailure) {
    let extension;
    const context = vm.createContext({ window: {}, console, queueMicrotask });
    const app = { registerExtension(value) { extension = value; } };
    const imports = {
        '../../scripts/app.js': { app, ComfyApp: {} },
        '../../scripts/api.js': { api: {
            addEventListener() {},
            fetchApi: async () => ({ json: async () => ({ data: [], on_demand_mode: false }) }),
        } },
        './common.js': { customAlert() { assert.fail('Unexpected frontend warning'); }, isBeforeFrontendVersion: () => false },
    };
    const module = new vm.SourceTextModule(source, { context });
    await module.link(async specifier => {
        assert.ok(imports[specifier], `Unexpected import: ${specifier}`);
        const values = imports[specifier];
        return new vm.SyntheticModule(Object.keys(values), function () {
            for (const [key, value] of Object.entries(values)) this.setExport(key, value);
        }, { context });
    });
    await module.evaluate();
    const widgets = catalog.nodes.find(node => node.type === type).widgets.map(widget => ({
        ...widget, inputEl: {}, options: {},
    }));
    const node = { comfyClass: type, widgets, isSubgraphNode: () => false };
    const seed = widgets.find(widget => widget.name === 'seed');
    const mode = widgets.find(widget => widget.name === 'mode');
    const control = widgets.find(widget => widget.name === 'control_after_generate');
    seed.value = 42;
    control.value = 'increment';
    seed.linkedWidgets = [control];
    const calls = [];
    const result = {};
    const serializations = [];
    const draws = [];
    app.canvas = { draw() { draws.push(control.value); } };
    app.graphToPrompt = function (...args) {
        serializations.push({ receiver: this, args, policy: control.value });
        if (serializationFailure) throw serializationFailure;
        return result;
    };
    extension.setup?.();
    // Spy on the core boundary; actual seed arithmetic/bookkeeping is covered
    // by Chrome queues, rather than copied into a second implementation here.
    for (const hook of beforeQueued ? ['beforeQueued', 'afterQueued'] : ['afterQueued']) {
        control[hook] = function (...args) {
            calls.push({ hook, receiver: this, args, value: this.value });
            if (failure) throw failure;
            return result;
        };
    }
    const before = widgets.map(({ name, value }) => ({ name, value }));
    extension.nodeCreated(node, app);
    assert.deepEqual(widgets.map(({ name, value }) => ({ name, value })), before);
    return { node, seed, mode, control, calls, result, serializations, draws, extension, app };
}

function promotedHost(node, extension, app, modeValue, projection) {
    node.id ??= 1;
    const inputs = [{ name: 'seed', link: null, widgetId: `host-${node.id}-seed` }];
    const widgets = { seed: { name: 'seed', value: 42 } };
    if (modeValue !== undefined) {
        inputs.push({ name: 'mode', link: null, widgetId: `host-${node.id}-mode` });
        widgets.mode = { name: 'mode', value: modeValue };
    }
    node.getWidgetFromSlot ??= input => node.widgets.find(w => w.name === input.name);
    const host = {
        id: Number(node.id) + 1, inputs,
        isSubgraphNode: () => true,
        subgraph: {
            inputNode: { slots: inputs.map((input, i) => ({ name: input.name, linkIds: [i] })) },
            getLink: i => ({ resolve: () => ({ inputNode: node, input: inputs[i] }) }),
        },
        getWidgetFromSlot: input => projection?.(input) ?? widgets[input.name],
    };
    extension.nodeCreated(host, app);
    return host;
}

for (const type of ['ImpactWildcardProcessor', 'ImpactWildcardEncode']) {
    test(`${type}: fixed and legacy false preserve the core hook contract`, async () => {
        const { mode, control, calls, result } = await setup(type);
        for (const value of ['fixed', false]) {
            mode.value = value;
            for (const selected of ['fixed', 'increment', 'decrement', 'randomize']) {
                control.value = selected;
                for (const hook of ['beforeQueued', 'afterQueued']) {
                    const argument = { isPartialExecution: true };
                    const count = calls.length;
                    assert.equal(control[hook](argument), result);
                    assert.equal(calls.length, count + 1);
                    assert.deepEqual(calls.at(-1), { hook, receiver: control, args: [argument], value: 'fixed' });
                    assert.equal(control.value, selected);
                }
            }
        }
    });

    test(`${type}: populate and reproduce delegate unchanged to core hooks`, async () => {
        const { mode, control, calls, result } = await setup(type);
        for (const value of ['populate', 'reproduce', true]) {
            mode.value = value;
            for (const hook of ['beforeQueued', 'afterQueued']) {
                const argument = {};
                assert.equal(control[hook](argument), result);
                assert.deepEqual(calls.at(-1), { hook, receiver: control, args: [argument], value: 'increment' });
            }
        }
        assert.equal(calls.length, 6);
    });

    test(`${type}: each call uses the current mode and preserves a manually edited seed`, async () => {
        const { seed, mode, control, calls } = await setup(type);
        mode.value = 'fixed';
        seed.value = 123;
        control.afterQueued();
        assert.equal(calls.at(-1).value, 'fixed');
        mode.value = 'populate';
        control.afterQueued();
        assert.equal(calls.at(-1).value, 'increment');
        assert.equal(calls.length, 2);
        assert.equal(seed.value, 123);
        assert.equal(control.value, 'increment');
    });

    test(`${type}: a core hook error propagates and restores the selected policy`, async () => {
        const error = new Error('core hook failed');
        const { mode, control, calls } = await setup(type, true, error);
        mode.value = 'fixed';
        for (const selected of ['increment', 'decrement', 'randomize']) {
            control.value = selected;
            for (const hook of ['beforeQueued', 'afterQueued']) {
                assert.throws(() => control[hook](), thrown => thrown === error);
                assert.equal(control.value, selected);
                assert.equal(calls.at(-1).value, 'fixed');
            }
        }
    });

    test(`${type}: older controls with only afterQueued remain supported`, async () => {
        const { mode, control, calls } = await setup(type, false);
        mode.value = 'fixed';
        control.afterQueued();
        assert.equal(calls.at(-1).value, 'fixed');
        mode.value = 'populate';
        control.afterQueued();
        assert.equal(calls.at(-1).value, 'increment');
        assert.equal(calls.length, 2);
        assert.equal(control.beforeQueued, undefined);
    });

    test(`${type}: promoted controls pause through the host phase and restore before serialization`, async () => {
        const { node, mode, control, result, app, serializations, extension } = await setup(type);
        const host = promotedHost(node, extension, app);
        mode.value = 'fixed';
        control.value = 'randomize';
        assert.equal(control.beforeQueued(), result);
        host.getWidgetFromSlot(host.inputs[0]);
        // The core reads the policy outside the hook to update the host seed.
        assert.equal(control.value, 'fixed');
        const argument = {};
        assert.equal(app.graphToPrompt(argument), result);
        assert.deepEqual(serializations, [{ receiver: app, args: [argument], policy: 'randomize' }]);
        assert.equal(control.value, 'randomize');
        await Promise.resolve();
        assert.equal(control.value, 'randomize');
        control.afterQueued();
        host.getWidgetFromSlot(host.inputs[0]);
        assert.equal(control.value, 'fixed');
        await Promise.resolve();
        assert.equal(control.value, 'randomize');
    });

    test(`${type}: unavailable host mode restores an already paused shared policy`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        const fixed = promotedHost(node, extension, app, 'fixed');
        const unknown = promotedHost(node, extension, app, 'populate');
        const input = unknown.inputs[0];
        mode.value = 'populate';
        control.value = 'decrement';
        control.afterQueued();
        fixed.getWidgetFromSlot(fixed.inputs[0]);
        assert.equal(control.value, 'fixed');
        unknown.inputs = undefined;
        unknown.getWidgetFromSlot(input);
        assert.equal(control.value, 'decrement'); // Core consumes it synchronously.
        await Promise.resolve();
        assert.equal(control.value, 'decrement');
    });

    test(`${type}: a missing promoted mode does not imply the interior fixed mode`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        const fixed = promotedHost(node, extension, app, 'fixed');
        const unknown = promotedHost(node, extension, app, 'populate');
        const original = unknown.getWidgetFromSlot;
        unknown.getWidgetFromSlot = function (input) {
            return input.name === 'mode' ? undefined : original.call(this, input);
        };
        mode.value = 'fixed';
        control.value = 'randomize';
        control.afterQueued();
        fixed.getWidgetFromSlot(fixed.inputs[0]);
        assert.equal(control.value, 'fixed');
        unknown.getWidgetFromSlot(unknown.inputs[0]);
        assert.equal(control.value, 'randomize');
        await Promise.resolve();
        assert.equal(control.value, 'randomize');
    });

    test(`${type}: promoted pending controls resume the selected policy on a mode change`, async () => {
        const { node, mode, control, calls, extension, app } = await setup(type);
        const host = promotedHost(node, extension, app);
        mode.value = 'fixed';
        control.afterQueued();
        host.getWidgetFromSlot(host.inputs[0]);
        mode.value = 'populate';
        control.beforeQueued();
        assert.equal(calls.at(-1).value, 'increment');
        assert.equal(control.value, 'increment');
        await Promise.resolve();
        assert.equal(control.value, 'increment');
    });

    test(`${type}: shared hosts use their own promoted modes`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        const fixed = promotedHost(node, extension, app, 'fixed');
        const populate = promotedHost(node, extension, app, 'populate');
        mode.value = 'populate';
        control.beforeQueued();
        fixed.getWidgetFromSlot(fixed.inputs[0]);
        assert.equal(control.value, 'fixed');
        populate.getWidgetFromSlot(populate.inputs[0]);
        assert.equal(control.value, 'increment');
        fixed.getWidgetFromSlot(fixed.inputs[0]);
        assert.equal(control.value, 'fixed');
        await Promise.resolve();
        assert.equal(control.value, 'increment');
    });

    test(`${type}: native promoted projections keep their own identity and frozen values`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        let current = Object.freeze({ name: 'seed', value: 42 });
        const host = promotedHost(node, extension, app, undefined, () => current);
        mode.value = 'fixed';
        control.beforeQueued();
        assert.equal(host.getWidgetFromSlot(host.inputs[0]), current);
        current = Object.freeze({ name: 'seed', value: 99 });
        const target = host.getWidgetFromSlot(host.inputs[0]);
        assert.equal(target, current);
        assert.equal(target.value, 99);
        await Promise.resolve();
        assert.equal(host.getWidgetFromSlot(host.inputs[0]), current);
    });

    test(`${type}: native promoted widget errors propagate unchanged`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        const error = new Error('native widget lookup failed');
        const host = promotedHost(node, extension, app, undefined, () => { throw error; });
        mode.value = 'fixed';
        control.beforeQueued();
        assert.throws(() => host.getWidgetFromSlot(host.inputs[0]), thrown => thrown === error);
        assert.equal(control.value, 'increment');
        await Promise.resolve();
        assert.equal(control.value, 'increment');
    });

    test(`${type}: API drift declines promotion without breaking the queue`, async () => {
        const changes = [
            host => { host.subgraph.inputNode = null; },
            host => { host.subgraph.inputNode.slots = {}; },
            host => { host.subgraph.inputNode.slots[0].linkIds = {}; },
            host => { host.subgraph.getLink = undefined; },
            host => { host.subgraph.getLink = () => null; },
            host => { host.subgraph.getLink = () => ({}); },
            host => { host.subgraph.getLink = () => ({ resolve: () => null }); },
            host => { host.subgraph.getLink = () => ({ resolve() { throw new Error('API drift'); } }); },
            (_, node) => { node.getWidgetFromSlot = undefined; },
            host => { host.inputs = undefined; },
            (_, node) => { node.widgets = undefined; },
            (_, node, seed) => { seed.linkedWidgets = {}; },
        ];
        for (const change of changes) {
            const { node, seed, mode, control, app, extension } = await setup(type);
            const current = { name: 'seed', value: 42 };
            const host = promotedHost(node, extension, app, undefined, () => current);
            const input = host.inputs[0];
            mode.value = 'fixed';
            control.beforeQueued();
            change(host, node, seed);
            assert.equal(host.getWidgetFromSlot(input), current);
            assert.equal(control.value, 'increment');
            await Promise.resolve();
            assert.equal(control.value, 'increment');
        }
    });

    test(`${type}: after-phase restoration redraws the selected policy`, async () => {
        const { node, mode, control, app, extension, draws } = await setup(type);
        const host = promotedHost(node, extension, app);
        mode.value = 'fixed';
        control.value = 'decrement';
        control.afterQueued();
        host.getWidgetFromSlot(host.inputs[0]);
        app.canvas.draw(); // Core draws synchronously before microtasks.
        assert.equal(draws.at(-1), 'fixed');
        await Promise.resolve();
        assert.equal(control.value, 'decrement');
        assert.deepEqual(draws, ['fixed', 'decrement']);
    });

    test(`${type}: nested promoted modes are resolved through the host chain`, async () => {
        const { node, mode, control, app, extension } = await setup(type);
        const inner = promotedHost(node, extension, app, 'populate');
        const outer = promotedHost(inner, extension, app, false);
        mode.value = 'populate';
        control.beforeQueued();
        outer.getWidgetFromSlot(outer.inputs[0]);
        assert.equal(control.value, 'fixed');
        await Promise.resolve();
        assert.equal(control.value, 'increment');
    });

    test(`${type}: serialization errors preserve the policy and propagate unchanged`, async () => {
        const error = new Error('serialization failed');
        const { node, mode, control, app, extension } = await setup(type, true, undefined, error);
        const host = promotedHost(node, extension, app);
        mode.value = 'fixed';
        control.beforeQueued();
        host.getWidgetFromSlot(host.inputs[0]);
        assert.throws(() => app.graphToPrompt(), thrown => thrown === error);
        assert.equal(control.value, 'increment');
        await Promise.resolve();
        assert.equal(control.value, 'increment');
    });
}

test('unrelated node types are untouched', async () => {
    const { extension, app } = await setup('ImpactWildcardProcessor');
    extension.nodeCreated({ comfyClass: 'CheckpointLoaderSimple', get widgets() {
        assert.fail('Accessed unrelated node widgets');
    } }, app);
});
