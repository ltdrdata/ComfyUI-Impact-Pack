import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function setup() {
    const pending = [];
    let extension;
    const app = { registerExtension(value) { extension = value; }, graph: { setDirtyCanvas() {} } };
    const fetch = url => new Promise(resolve => pending.push({ url, resolve }));
    const source = fs.readFileSync(new URL('../js/impact-wildcard.js', import.meta.url), 'utf8')
        .replace(/^import .*;\s*$/gm, '');
    vm.runInNewContext(source, { app, api: { fetchApi: fetch }, fetch, console: { log() {}, error() {} } });
    const category = { name: 'category', value: 'first', callback() {} };
    const wildcard = { name: 'lora_wildcard', value: 'A', options: { values: ['A'] } };
    const node = { comfyClass: 'LoraWildcardSource', widgets: [category, wildcard], setDirtyCanvas() {} };
    extension.nodeCreated(node, app);
    async function reply(index, items) {
        pending[index].resolve({ ok: true, json: async () => items });
        await new Promise(setImmediate);
    }
    return { pending, category, wildcard, node, reply, extension, app };
}

test('late creation response cannot replace a restored selection', async () => {
    const { category, wildcard, node, reply } = setup();
    category.value = 'saved';
    wildcard.value = 'Saved second';
    node.onConfigure({});
    await reply(1, ['Saved first', 'Saved second']);
    await reply(0, ['A']);
    assert.equal(category.value, 'saved');
    assert.equal(wildcard.value, 'Saved second');
    assert.deepEqual(Array.from(wildcard.options.values), ['Saved first', 'Saved second']);
});

test('only the latest category response can update the dropdown', async () => {
    const { category, wildcard, reply } = setup();
    await reply(0, ['A']);
    category.value = 'second';
    category.callback('second');
    category.value = 'third';
    category.callback('third');
    await reply(2, ['C']);
    await reply(1, ['B']);
    assert.equal(wildcard.value, 'C');
    assert.deepEqual(Array.from(wildcard.options.values), ['C']);
});

test('returning to a category rejects its older response after a new selection', async () => {
    const { category, wildcard, reply } = setup();
    await reply(0, ['A']);
    for (const value of ['saved', 'other', 'saved']) {
        category.value = value;
        category.callback(value);
    }
    await reply(3, ['Saved first', 'Saved second']);
    wildcard.value = 'Saved second';
    await reply(1, ['Saved first', 'Saved second']);
    await reply(2, ['Other']);
    assert.equal(wildcard.value, 'Saved second');
    assert.deepEqual(Array.from(wildcard.options.values), ['Saved first', 'Saved second']);
});

test('missing saved choices survive restoration for backend validation', async () => {
    const { category, wildcard, node, reply } = setup();
    await reply(0, ['A']);
    category.value = 'saved';
    wildcard.value = 'Removed LoRA';
    node.onConfigure({});
    await reply(1, ['Different LoRA']);
    assert.equal(wildcard.value, 'Removed LoRA');
});

test('category query values are encoded', async () => {
    const { category, pending, reply } = setup();
    await reply(0, ['A']);
    category.value = 'ink & color#1';
    category.callback(category.value);
    assert.equal(new URL(pending[1].url, 'http://localhost').searchParams.get('category'), category.value);
});

test('existing nodes are not changed or queried', () => {
    const { extension, app, pending } = setup();
    const baseline = JSON.parse(fs.readFileSync(new URL('./workflows/all-impact-nodes.widgets.json', import.meta.url)));
    for (const entry of baseline.nodes.filter(n => n.type !== 'LoraWildcardSource')) {
        const node = { comfyClass: entry.type };
        Object.defineProperty(node, 'widgets', { get() { assert.fail(`Accessed ${entry.type}`); } });
        extension.nodeCreated(node, app);
    }
    assert.equal(pending.length, 1);
});
