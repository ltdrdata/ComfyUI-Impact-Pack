import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/switch-none.js', import.meta.url), 'utf8');
const titles = { 1: 'Load Checkpoint', 3: 'Branch', 4: 'Which checkpoint', 5: 'Switch (Any)', 6: 'Switch (Any)', 8: 'Negative' };

function prompt(select, input1 = ['1', 1]) {
  return {
    1: { class_type: 'CheckpointLoaderSimple', inputs: {} },
    3: { class_type: 'ImpactConditionalBranchSelMode', inputs: {} },
    4: { class_type: 'ImpactInt', inputs: { value: 3 } },
    5: { class_type: 'ImpactSwitch', inputs: { select, input1, input2: ['1', 1] } },
    6: { class_type: 'ImpactSwitch', inputs: { select: ['4', 0], input1: ['1', 1] } },
    8: { class_type: 'CLIPTextEncode', inputs: { clip: ['5', 0], text: 'blurry' } },
    9: { class_type: 'VAEDecode', inputs: { vae: ['1', 2] } },
  };
}

function element() {
  return {
    style: {}, children: [], listeners: {}, textContent: '',
    append(...added) { for (const child of added) { child.parent = this; this.children.push(child); } },
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); },
    addEventListener(name, listener) { this.listeners[name] = listener; },
    getBoundingClientRect: () => ({ left: 0, top: 0, bottom: 0 }),
  };
}

async function load(output) {
  const listeners = {};
  const area = element();
  const graph = { getNodeById: id => ({ id: String(id), title: titles[id], graph }) };
  const app = {
    rootGraph: graph,
    canvas: { graph, canvas: { parentElement: area } },
    graphToPrompt: async () => ({ output }),
    registerExtension: extension => extension.setup(),
  };
  const head = element();
  let nextFrame;
  const context = vm.createContext({ console, requestAnimationFrame(callback) { nextFrame = callback; }, document: { createElement: element, head, querySelector: element } });
  const imports = { '../../scripts/app.js': { app }, '../../scripts/api.js': { api: { addEventListener(name, listener) { listeners[name] = listener; } } } };
  const module = new vm.SourceTextModule(source, { context });
  await module.link(specifier => new vm.SyntheticModule(Object.keys(imports[specifier]), function () {
    for (const [key, value] of Object.entries(imports[specifier])) this.setExport(key, value);
  }, { context }));
  await module.evaluate();
  const send = (name, detail) => listeners[name]({ detail });
  const fail = (node_id, current_inputs) => send('execution_error', { node_id, node_type: output[node_id].class_type, current_inputs });
  const bubbles = () => area.children[0].children;
  const marks = () => Array.from(head.children[1].textContent.matchAll(/data-node-id="(\w+)"/g), (match, index) => [match[1], bubbles()[index].textContent]);
  const close = index => bubbles()[index].children[0].listeners.click();
  const frame = () => nextFrame();
  return { send, fail, marks, close, frame };
}

const NO_LINK = [['5', 'returned None: select is 3, input3 has no link']];

test('the select of a switch names an input without a link: the switch is marked and named', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('executing', '5');
  page.send('impact-switch-none', { node_id: '5', select: 3 });
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), NO_LINK);
});

test('the next prompt takes the None from the cache, without the switch running: the mark is the same', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('executing', '5');
  page.send('impact-switch-none', { node_id: '5', select: 3 });
  page.send('executing', '8');
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), NO_LINK);
});

test('the selected input of a switch gave None: the switch and the node linked to that input are marked', async () => {
  const page = await load(prompt(['4', 0], ['3', 0]));
  page.send('impact-switch-none', { node_id: '5', select: 1 });
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), [['5', 'passed on the None of input1'], ['3', 'gave None to the Switch (Any)']]);
});

test('the node linked to the selected input is another switch: it is marked as the giver, and the way is not followed further', async () => {
  const page = await load(prompt(['4', 0], ['6', 0]));
  page.send('impact-switch-none', { node_id: '6', select: 3 });
  page.send('impact-switch-none', { node_id: '5', select: 1 });
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), [['5', 'passed on the None of input1'], ['6', 'gave None to the Switch (Any)']]);
});

test('what a switch selected is not known: no mark', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('impact-switch-none', { node_id: '5', select: 3 });
  page.send('executing', '5');
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), []);
});

test('after a reload of the page, a select that is a widget value still tells', async () => {
  const page = await load(prompt(3));
  await page.fail('8', { clip: [null], text: ['blurry'] });
  assert.deepEqual(page.marks(), NO_LINK);
});

test('an error that is not about a None of a switch gets no mark', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('impact-switch-none', { node_id: '5', select: 3 });
  await page.fail('9', { vae: ['<VAE>'], samples: ['<latent>'] });
  await page.fail('9', { vae: [null] });
  assert.deepEqual(page.marks(), []);
});

test('the marks go when the next prompt starts', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('impact-switch-none', { node_id: '5', select: 3 });
  await page.fail('8', { clip: [null], text: ['blurry'] });
  page.send('execution_start', {});
  assert.deepEqual(page.marks(), []);
});

test('the x of a bubble removes that mark and leaves the other', async () => {
  const page = await load(prompt(['4', 0], ['3', 0]));
  page.send('impact-switch-none', { node_id: '5', select: 1 });
  await page.fail('8', { clip: [null], text: ['blurry'] });
  page.close(0);
  assert.deepEqual(page.marks(), [['3', 'gave None to the Switch (Any)']]);
});

const UNPAIRED = 'returned mask1 unchanged: mask1 is a batch of 2 and mask2 a batch of 3';

test('the server warns of a node: the node is marked with the text', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('execution_start', {});
  page.send('executing', '9');
  page.send('impact-node-warning', { node_id: '9', text: UNPAIRED });
  assert.deepEqual(page.marks(), [['9', UNPAIRED]]);
});

test('the next prompt takes the result of that node from the cache: the mark is the same', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('executing', '9');
  page.send('impact-node-warning', { node_id: '9', text: UNPAIRED });
  page.send('execution_start', {});
  assert.deepEqual(page.marks(), []);
  page.send('execution_cached', { nodes: ['1', '9'] });
  page.frame();
  assert.deepEqual(page.marks(), [['9', UNPAIRED]]);
});

test('the node runs again and the server does not warn: no mark, and none from the cache after it', async () => {
  const page = await load(prompt(['4', 0]));
  page.send('impact-node-warning', { node_id: '9', text: UNPAIRED });
  page.send('execution_start', {});
  page.send('executing', '9');
  page.send('execution_start', {});
  page.send('execution_cached', { nodes: ['9'] });
  page.frame();
  assert.deepEqual(page.marks(), []);
});
