import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/impact-pack.js', import.meta.url), 'utf8');
const nodeTypes = [
  'ImpactControlBridge', 'ImpactConditionalBranch', 'ImpactConditionalBranchSelMode',
  'ImpactCompare', 'ImpactSelectNthItemOfAnyList',
];

async function loadNode(name) {
  const extensions = [];
  const app = {
    graph: { getNodeById: () => undefined },
    registerExtension: extension => extensions.push(extension),
  };
  const context = vm.createContext({ window: {}, console });
  const imports = {
    '../../scripts/app.js': { app, ComfyApp: {} },
    '../../scripts/api.js': { api: {
      addEventListener() {},
      fetchApi: async () => ({ json: async () => ({ data: [] }) }),
    } },
    './common.js': { customAlert() {}, isBeforeFrontendVersion: () => false },
  };
  const module = new vm.SourceTextModule(source, { context });
  await module.link(specifier => {
    const values = imports[specifier];
    assert.ok(values, `Unexpected import: ${specifier}`);
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await module.evaluate();
  class Node {}
  const extension = extensions.find(extension => extension.name === 'Comfy.Impack');
  assert.ok(extension, 'Impact Pack extension was not registered');
  await extension.beforeRegisterNodeDef(Node, { name }, app);
  const twoInputs = name === 'ImpactCompare' || name.startsWith('ImpactConditionalBranch');
  const inputCount = twoInputs ? 2 : 1;
  const node = Object.assign(new Node(), {
    inputs: Array.from({ length: inputCount }, () => ({ type: '*' })),
    outputs: [{ type: name === 'ImpactCompare' ? 'BOOLEAN' : '*' }],
  });
  function expectTypes(type) {
    assert.deepEqual(node.inputs.map(slot => slot.type), Array(inputCount).fill(type));
    assert.deepEqual(node.outputs.map(slot => slot.type), [name === 'ImpactCompare' ? 'BOOLEAN' : type]);
  }
  return { node, app, expectTypes };
}

for (const name of nodeTypes) {
  for (const missing of ['node', 'output']) {
    test(`${name}: restore the link type while the source ${missing} is unavailable`, async () => {
      const { node, app, expectTypes } = await loadNode(name);
      app.graph.getNodeById = () => missing === 'node' ? undefined : { outputs: [{ type: 'STRING' }] };
      node.onConnectionsChange(1, 0, true, { origin_id: 1, origin_slot: 1, type: 'IMAGE' });
      expectTypes('IMAGE');
    });
  }

  test(`${name}: an unknown type stays wildcard and can be resolved later`, async () => {
    const { node, app, expectTypes } = await loadNode(name);
    const link = { origin_id: 1, origin_slot: 1 };
    node.onConnectionsChange(1, 0, true, link);
    expectTypes('*');
    app.graph.getNodeById = () => ({ outputs: [{ type: 'STRING' }, { type: 'IMAGE' }] });
    node.onConnectionsChange(1, 0, true, link);
    expectTypes('IMAGE');
  });

  test(`${name}: an available source type takes precedence over saved link metadata`, async () => {
    const { node, app, expectTypes } = await loadNode(name);
    app.graph.getNodeById = () => ({ outputs: [{ type: 'IMAGE' }] });
    node.onConnectionsChange(1, 0, true, { origin_id: 1, origin_slot: 0, type: 'STRING' });
    expectTypes('IMAGE');
  });
}
