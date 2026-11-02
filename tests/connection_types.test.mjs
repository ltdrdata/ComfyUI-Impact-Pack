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
  await extensions[0].beforeRegisterNodeDef(Node, { name }, app);
  const twoInputs = name === 'ImpactCompare' || name.startsWith('ImpactConditionalBranch');
  const node = Object.assign(new Node(), {
    inputs: Array.from({ length: twoInputs ? 2 : 1 }, () => ({ type: '*' })),
    outputs: [{ type: name === 'ImpactCompare' ? 'BOOLEAN' : '*' }],
  });
  const typedSlots = [node.inputs[0]];
  if (twoInputs) typedSlots.push(node.inputs[1]);
  if (name !== 'ImpactCompare') typedSlots.push(node.outputs[0]);
  return { node, app, typedSlots };
}

for (const name of nodeTypes) {
  for (const missing of ['node', 'output']) {
    test(`${name}: restore the link type while the source ${missing} is unavailable`, async () => {
      const { node, app, typedSlots } = await loadNode(name);
      app.graph.getNodeById = () => missing === 'node' ? undefined : { outputs: [{ type: 'STRING' }] };
      node.onConnectionsChange(1, 0, true, { origin_id: 1, origin_slot: 1, type: 'IMAGE' });
      assert.deepEqual(typedSlots.map(slot => slot.type), typedSlots.map(() => 'IMAGE'));
      if (name === 'ImpactCompare') assert.equal(node.outputs[0].type, 'BOOLEAN');
    });
  }

  test(`${name}: an unknown type stays wildcard and can be resolved later`, async () => {
    const { node, app, typedSlots } = await loadNode(name);
    const link = { origin_id: 1, origin_slot: 1 };
    node.onConnectionsChange(1, 0, true, link);
    assert.ok(typedSlots.every(slot => slot.type === '*'));
    app.graph.getNodeById = () => ({ outputs: [{ type: 'STRING' }, { type: 'IMAGE' }] });
    node.onConnectionsChange(1, 0, true, link);
    assert.ok(typedSlots.every(slot => slot.type === 'IMAGE'));
  });

  test(`${name}: an available source type takes precedence over saved link metadata`, async () => {
    const { node, app, typedSlots } = await loadNode(name);
    app.graph.getNodeById = () => ({ outputs: [{ type: 'IMAGE' }] });
    node.onConnectionsChange(1, 0, true, { origin_id: 1, origin_slot: 0, type: 'STRING' });
    assert.ok(typedSlots.every(slot => slot.type === 'IMAGE'));
  });
}
