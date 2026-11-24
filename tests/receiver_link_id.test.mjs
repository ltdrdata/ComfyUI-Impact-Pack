import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/impact-pack.js', import.meta.url), 'utf8');

const RECEIVERS = {
  ImageReceiver: { event: 'img-send', widgets: ['image', 'link_id'], detail: { images: [{ filename: 'sent.png', type: 'temp', subfolder: '' }] }, taken: 'sent.png [temp]' },
  LatentReceiver: { event: 'latent-send', widgets: ['latent', 'link_id'], detail: { images: [{ filename: 'sent.latent.png', type: 'temp', subfolder: '' }] }, taken: 'sent.latent.png [temp]' },
  ImpactValueReceiver: { event: 'value-send', widgets: ['typ', 'value', 'link_id'], detail: { value: 37 }, taken: 37 },
};

async function send(type, { widget, linked }) {
  const listeners = {};
  const receiver = { type, size: [200, 100], inputs: [{ name: 'link_id', link: linked === undefined ? null : 9 }],
                     widgets: RECEIVERS[type].widgets.map(name => ({ name, value: name == 'link_id' ? widget : 'before' })) };
  const graph = {
    _nodes: [receiver],
    getLink: id => id == null ? undefined : { origin_id: 4 },
    getNodeById: id => ({ 4: { type: 'ImpactInt', widgets: [{ name: 'value', value: linked }] } })[id],
  };
  receiver.graph = graph;
  const app = { graph, canvas: { setDirty() {} }, getPreviewFormatParam: () => '', registerExtension() {} };
  const context = vm.createContext({ window: {}, console, document: {}, Image: class {} });
  const imports = {
    '../../scripts/app.js': { app, ComfyApp: {} },
    '../../scripts/api.js': { api: {
      addEventListener(name, listener) { listeners[name] = listener; },
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
  listeners[RECEIVERS[type].event]({ detail: { link_id: 5, ...RECEIVERS[type].detail } });
  const held = receiver.widgets[type == 'ImpactValueReceiver' ? 1 : 0].value;
  return held === RECEIVERS[type].taken;
}

for (const type of Object.keys(RECEIVERS)) {
  test(`${type} without a link goes by its own link_id`, async () => {
    assert.equal(await send(type, { widget: 5 }), true);
    assert.equal(await send(type, { widget: 0 }), false);
  });

  test(`${type} with a linked link_id goes by the node behind the link`, async () => {
    assert.equal(await send(type, { widget: 0, linked: 5 }), true);
    assert.equal(await send(type, { widget: 5, linked: 7 }), false);
  });
}
