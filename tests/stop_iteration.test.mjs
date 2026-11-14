import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/impact-pack.js', import.meta.url), 'utf8');

// Loads the shipped extension and returns the frontend state its stop-iteration listener works on.
async function load(mode, autoQueueEnabled = false) {
  const listeners = {};
  const checkbox = { checked: autoQueueEnabled };
  const app = {
    extensionManager: { queueSettings: { mode } },
    ui: { autoQueueEnabled },
    registerExtension() {},
  };
  const document = { getElementById: id => id === 'autoQueueCheckbox' ? checkbox : null };
  const context = vm.createContext({ window: {}, console, document });
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
  assert.equal(typeof listeners['stop-iteration'], 'function');
  return { queue: app.extensionManager.queueSettings, ui: app.ui, checkbox, stop: listeners['stop-iteration'] };
}

test('a running Run (Instant) stops repeating and stays selected', async () => {
  const { queue, stop } = await load('instant-running');
  stop();
  assert.equal(queue.mode, 'instant-idle');
});

test('Run (On Change) is turned off, as it has no idle state', async () => {
  const { queue, stop } = await load('change');
  stop();
  assert.equal(queue.mode, 'disabled');
});

for (const mode of ['disabled', 'instant-idle']) {
  test(`queue mode '${mode}' is left as selected`, async () => {
    const { queue, stop } = await load(mode);
    stop();
    assert.equal(queue.mode, mode);
  });
}

test('legacy menu: Auto Queue is turned off, not only unchecked', async () => {
  const { queue, ui, checkbox, stop } = await load('disabled', true);
  stop();
  assert.deepEqual({ enabled: ui.autoQueueEnabled, checked: checkbox.checked, mode: queue.mode },
                   { enabled: false, checked: false, mode: 'disabled' });
});
