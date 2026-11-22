import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/impact-sam-editor.js', import.meta.url), 'utf8');

async function loadSam(legacyUI = false) {
  const window = new EventTarget();
  Object.assign(window, { innerWidth: 1600, innerHeight: 1000 });
  const ui = { ComfyDialog: class {}, $el: () => {} };
  if (!legacyUI) window.comfyAPI = { ui };
  const registered = [];
  const context = vm.createContext({
    window, Event, URL,
    Image: class {
      set src(value) {
        this.url = String(value);
        [this.width, this.height] = this.url.includes('portrait') ? [700, 1300] : [1400, 800];
      }
      get src() { return this.url; }
    },
  });
  const ComfyApp = {};
  const toasts = [];
  const app = { registerExtension: extension => registered.push(extension), extensionManager: { toast: { add: toast => toasts.push(toast) } } };
  const imports = {
    '../../scripts/app.js': { app, ComfyApp },
    '../../scripts/api.js': { api: { fetchApi: async () => ({ ok: true }) } },
    '../../scripts/ui.js': ui,
  };
  const requests = [];
  async function load(specifier) {
    requests.push(specifier);
    if (specifier === './image-path.js') {
      const shipped = new vm.SourceTextModule(readFileSync(new URL('../js/image-path.js', import.meta.url), 'utf8'), { context });
      await shipped.link(load);
      await shipped.evaluate();
      return shipped;
    }
    assert.ok(imports[specifier], `Unexpected import: ${specifier}`);
    const values = imports[specifier];
    const module = new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
    await module.link(load);
    await module.evaluate();
    return module;
  }
  const module = new vm.SourceTextModule(`${source}\nexport { ImpactSamEditorDialog, uploadMask };`, {
    context, importModuleDynamically: load,
  });
  await module.link(load);
  await module.evaluate();
  assert.deepEqual(registered.map(extension => extension.name), ['Comfy.Impact.SAMEditor']);
  assert.equal(requests.includes('../../scripts/ui.js'), legacyUI);
  return { module, window, ComfyApp, app, registered, toasts };
}

for (const legacyUI of [false, true]) {
  test(`resize redraws only the current image (${legacyUI ? 'UI module fallback' : 'global UI exports'})`, async () => {
    const { module, window } = await loadSam(legacyUI);
    const draws = [];
    const imageCanvas = {
      width: 0, height: 0, offsetTop: 0, offsetLeft: 0,
      get clientWidth() { return this.width; },
      getContext: () => ({ drawImage: image => draws.push(image.src) }),
    };
    const dialog = Object.create(module.namespace.ImpactSamEditorDialog.prototype);
    Object.assign(dialog, {
      maskCanvas: { style: {} }, maskCtx: {},
      pointsCanvas: { style: {} }, pointsCtx: { canvas: {} },
      prompt_points: [], saveButton: {},
    });
    for (const name of ['landscape', 'portrait']) {
      dialog.setImages(`http://localhost/${name}.png`, imageCanvas, dialog.pointsCanvas);
      dialog.image.onload();
    }
    draws.length = 0;
    window.innerWidth = 1100;
    window.dispatchEvent(new Event('resize'));
    assert.equal(draws.length, 1, 'Resize redrew images from previous editing sessions');
    assert.match(draws[0], /portrait\.png/);
  });
}

test('upload creates missing image metadata for core preview refresh', async () => {
  const { module, ComfyApp } = await loadSam();
  ComfyApp.clipspace = { selectedIndex: 0, imgs: [{ src: 'http://localhost/original.png' }] };
  let refreshes = 0;
  ComfyApp.clipspace_invalidate_handler = () => { refreshes++; };
  const saved = { filename: 'saved.png', subfolder: '', type: 'temp' };
  await module.namespace.uploadMask(saved, {});
  assert.deepEqual(ComfyApp.clipspace.images?.[0], saved,
    'Saved image metadata is required when the core refreshes node previews');
  assert.equal(refreshes, 1);
});

test('Open in SAM Detector explains a node that shows no image and opens one that does', async () => {
  const { module, ComfyApp, app, registered, toasts } = await loadSam();
  const copied = [];
  ComfyApp.copyToClipspace = node => copied.push(node);
  let shown = 0;
  module.namespace.ImpactSamEditorDialog.getInstance = () => ({ show: () => { shown++; } });
  class NodeType { getExtraMenuOptions() {} }
  await registered[0].beforeRegisterNodeDef(NodeType, { output: ['IMAGE', 'MASK'] }, app);
  const open = node => {
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.equal(options[0].content, 'Open in SAM Detector');
    options[0].callback();
  };

  for (const imgs of [undefined, [{ src: '' }], [{ src: 'http://localhost/view?filename=gone.png', complete: true, naturalWidth: 0 }]]) {
    open(Object.assign(new NodeType(), { imgs }));
  }
  assert.equal(toasts.length, 3);
  assert.match(toasts[0].detail, /after the node itself has been executed/);
  assert.deepEqual([copied.length, shown], [0, 0]);

  const node = Object.assign(new NodeType(), { imgs: [{ src: 'http://localhost/view?filename=a.png', complete: true, naturalWidth: 256 }] });
  open(node);
  assert.equal(toasts.length, 3);
  assert.deepEqual([copied, shown, ComfyApp.clipspace_return_node], [[node], 1, node]);
});
