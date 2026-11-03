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
  const imports = {
    '../../scripts/app.js': { app: { registerExtension: extension => registered.push(extension) }, ComfyApp },
    '../../scripts/api.js': { api: { fetchApi: async () => ({ ok: true }) } },
    '../../scripts/ui.js': ui,
  };
  const requests = [];
  async function load(specifier) {
    requests.push(specifier);
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
  return { module, window, ComfyApp };
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
