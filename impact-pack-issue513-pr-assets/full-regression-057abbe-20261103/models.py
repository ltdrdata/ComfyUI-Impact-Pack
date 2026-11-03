from runlib import *
from argparse import Namespace
import hashlib
import sys
from PIL import Image

sys.path.insert(0, str(REPO / 'tests'))
from e2e_ui import browser_session
from test_e2e_wildcard_cache import control_timing
from wildcard_tutorial_cases import tutorial

# Keep real tensor checks and GPU inference sequential on the installed GPU.
run('wildcard-cache-tensors', [PY, 'tests/integration_wildcard_cache_tensors.py', '--comfy-root', COMFY,
                             '--report', ROOT / 'wildcard-cache-tensors-report.json'])
run('wildcard-tutorial-tensors', [PY, 'tests/integration_wildcard_tutorial_tensors.py', '--comfy-root', COMFY,
    '--tutorials-root', ENV['IMPACT_TUTORIALS_DIR'], '--inspire-root', COMFY / 'custom_nodes/comfyui-inspire-pack',
    '--report', ROOT / 'wildcard-tutorial-tensors-report.json'])
run('lora-source-tensors', [PY, 'tests/integration_lora_wildcard_tensors.py', '--comfy-root', COMFY])

with server('baseline-models', 18503, '--gpu', '--fp32', '--baseline', '--samples', 'model_samples') as url:
    folder = ROOT / 'main-lbw-reference'
    args = Namespace(base_url=url, browser_channel='chrome', headed=False, timeout=900, artifacts=folder)
    with browser_session(args) as ui:
        ui.new_workflow()
        control_timing(ui, 'after')
        path, original, target = tutorial(ENV['IMPACT_TUTORIALS_DIR'], 'lbw')
        ui.load_workflow(path, len(original['nodes']))
        entry = ui.run()
        (folder / 'history.json').write_text(json.dumps(entry, indent=2) + '\n')
        image = ui.images(entry)['9'][0]
        image.save(folder / 'output.png')
        ui.page.screenshot(path=str(folder / 'workflow.png'))
        (folder / 'reference.json').write_text(json.dumps({
            'revision': META['base'], 'frontend': '1.57.6', 'device': 'deterministic CUDA FP32',
            'workflow_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'seed': target['widgets_values'][5], 'prompt_id': entry['prompt'][1],
            'source_js_sha256': hashlib.sha256((ROOT / 'baseline-source/js/impact-pack.js').read_bytes()).hexdigest(),
        }, indent=2) + '\n')

with server('models', 18503, '--gpu', '--fp32', '--samples', 'model_samples') as url:
    with browser_session(Namespace(base_url=url, browser_channel='chrome', headed=False, timeout=900,
                                   artifacts=ROOT / 'model-settings')) as ui:
        control_timing(ui, 'after')
    env = {'IMPACT_E2E_BASE_URL': url, 'IMPACT_E2E_REFERENCE_IMAGE': str(ROOT / 'main-lbw-reference/output.png')}
    run('lora-source-inference', [PY, 'tests/e2e_lora_wildcard_inference.py', '--base-url', url,
        '--timeout', '900', '--artifacts', ROOT / 'lora-source-inference'], env=env, timeout=3000)
    pytest('lbw-tutorial-inference', 'tests/test_e2e_wildcard_tutorials.py', '-k', 'inference', env=env)
    for label, name in [('detailer-inference', 'e2e_dd_compat.py'),
                        ('conditioning-inference', 'e2e_conditionings.py'),
                        ('switch-tutorial-inference', 'e2e_tutorials.py')]:
        run(label, [PY, 'tests/' + name, '--base-url', url, '--timeout', '900',
                    '--artifacts', ROOT / label], env=env, timeout=3000)
    tutorials = Path(ENV['IMPACT_TUTORIALS_DIR'])
    original = tutorials / 'ComfyUI-Impact-Pack/workflow/regional-sampler-img2img.png'
    companion = tutorials / 'ComfyUI-Inspire-Pack/workflow/regional-conditioning-img2img.png'
    with Image.open(original) as image:
        workflow = json.loads(image.info['workflow'])
    with Image.open(companion) as image:
        masks = {node['id']: node for node in json.loads(image.info['workflow'])['nodes']}
    nodes = {node['id']: node for node in workflow['nodes']}
    for node_id, slot in [(35, 3), (49, 1)]:
        assert nodes[node_id]['type'] == masks[node_id]['type']
        value = masks[node_id]['widgets_values'][slot]
        assert value.startswith('data:image/')
        nodes[node_id]['widgets_values'][slot] = value
    restored = ROOT / 'regional-restored.json'
    restored.write_text(json.dumps(workflow))
    (ROOT / 'regional-media-provenance.json').write_text(json.dumps({
        'original': str(original), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'companion': str(companion), 'companion_sha256': hashlib.sha256(companion.read_bytes()).hexdigest(),
        'restoration': 'Only missing embedded mask pixels at existing widget positions 35:3 and 49:1, from the original companion PNG. Original tutorial files remain unchanged.',
        'derived_workflow_sha256': hashlib.sha256(restored.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    run('regional-inference', [PY, 'tests/e2e_regional.py', '--base-url', url, '--timeout', '900',
        '--artifacts', ROOT / 'regional-inference'], env=env | {'IMPACT_REGIONAL_WORKFLOW': str(restored)}, timeout=3000)
