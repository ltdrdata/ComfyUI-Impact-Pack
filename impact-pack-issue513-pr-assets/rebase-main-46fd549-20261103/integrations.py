from runlib import *

run('input-validation', [PY, 'tests/integration_lora_wildcard_validation.py', '--comfy-root', COMFY])
run('cache-integration', [PY, 'tests/integration_wildcard_cache.py', '--comfy-root', COMFY,
                        '--report', ROOT / 'cache-integration-report.json'])
run('cache-tensors', [PY, 'tests/integration_wildcard_cache_tensors.py', '--comfy-root', COMFY,
                    '--report', ROOT / 'cache-tensors-report.json'])
run('tutorial-tensors', [PY, 'tests/integration_wildcard_tutorial_tensors.py', '--comfy-root', COMFY,
    '--tutorials-root', ENV['IMPACT_TUTORIALS_DIR'], '--inspire-root', COMFY / 'custom_nodes/comfyui-inspire-pack',
    '--report', ROOT / 'tutorial-tensors-report.json'])
run('lora-tensors', [PY, 'tests/integration_lora_wildcard_tensors.py', '--comfy-root', COMFY])
run('sampling-capture', [PY, ROOT / 'sampling_capture.py'], timeout=1800)
run('sampling-overlays', [PY, ROOT / 'sampling/overlays/render.py'])
