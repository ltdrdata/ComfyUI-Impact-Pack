from runlib import *

run('javascript', ['/home/rho/.nvm/versions/node/v24.12.0/bin/node', '--experimental-vm-modules', '--test-reporter=tap', '--test',
                  *map(str, sorted((REPO / 'tests').glob('*.test.mjs')))])
pytest('wildcard-unit', cwd=REPO / 'tests')
pytest('mask-blur', 'tests/test_mask_blur.py')
pytest('sam-and-mask', 'tests/test_sam_masks.py', 'tests/test_mask_to_segs.py')
pytest('animatediff-assembly', 'tests/test_animatediff_assembly.py')
pytest('browser-harness', 'tests/test_e2e_ui.py', 'tests/test_wildcard_cache_artifacts.py')
run('input-validation', [PY, 'tests/integration_lora_wildcard_validation.py', '--comfy-root', COMFY])
run('wildcard-cache-integration', [PY, 'tests/integration_wildcard_cache.py', '--comfy-root', COMFY,
                                   '--report', ROOT / 'wildcard-cache-integration-report.json'])
