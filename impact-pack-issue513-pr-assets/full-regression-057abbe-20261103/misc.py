from runlib import *

with server('misc', 18501, '--cache', 'none') as url:
    env = {'IMPACT_E2E_BASE_URL': url}
    pytest('connection-types', 'tests/test_e2e_connection_types.py', env=env, timeout=2400)
    pytest('switches', 'tests/test_e2e_switches.py', 'tests/test_e2e_switch_loading.py', env=env)
    pytest('mask-browser', 'tests/test_e2e_mask_to_segs.py', 'tests/test_e2e_mask_tutorials.py', env=env)
    run('animatediff-native', [PY, 'tests/e2e_animatediff_tutorials.py', '--base-url', url,
                             '--artifacts', ROOT / 'animatediff-native'], env=env)
    run('wildcard-ui', [PY, 'tests/e2e_wildcards.py', '--base-url', url,
                       '--artifacts', ROOT / 'wildcard-ui'], env=env)
    run('utf8-api', ['bash', 'tests/test_encoding.sh', url], env=env)

with server('sam', 18501, '--cache', 'none', '--minimal') as url:
    env = {'IMPACT_E2E_BASE_URL': url, 'CUDA_VISIBLE_DEVICES': '',
           'IMPACT_E2E_SAM_IMAGE': str(COMFY / 'input/ComfyUI_00156_.png'),
           'IMPACT_E2E_SAM_MODEL': str(COMFY / 'models/sams/sam_vit_b_01ec64.pth'),
           'IMPACT_E2E_TUTORIAL_IMAGE': str(COMFY / 'input/ComfyUI_temp_xltgv_00001_.png')}
    pytest('sam-browser-api', 'tests/test_e2e_sam_masks.py', 'tests/test_e2e_sam_tutorials.py',
           'tests/test_e2e_legacy_sam.py', 'tests/test_sam_api.py', env=env, timeout=3600)

