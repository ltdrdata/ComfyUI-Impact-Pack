from runlib import *
server_env={'IMPACT_E2E_BASE_URL':'http://127.0.0.1:18196','IMPACT_E2E_REFERENCE_IMAGE':'/mnt/teratera/git/impact-pack-pr1190-followup-pr-assets/tutorial-audit/baseline-inference/output.png'}
pytest('catalog','tests/test_e2e_widget_catalog.py',env=server_env)
pytest('wildcard-browser','tests/test_e2e_wildcard_cache.py','tests/test_e2e_wildcard_tutorials.py',env=server_env)
pytest('switch-browser','tests/test_e2e_switches.py','tests/test_e2e_switch_loading.py',env=server_env)
pytest('mask-browser','tests/test_e2e_mask_to_segs.py','tests/test_e2e_mask_tutorials.py',env=server_env)
run('animatediff-native',[PY,'tests/e2e_animatediff_tutorials.py','--base-url',server_env['IMPACT_E2E_BASE_URL'],'--artifacts',ROOT/'animatediff-native'],env=server_env)
run('wildcard-ui',[PY,'tests/e2e_wildcards.py','--base-url',server_env['IMPACT_E2E_BASE_URL'],'--artifacts',ROOT/'wildcard-ui'],env=server_env)
run('utf8-api',['bash','tests/test_encoding.sh',server_env['IMPACT_E2E_BASE_URL']],env=server_env)
