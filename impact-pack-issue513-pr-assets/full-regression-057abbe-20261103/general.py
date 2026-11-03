from runlib import *

with server('general', 18500) as url:
    env = {'IMPACT_E2E_BASE_URL': url}
    pytest('catalog', 'tests/test_e2e_widget_catalog.py', env=env)
    pytest('wildcard-classic', 'tests/test_e2e_wildcard_cache.py', 'tests/test_e2e_wildcard_tutorials.py',
           '-k', 'not inference', env=env, timeout=3600)

