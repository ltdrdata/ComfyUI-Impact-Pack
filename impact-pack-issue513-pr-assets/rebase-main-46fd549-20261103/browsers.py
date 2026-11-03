from runlib import *
import sys

new_cases = 'mapped_list or scalar_runtime or constant_text_link or runtime_population_source or execution_metadata'
profile = sys.argv[1]
if profile == 'classic':
    with server('cache-classic', 18510) as url:
        pytest('cache-and-tutorials', 'tests/test_e2e_wildcard_cache.py', 'tests/test_e2e_wildcard_tutorials.py',
               '-k', 'not (' + new_cases + ' or inference)', env={'IMPACT_E2E_BASE_URL': url}, timeout=3600)
elif profile == 'lru':
    with server('cache-lru', 18511, '--cache', 'lru') as url:
        pytest('cache-lru', 'tests/test_e2e_wildcard_cache.py', env={'IMPACT_E2E_BASE_URL': url}, timeout=3600)
elif profile == 'upstream':
    with server('upstream', 18512) as url:
        env = {'IMPACT_E2E_BASE_URL': url}
        pytest('upstream-interactions', 'tests/test_e2e_wildcard_cache.py', 'tests/test_e2e_wildcard_controls.py',
               '-k', new_cases + ' or wildcard_controls', env=env, timeout=3600)
        pytest('catalog', 'tests/test_e2e_widget_catalog.py', env=env)
