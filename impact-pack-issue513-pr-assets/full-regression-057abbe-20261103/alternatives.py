from runlib import *
from urllib.request import urlopen

with server('lru', 18502, '--cache', 'lru') as url:
    pytest('wildcard-lru', 'tests/test_e2e_wildcard_cache.py', env={'IMPACT_E2E_BASE_URL': url}, timeout=3600)
with server('dropdown', 18502, '--samples', 'lora_samples') as url:
    pytest('lora-source-loading', 'tests/test_e2e_lora_wildcard_source.py', env={'IMPACT_E2E_BASE_URL': url})
with server('ondemand', 18502, '--wildcard-limit', '0.000001') as url:
    with urlopen(url + '/impact/wildcards/list/loaded') as response:
        assert json.load(response)['on_demand_mode']
    run('wildcard-ondemand-ui', [PY, 'tests/e2e_wildcards.py', '--base-url', url,
                               '--artifacts', ROOT / 'wildcard-ondemand-ui'], env={'IMPACT_E2E_BASE_URL': url})
