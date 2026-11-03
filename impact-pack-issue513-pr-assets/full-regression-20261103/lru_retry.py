from runlib import *
from urllib.request import urlopen
with (ROOT/'server-lru-retry.log').open('w') as log:
 server=subprocess.Popen([PY,ROOT/'serve.py','--profile','lru','--cache','lru','--port','18199'],stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert server.poll() is None
   try:
    with urlopen('http://127.0.0.1:18199/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):break
   except OSError:time.sleep(.5)
  else:raise RuntimeError('Server did not start')
  for i in range(1,4):
   pytest('wildcard-lru-retry-'+str(i),'tests/test_e2e_wildcard_cache.py::test_fixed_mode_suppresses_automatic_seed_changes[choice-decrement-before-ImpactWildcardEncode]',env={'IMPACT_E2E_BASE_URL':'http://127.0.0.1:18199'})
 finally:
  server.terminate();server.wait(timeout=20)
