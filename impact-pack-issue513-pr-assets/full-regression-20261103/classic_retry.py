from runlib import *
from urllib.request import urlopen
with (ROOT/'server-classic-retry.log').open('w') as log:
 server=subprocess.Popen([PY,ROOT/'serve.py','--profile','classic-retry','--cache','classic','--port','18199'],stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert server.poll() is None
   try:
    with urlopen('http://127.0.0.1:18199/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):break
   except OSError:time.sleep(.5)
  else:raise RuntimeError('Server did not start')
  for i in range(1,4):
   pytest('wildcard-classic-retry-'+str(i),'tests/test_e2e_wildcard_cache.py::test_linked_inputs_and_subgraph_native_queue[ImpactWildcardEncode-named]',env={'IMPACT_E2E_BASE_URL':'http://127.0.0.1:18199'})
 finally:
  server.terminate();server.wait(timeout=20)
