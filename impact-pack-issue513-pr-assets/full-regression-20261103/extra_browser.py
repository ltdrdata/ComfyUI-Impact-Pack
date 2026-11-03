from runlib import *
from urllib.request import urlopen
url='http://127.0.0.1:18199'
for profile,args in [('dropdown',['--samples','lora_samples']),('lru',['--cache','lru']),('ondemand',['--wildcard-limit','0.000001'])]:
 with (ROOT/('server-'+profile+'.log')).open('w') as log:
  server=subprocess.Popen([PY,ROOT/'serve.py','--profile',profile,'--port','18199',*args],stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(100):
    assert server.poll() is None,'Server exited'
    try:
     with urlopen(url+'/object_info',timeout=1) as response:
      if 'ImpactWildcardProcessor' in json.load(response):break
    except OSError:time.sleep(.5)
   else:raise RuntimeError('Server did not start')
   env={'IMPACT_E2E_BASE_URL':url}
   if profile=='dropdown':pytest('lora-source-browser','tests/test_e2e_lora_wildcard_source.py',env=env)
   elif profile=='lru':pytest('wildcard-lru-browser','tests/test_e2e_wildcard_cache.py','-k','fixed_mode_suppresses or fixed_policy_edit or linked_seed_remains or batched_queue or populate_reuses',env=env)
   else:
    with urlopen(url+'/impact/wildcards/list/loaded') as response:assert json.load(response)['on_demand_mode']
    run('wildcard-ondemand-ui',[PY,'tests/e2e_wildcards.py','--base-url',url,'--artifacts',ROOT/'wildcard-ondemand-ui'],env=env)
  finally:
   server.terminate()
   try:server.wait(timeout=20)
   except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
