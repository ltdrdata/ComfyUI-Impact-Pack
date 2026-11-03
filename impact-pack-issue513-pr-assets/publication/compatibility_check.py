from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time,hashlib,xml.etree.ElementTree as ET
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/teratera/git/impact-pack-issue513-publication-check')
CODE=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
PY='/home/rho/venv/bin/python'
env=os.environ|{'IMPACT_COMFY_ROOT':'/mnt/teratera/git/ComfyUI','IMPACT_E2E_BASE_URL':'http://127.0.0.1:18216','IMPACT_E2E_ARTIFACTS':str(ROOT/'merge-browser'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
record=json.loads((ROOT/'compatibility.json').read_text());results={}
def run(label,cmd,cwd=REPO):
 start=time.monotonic()
 with (ROOT/(label+'.log')).open('w') as log:
  r=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 results[label]={'command':[str(x) for x in cmd],'cwd':str(cwd),'exit_code':r.returncode,'duration_seconds':round(time.monotonic()-start,2)}
 if (ROOT/(label+'.xml')).exists():results[label]['xml_counts']=ET.parse(ROOT/(label+'.xml')).find('testsuite').attrib
 print(label,r.returncode,flush=True)
 return r.returncode
run('merge-unit',[PY,'-m','pytest','-q','--junitxml='+str(ROOT/'merge-unit.xml')],REPO/'tests')
run('merge-node',['node','--experimental-vm-modules','--test',*sorted((REPO/'tests').glob('*.test.mjs'))])
source=(ROOT.parent/'full-regression-20261103/serve.py').read_text().replace("REPO=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')","REPO=Path('/mnt/teratera/git/impact-pack-issue513-publication-check')")
(ROOT/'serve.py').write_text(source)
with (ROOT/'server.log').open('w') as log:
 server=subprocess.Popen([PY,ROOT/'serve.py','--profile','latest-main','--port','18216'],stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert server.poll() is None
   try:
    with urlopen(env['IMPACT_E2E_BASE_URL']+'/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):break
   except OSError:time.sleep(.5)
  else:raise RuntimeError('Server did not start')
  run('merge-browser',[PY,'-m','pytest','tests/test_e2e_wildcard_cache.py','-k','fixed_policy_edit or linked_seed_remains or batched_queue or fixed_native_workflow','-v','--tb=short','--junitxml='+str(ROOT/'merge-browser.xml')])
 finally:
  server.terminate();server.wait(timeout=20)
  record['server_stopped']=True
record.update(results=results,completed_utc=datetime.now(timezone.utc).isoformat(),production_js_unchanged_from_tested_head=hashlib.sha256((REPO/'js/impact-pack.js').read_bytes()).hexdigest()==hashlib.sha256((CODE/'js/impact-pack.js').read_bytes()).hexdigest(),trusted_fixtures_unchanged=all((REPO/'tests/workflows'/n).read_bytes()==(CODE/'tests/workflows'/n).read_bytes() for n in ['all-impact-nodes.json','all-impact-nodes.widgets.json']))
(ROOT/'compatibility.json').write_text(json.dumps(record,indent=2)+'\n')
assert all(r['exit_code']==0 for r in results.values()),results
print('Latest Main compatibility passed.',flush=True)
