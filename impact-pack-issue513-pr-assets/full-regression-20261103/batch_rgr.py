from runlib import *
import hashlib
import xml.etree.ElementTree as ET
from urllib.request import urlopen
for _ in range(90):
 try:
  with urlopen('http://127.0.0.1:18199/object_info',timeout=1) as response:
   if 'ImpactWildcardEncode' in json.load(response):break
 except OSError:time.sleep(.5)
else:raise RuntimeError('Edge server not ready')
source=ROOT/'runtime-edge/custom_nodes/comfyui-impact-pack/js/impact-pack.js'
fixed=source.read_bytes()
baseline=subprocess.check_output(['git','show','244e93e:js/impact-pack.js'],cwd=REPO)
testpath=REPO/'tests/test_e2e_wildcard_cache.py';frozen=hashlib.sha256(testpath.read_bytes()).hexdigest()
reports={}
try:
 for phase,data in [('red',baseline),('green',fixed),('red-again',baseline)]:
  source.write_bytes(data)
  label='batch-rgr-'+phase
  record=pytest(label,'tests/test_e2e_wildcard_cache.py','-k','batched_queue',env={'IMPACT_E2E_BASE_URL':'http://127.0.0.1:18199'})
  suite=ET.parse(ROOT/(label+'.xml')).getroot().find('testsuite')
  counts={k:int(suite.attrib[k]) for k in ['tests','failures','errors','skipped']}
  failures=[n.attrib['name'] for n in suite.findall('testcase') if n.find('failure') is not None]
  reports[phase]={'counts':counts,'failures':failures,'source_sha256':hashlib.sha256(data).hexdigest()}
  assert counts=={'tests':4,'failures':0 if phase=='green' else 4,'errors':0,'skipped':0},counts
  assert frozen==hashlib.sha256(testpath.read_bytes()).hexdigest()
finally:
 source.write_bytes(fixed)
 (ROOT/'batch-rgr.json').write_text(json.dumps({'test_sha256':frozen,'reports':reports,'candidate_restored':source.read_bytes()==fixed},indent=2)+'\n')
assert reports['red']==reports['red-again']
