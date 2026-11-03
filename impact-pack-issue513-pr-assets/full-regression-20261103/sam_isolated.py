from runlib import *
from urllib.request import urlopen
import signal
record=json.loads((ROOT/'server-sam.json').read_text());pid=record['pid']
if Path(f'/proc/{pid}/cmdline').exists():
 assert str(ROOT/'runtime-sam') in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  if not Path(f'/proc/{pid}/cmdline').exists():break
  time.sleep(.1)
command=[x for x in record['command'] if not x.startswith('comfyui') or x in ['comfyui-impact-pack','comfyui-impact-subpack']]
with (ROOT/'server-sam-isolated.log').open('w') as log:
 server=subprocess.Popen(command,cwd=COMFY,env=ENV,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert server.poll() is None
   try:
    with urlopen('http://127.0.0.1:18197/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):break
   except OSError:time.sleep(.5)
  else:raise RuntimeError('Server did not start')
  (ROOT/'sam-isolation.json').write_text(json.dumps({'original':record,'isolated_command':command,'pid':server.pid,'reason':'Inspire regional.js imports /scripts/ui.js unconditionally; legacy SAM assertion checks all browser resources, not just Impact imports. Only Impact and Subpack enabled; assertions unchanged.'},indent=2)+'\n')
  pytest('legacy-sam-isolated','tests/test_e2e_legacy_sam.py',env={'IMPACT_E2E_BASE_URL':'http://127.0.0.1:18197','CUDA_VISIBLE_DEVICES':'','IMPACT_E2E_SAM_IMAGE':str(COMFY/'input/ComfyUI_00156_.png'),'IMPACT_E2E_SAM_MODEL':str(COMFY/'models/sams/sam_vit_b_01ec64.pth')})
 finally:
  server.terminate();server.wait(timeout=20)
