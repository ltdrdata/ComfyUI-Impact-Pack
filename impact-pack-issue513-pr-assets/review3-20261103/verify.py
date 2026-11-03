from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
import os,json,subprocess,time,sys,hashlib,socket,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;R=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls');PY='/home/rho/venv/bin/python'
ENV=os.environ|{'IMPACT_COMFY_ROOT':'/mnt/teratera/git/ComfyUI','COMFYUI_ROOT':'/mnt/teratera/git/ComfyUI','IMPACT_TUTORIALS_DIR':'/mnt/teratera/git/ComfyUI-extension-tutorials','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
def run(label,args,env=None,cwd=R):
 started=time.monotonic();record={'command':[str(x) for x in args],'cwd':str(cwd),'started_utc':datetime.now(timezone.utc).isoformat()}
 print('START',label,flush=True)
 with (P/(label+'.log')).open('w') as log:r=subprocess.run(args,cwd=cwd,env=ENV|(env or {})|{'IMPACT_E2E_ARTIFACTS':str(P/label)},stdout=log,stderr=subprocess.STDOUT,timeout=2400)
 record|={'exit_code':r.returncode,'seconds':round(time.monotonic()-started,2)}
 xml=P/(label+'.xml')
 if xml.exists():
  doc=ET.parse(xml);record['counts']={k:doc.find('testsuite').get(k) for k in ('tests','failures','errors','skipped')};record['failures']=[c.get('name') for c in doc.iter('testcase') if c.find('failure') is not None]
 (P/'results').mkdir(exist_ok=True);(P/'results'/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n');print('END',label,r.returncode,flush=True);return record

def pytest(label,*args,env=None,cwd=R):return run(label,[PY,'-m','pytest',*args,'-v','--tb=short','--maxfail=1','--junitxml='+str(P/(label+'.xml'))],env,cwd)
def start(profile,port):
 with socket.socket() as probe:assert probe.connect_ex(('127.0.0.1',port))!=0,'Occupied test port'
 log=(P/('server-'+profile+'.log')).open('w');args=[PY,P/'serve.py','--profile',profile,'--port',str(port)]
 if profile in ('lru','mode-lru'):args+=['--cache','lru']
 proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert proc.poll() is None
   try:
    with urlopen(f'http://127.0.0.1:{port}/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):return proc,log
   except OSError:time.sleep(.5)
  raise RuntimeError('Server unavailable')
 except BaseException:proc.terminate();proc.wait(timeout=20);log.close();raise
if __name__=='__main__':
 profile=sys.argv[1]
 if profile=='offline':
  run('node-all',['/home/rho/.nvm/versions/node/v24.12.0/bin/node','--experimental-vm-modules','--test-reporter=tap','--test',*sorted((R/'tests').glob('*.test.mjs'))]);pytest('wildcard-unit',cwd=R/'tests');pytest('other-python','tests/test_mask_blur.py','tests/test_sam_masks.py','tests/test_mask_to_segs.py','tests/test_animatediff_assembly.py','tests/test_e2e_ui.py','tests/test_wildcard_cache_artifacts.py');exit()
 port={'classic':18410,'lru':18411,'ordinary':18412,'catalog':18413,'rgr':18415,'mode':18417,'mode-rgr':18418,'mode-lru':18419}[profile];proc,log=start(profile,port);env={'IMPACT_E2E_BASE_URL':f'http://127.0.0.1:{port}'}
 try:
  if profile in ('classic','lru'):pytest('promoted-'+profile,'tests/test_e2e_wildcard_cache.py','-k','promoted_fixed_seed',env=env)
  elif profile=='ordinary':pytest('ordinary-browser','tests/test_e2e_wildcard_cache.py','tests/test_e2e_wildcard_tutorials.py','-k','not promoted_fixed_seed and not inference',env=env)
  elif profile in ('mode','mode-lru'):pytest('mode-browser' if profile=='mode' else 'mode-lru-browser','tests/test_e2e_wildcard_cache.py','-k','promoted_fixed_seed and promoted-mode',env=env)
  elif profile=='catalog':pytest('catalog-browser','tests/test_e2e_widget_catalog.py',env=env)
  else:
   target=P/('runtime-'+profile)/'custom_nodes/comfyui-impact-pack/js/impact-pack.js';candidate=(R/'js/impact-pack.js').read_bytes();reports={}
   if profile=='mode-rgr':
    text=candidate.decode();a=text.index('function promoted_wildcard_mode(');b=text.index('\nif(is_legacy_front())',a);baseline=(text[:a]+"function promoted_wildcard_mode(node) { return wildcard_seed_modes.get(node)?.value; }\n"+text[b:]).encode();selection='promoted_fixed_seed and promoted-mode and after';label='browser-mode-rgr'
   else:baseline=subprocess.check_output(['git','show','b1ee82d:js/impact-pack.js'],cwd=R);selection='promoted_fixed_seed and positional';label='browser-rgr'
   try:
    for phase,data in [('red',baseline),('green',candidate),('red-again',baseline)]:
     target.write_bytes(data);reports[phase]=pytest(label+'-'+phase,'tests/test_e2e_wildcard_cache.py','-k',selection,env=env)
   finally:target.write_bytes(candidate)
   assert len(reports['red']['failures'])==(2 if profile=='mode-rgr' else 4) and reports['red']['failures']==reports['red-again']['failures'] and reports['green']['exit_code']==0
   (P/(label+'.json')).write_text(json.dumps({'candidate_js_sha256':hashlib.sha256(candidate).hexdigest(),'baseline':'Ignore promoted modes' if profile=='mode-rgr' else 'b1ee82d','results':reports},indent=2)+'\n')
 finally:proc.terminate();proc.wait(timeout=20);log.close()
