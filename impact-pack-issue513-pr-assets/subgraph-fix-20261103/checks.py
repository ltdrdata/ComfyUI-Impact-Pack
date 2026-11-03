from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
import os,json,subprocess,time,sys,hashlib,socket,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent;REPO=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls');PY='/home/rho/venv/bin/python'
PREFIX=os.environ.get('IMPACT_CHECK_PREFIX','')
ENV=os.environ|{'IMPACT_COMFY_ROOT':'/mnt/teratera/git/ComfyUI','COMFYUI_ROOT':'/mnt/teratera/git/ComfyUI','IMPACT_TUTORIALS_DIR':'/mnt/teratera/git/ComfyUI-extension-tutorials','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
def run(label,cmd,cwd=REPO,env=None,timeout=2400):
 label=PREFIX+label
 start=time.monotonic();record={'command':[str(x) for x in cmd],'cwd':str(cwd),'started_utc':datetime.now(timezone.utc).isoformat()}
 print('START',label,flush=True)
 with (ROOT/(label+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,cwd=cwd,env=ENV|(env or {})|{'IMPACT_E2E_ARTIFACTS':str(ROOT/label)},stdout=log,stderr=subprocess.STDOUT,timeout=timeout);record['exit_code']=r.returncode
  except subprocess.TimeoutExpired:record['exit_code']=124
 record['duration_seconds']=round(time.monotonic()-start,2)
 f=ROOT/(label+'.xml')
 if f.exists():
  doc=ET.parse(f);record['counts']={k:doc.find('testsuite').attrib.get(k) for k in ['tests','failures','errors','skipped']}
  record['failures']=[c.attrib['name'] for c in doc.iter('testcase') if c.find('failure') is not None]
 (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results'/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n');print('END',label,record['exit_code'],flush=True)
 return record

def pytest(label,*args,**kwargs):return run(label,[PY,'-m','pytest',*args,'-v','--tb=short','--junitxml='+str(ROOT/(PREFIX+label+'.xml'))],**kwargs)

def start(profile,port,extra):
 with socket.socket() as probe:assert probe.connect_ex(('127.0.0.1',port))!=0,'Test port is occupied'
 profile=PREFIX+profile
 log=(ROOT/('server-'+profile+'.log')).open('w');server=subprocess.Popen([PY,ROOT/'serve.py','--profile',profile,'--port',str(port),*extra],stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   assert server.poll() is None
   try:
    with urlopen(f'http://127.0.0.1:{port}/object_info',timeout=1) as response:
     if 'ImpactWildcardProcessor' in json.load(response):return server,log
   except OSError:time.sleep(.5)
  raise RuntimeError('Server did not start')
 except BaseException:server.terminate();server.wait(timeout=20);log.close();raise

if __name__=='__main__':
 profile=sys.argv[1]
 if profile=='offline':
  run('node-all',['node','--experimental-vm-modules','--test',*sorted((REPO/'tests').glob('*.test.mjs'))])
  pytest('wildcard-unit',cwd=REPO/'tests')
  pytest('other-python','tests/test_mask_blur.py','tests/test_sam_masks.py','tests/test_mask_to_segs.py','tests/test_animatediff_assembly.py','tests/test_e2e_ui.py')
  exit()
 port={'regression':18219,'catalog':18220,'lru':18221,'rgr':18222,'connection':18224,'promoted':18225,'lru-retry':18226,'mode-classic':18227,'mode-lru':18228}[profile]
 server,log=start(profile,port,['--cache','lru'] if profile in ('lru','lru-retry','mode-lru') else ['--cache','none'] if profile=='connection' else [])
 env={'IMPACT_E2E_BASE_URL':f'http://127.0.0.1:{port}'}
 try:
  if profile=='regression':pytest('ordinary-browser','tests/test_e2e_wildcard_cache.py','tests/test_e2e_wildcard_tutorials.py','-k','not promoted_fixed_seed and not inference',env=env)
  elif profile=='catalog':pytest('catalog-browser','tests/test_e2e_widget_catalog.py',env=env)
  elif profile=='connection':pytest('connection-browser','tests/test_e2e_connection_types.py',env=env)
  elif profile in ('mode-classic','mode-lru'):pytest(profile,'tests/test_e2e_wildcard_cache.py','-k','promoted-mode',env=env)
  elif profile=='lru-retry':pytest('promoted-lru-retry','tests/test_e2e_wildcard_cache.py','-k','promoted-mode',env=env)
  elif profile in ('lru','promoted'):pytest('promoted-'+('lru' if profile=='lru' else 'classic'),'tests/test_e2e_wildcard_cache.py','-k','promoted_fixed_seed',env=env)
  else:
   source=ROOT/('runtime-'+PREFIX+'rgr/custom_nodes/comfyui-impact-pack/js/impact-pack.js');candidate=(REPO/'js/impact-pack.js').read_bytes();base=subprocess.check_output(['git','show','HEAD:js/impact-pack.js'],cwd=REPO);records={}
   try:
    for phase,data in [('red',base),('green',candidate),('red-again',base)]:
     source.write_bytes(data);records[phase]=pytest('browser-rgr-'+phase,'tests/test_e2e_wildcard_cache.py','-k','promoted_fixed_seed and positional',env=env)
   finally:source.write_bytes(candidate)
   assert records['green']['exit_code']==0 and len(records['red']['failures'])==4 and records['red']['failures']==records['red-again']['failures'],records
   (ROOT/(PREFIX+'browser-rgr.json')).write_text(json.dumps(records,indent=2)+'\n')
 finally:
  server.terminate();server.wait(timeout=20);log.close()
