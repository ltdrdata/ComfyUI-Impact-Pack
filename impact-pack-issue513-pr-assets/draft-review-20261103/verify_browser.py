from pathlib import Path
import hashlib,json,os,shutil,socket,subprocess,sys,time
from urllib.request import urlopen
import xml.etree.ElementTree as ET

ROOT=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
COMFY=Path('/mnt/teratera/git/ComfyUI')
OUT=Path(__file__).resolve().parent
runtime=OUT/'runtime'
custom=runtime/'custom_nodes'
custom.mkdir(parents=True,exist_ok=True)
for name,target in [('comfyui-impact-pack',ROOT),('comfyui-inspire-pack',COMFY/'custom_nodes/comfyui-inspire-pack')]:
    (custom/name).symlink_to(target,target_is_directory=True)
shutil.copytree(OUT.parent/'fixed-mode/runtime/user',runtime/'user')
PORT=18195
URL=f'http://127.0.0.1:{PORT}'
with socket.socket() as s: assert s.connect_ex(('127.0.0.1',PORT))!=0,'Port in use'
env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONHASHSEED':'0',
 'IMPACT_E2E_BASE_URL':URL,'IMPACT_E2E_ARTIFACTS':str(OUT/'browser'),
 'IMPACT_TUTORIALS_DIR':'/mnt/teratera/git/ComfyUI-extension-tutorials'}
for key in ['IMPACT_E2E_CHECKPOINT','IMPACT_E2E_LORA','IMPACT_TUTORIAL_SD15_CHECKPOINT','IMPACT_TUTORIAL_SDXL_CHECKPOINT']:
    env.pop(key,None)
server_cmd=[sys.executable,'main.py','--cpu','--listen','127.0.0.1','--port',str(PORT),'--disable-auto-launch',
 '--cache-classic','--base-directory',str(runtime),'--models-directory',str(COMFY/'models'),
 '--disable-all-custom-nodes','--whitelist-custom-nodes','comfyui-impact-pack','comfyui-inspire-pack']
command=[sys.executable,'-m','pytest','tests/test_e2e_wildcard_cache.py','tests/test_e2e_wildcard_tutorials.py',
 '-k','not inference','-v','--tb=short',f'--junitxml={OUT}/browser.xml']
report={'server_command':server_cmd,'test_command':command,'cwd':str(ROOT),
 'source_sha256':hashlib.sha256((ROOT/'js/impact-pack.js').read_bytes()).hexdigest(),
 'test_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['tests/test_e2e_wildcard_cache.py','tests/test_e2e_wildcard_tutorials.py']}}
with (OUT/'server.log').open('w') as log:
    server=subprocess.Popen(server_cmd,cwd=COMFY,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            assert server.poll() is None,'Server stopped during startup'
            try:
                with urlopen(URL+'/object_info',timeout=1) as response:
                    if 'ImpactWildcardEncode' in json.load(response): break
            except OSError: pass
            time.sleep(.5)
        else: raise TimeoutError('Server startup')
        print('Isolated server ready; running cache matrix and native restoration.',flush=True)
        with (OUT/'browser.log').open('w') as logfile:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,timeout=1500)
        report['exit_code']=p.returncode
        suite=ET.parse(OUT/'browser.xml').getroot().find('testsuite')
        report['counts']={k:int(suite.attrib[k]) for k in ['tests','failures','errors','skipped']}
        print(report['counts'],flush=True)
    finally:
        server.terminate()
        try: server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill();server.wait(timeout=10)
        report['server_stopped']=True
        (OUT/'browser-review.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['exit_code']==0
assert report['counts']=={'tests':50,'failures':0,'errors':0,'skipped':0}
