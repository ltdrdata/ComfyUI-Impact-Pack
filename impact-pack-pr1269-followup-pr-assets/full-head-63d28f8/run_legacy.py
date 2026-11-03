from pathlib import Path
import difflib,io,json,os,re,shlex,shutil,socket,subprocess,sys,time
from urllib.request import urlopen

root=Path(__file__).resolve().parent
base=root/'legacy-with-data';base.mkdir(exist_ok=True)
stop=base/'stop_owned.py'
stop.write_text('''from pathlib import Path
import psutil,sys,time
pidfile=Path(sys.argv[1]);base=sys.argv[2]
if pidfile.exists():
 try:
  p=psutil.Process(int(pidfile.read_text()))
  if base in p.cmdline():
   p.terminate()
   for _ in range(200):
    if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:break
    time.sleep(.1)
   else:p.kill()
 except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
''')
files=['tests/test_config_quotes.sh','tests/test_deep_nesting.sh','tests/test_edge_cases.sh','tests/test_error_handling.sh','tests/test_ondemand_loading.sh','tests/test_dynamic_prompts_full.sh',
 'tests/wildcards/test_progressive_ondemand.sh','tests/wildcards/test_sequential_loading.sh','tests/wildcards/test_wildcard_consistency.sh','tests/wildcards/test_versatile_prompts.sh','tests/wildcards/test_lazy_load_api.sh']
results=[]
for filename in files:
 name=Path(filename).stem;case=base/name;case.mkdir(exist_ok=True)
 source=case/'source';shutil.copytree(root/'legacy-data-source',source,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.hypothesis'))
 runtime=case/'runtime';(runtime/'custom_nodes').mkdir(parents=True);(runtime/'custom_nodes/comfyui-impact-pack').symlink_to(source)
 shim=case/'comfy';shim.mkdir();(case/'tmp').mkdir()
 pidfile=case/'server.pid'
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 (source/'impact-pack.ini').write_text(f'[default]\ncustom_wildcards = {source}/custom_wildcards\nwildcard_cache_limit_mb = 50\n')
 args=[sys.executable,'/mnt/teratera/git/ComfyUI/main.py','--cpu','--cache-classic','--base-directory',str(runtime),'--models-directory','/mnt/teratera/git/ComfyUI/models','--disable-all-custom-nodes','--whitelist-custom-nodes','comfyui-impact-pack','--disable-auto-launch']
 (shim/'run.sh').write_text('#!/bin/bash\nprintf "%s" "$$" > '+shlex.quote(str(pidfile))+'\nexec '+shlex.join(args)+' "$@"\n')
 original=(source/filename).read_text();modified=original.replace('/tmp/',str(case/'tmp')+'/')
 modified=re.sub(r'^COMFYUI_DIR=.*$',f'COMFYUI_DIR="{shim}"',modified,flags=re.M)
 modified=re.sub(r'^(IMPACT_DIR|IMPACT_PACK_DIR)=.*$',lambda m:f'{m.group(1)}="{source}"',modified,flags=re.M)
 modified=re.sub(r'\b81[89]\d\b',str(port),modified)
 safe_stop=shlex.join([sys.executable,str(stop),str(pidfile),str(runtime)])
 modified=re.sub(r'pkill[^\n]*',safe_stop+' 2>/dev/null || true',modified)
 target=source/filename;target.write_text(modified)
 (case/'environment-adaptation.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='original/'+filename,tofile='isolated/'+filename)))
 started=time.time();managed=None
 try:
  if name=='test_dynamic_prompts_full':
   log=(case/'startup.log').open('w')
   managed=subprocess.Popen(['bash','run.sh','--listen','127.0.0.1','--port',str(port)],cwd=shim,stdout=log,stderr=subprocess.STDOUT)
   for _ in range(120):
    assert managed.poll() is None
    try:
     with urlopen(f'http://127.0.0.1:{port}/system_stats',timeout=1) as response:
      assert str(runtime) in json.load(response)['system']['argv'];break
    except OSError:time.sleep(.5)
   else:raise TimeoutError('Legacy server startup')
  with (case/'test.log').open('w') as log:
   result=subprocess.run(['bash',str(target)],cwd=source,env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT,timeout=600)
  record={'file':filename,'exit_code':result.returncode,'seconds':time.time()-started,'port':port}
 except Exception as error:
  record={'file':filename,'runner_error':repr(error),'seconds':time.time()-started,'port':port}
 finally:
  subprocess.run([sys.executable,str(stop),str(pidfile),str(runtime)],check=True)
  if managed is not None:managed.wait()
 results.append(record);(base/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(record,flush=True)
