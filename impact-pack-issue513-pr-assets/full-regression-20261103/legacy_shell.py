from runlib import *
import io,re,shlex,socket,tarfile
area=ROOT/'legacy-isolated';area.mkdir(exist_ok=True)
runtime=area/'runtime';pack=runtime/'custom_nodes/comfyui-impact-pack';pack.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=REPO))) as archive:archive.extractall(pack,filter='data')
pidfile=area/'server.pid'
stop=area/'stop.py'
stop.write_text('''from pathlib import Path
import os,signal,time
p=Path(__file__).parent/'server.pid'
if p.exists():
 pid=int(p.read_text());cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and cmd.read_bytes():
  assert str(Path(__file__).parent/'runtime').encode() in cmd.read_bytes()
  os.kill(pid,signal.SIGTERM)
  for _ in range(50):
   if not cmd.exists() or not cmd.read_bytes():break
   time.sleep(.1)
 p.unlink(missing_ok=True)
''')
launch=area/'launch.py'
launch.write_text('''from pathlib import Path
import os,sys
root=Path(__file__).parent
(root/'server.pid').write_text(str(os.getpid()))
command=['/home/rho/venv/bin/python','/mnt/teratera/git/ComfyUI/main.py','--cpu','--disable-auto-launch','--base-directory',str(root/'runtime'),'--disable-all-custom-nodes','--whitelist-custom-nodes','comfyui-impact-pack',*sys.argv[1:]]
os.chdir('/mnt/teratera/git/ComfyUI');os.execv(command[0],command)
''')
(area/'run.sh').write_text('exec '+shlex.quote(PY)+' '+shlex.quote(str(launch))+' "$@"\n')
port=18210
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
adaptations={}
for name in ['test_error_handling.sh','test_edge_cases.sh','test_deep_nesting.sh','test_ondemand_loading.sh','test_config_quotes.sh']:
 original=(REPO/'tests'/name).read_text()
 script=re.sub(r'^PORT=\d+',f'PORT={port}',original,flags=re.M)
 script=script.replace('COMFYUI_DIR="/mnt/teratera/git/ComfyUI"','COMFYUI_DIR="'+str(area)+'"')
 script=re.sub(r'pkill -f "python\.\*main.py\.\*--port \$PORT"',shlex.quote(PY)+' '+shlex.quote(str(stop)),script)
 script=script.replace('/tmp/',str(area)+'/')
 assert 'pkill ' not in script and 'COMFYUI_DIR="/mnt/teratera/git/ComfyUI"' not in script
 target=pack/'tests'/name;target.write_text(script)
 adaptations[name]='Disposable package copy; isolated port/config/log/temp paths; run.sh pins this package; termination targets only the PID owned by this runner. Assertions unchanged.'
 try:run('legacy-shell-'+name.removeprefix('test_').removesuffix('.sh'),['bash',target],cwd=pack,timeout=600)
 finally:subprocess.run([PY,stop],check=True)
# The statistical script only uses HTTP; point it at the already isolated general server.
name='test_dynamic_prompts_full.sh';script=(REPO/'tests'/name).read_text()
script=re.sub(r'^PORT=\d+','PORT=18196',script,flags=re.M).replace('/tmp/',str(area)+'/')
target=pack/'tests'/name;target.write_text(script)
run('legacy-shell-dynamic-prompts',['bash',target],cwd=pack)
adaptations[name]='Only HTTP port and log path changed; statistical assertions unchanged.'
(ROOT/'legacy-adaptations.json').write_text(json.dumps(adaptations,indent=2)+'\n')
