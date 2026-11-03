from runlib import *
import hashlib,re,shlex,shutil,socket
area=ROOT/'legacy-isolated';pack=area/'runtime/custom_nodes/comfyui-impact-pack';stop=area/'stop.py'
corpus=COMFY/'custom_nodes/comfyui-impact-pack/custom_wildcards'
shutil.copytree(corpus,pack/'custom_wildcards',dirs_exist_ok=True)
(ROOT/'legacy-corpus.json').write_text(json.dumps({'source':str(corpus),'sha256':{str(p.relative_to(corpus)):hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus.rglob('*') if p.is_file()}},indent=2)+'\n')
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18210))!=0
adaptations={}
for name in ['test_lazy_load_api.sh','test_progressive_ondemand.sh','test_sequential_loading.sh','test_versatile_prompts.sh','test_wildcard_consistency.sh']:
 s=(REPO/'tests/wildcards'/name).read_text()
 s=re.sub(r'^(IMPACT_PACK_DIR|IMPACT_DIR)=.*$',lambda m:m[1]+'='+shlex.quote(str(pack)),s,flags=re.M)
 s=re.sub(r'^COMFYUI_DIR=.*$','COMFYUI_DIR='+shlex.quote(str(area)),s,flags=re.M)
 s=re.sub(r'pkill(?: -9)? -f "[^"]+"',shlex.quote(PY)+' '+shlex.quote(str(stop)),s)
 s=re.sub(r'\b819[0-9]\b','18210',s)
 s=s.replace('/tmp/',str(area)+'/')
 assert 'pkill ' not in s
 p=pack/'tests/wildcards'/name;p.write_text(s)
 try:run('legacy-extra-'+name.removeprefix('test_').removesuffix('.sh'),['bash',p],cwd=pack,timeout=600)
 finally:subprocess.run([PY,stop],check=True)
 adaptations[name]='Copied local wildcard corpus; corrected old parent-directory detection to disposable package; replaced ports/logs and broad pkill with owned PID cleanup. Assertions unchanged.'
(pack/'impact-pack.ini').write_text('[default]\ncustom_wildcards = '+str(pack/'custom_wildcards')+'\nwildcard_cache_limit_mb = 50\n')
pytest('legacy-lazy-verification-with-corpus','tests/wildcards/test_lazy_load_verification.py',cwd=pack)
name='test_wildcard_final.py';s=(REPO/'tests/wildcards'/name).read_text()
s=re.sub(r'^IMPACT_PACK_DIR = .*$', 'IMPACT_PACK_DIR = Path('+repr(str(pack))+')',s,flags=re.M)
s=re.sub(r'^COMFYUI_DIR = .*$', 'COMFYUI_DIR = Path('+repr(str(area))+')',s,flags=re.M)
s=s.replace('8191','18210')
p=pack/'tests/wildcards'/name;p.write_text(s)
try:run('legacy-final-cli',[PY,p],cwd=pack,timeout=300)
finally:subprocess.run([PY,stop],check=True)
adaptations[name]='Corrected parent-directory detection to disposable package and server shim; isolated port only. Test cases unchanged.'
(ROOT/'legacy-extended-adaptations.json').write_text(json.dumps(adaptations,indent=2)+'\n')
