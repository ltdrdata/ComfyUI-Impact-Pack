from pathlib import Path
import argparse,hashlib,io,json,os,shutil,socket,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/teratera/git/impact-pack-issue513-publication-check')
COMFY=Path('/mnt/teratera/git/ComfyUI')
p=argparse.ArgumentParser();p.add_argument('--profile',default='general');p.add_argument('--cache',default='classic',choices=['classic','none','lru']);p.add_argument('--gpu',action='store_true');p.add_argument('--fp32',action='store_true');p.add_argument('--samples',default='samples');p.add_argument('--wildcard-limit',type=float,default=50);p.add_argument('--port',type=int,default=18196);a=p.parse_args()
runtime=ROOT/('runtime-'+a.profile);custom=runtime/'custom_nodes';custom.mkdir(parents=True,exist_ok=True)
pack=custom/'comfyui-impact-pack'
if not pack.exists():
 pack.mkdir()
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=REPO))) as tar:tar.extractall(pack,filter='data')
# Freeze the current PR source without copying ignored local configuration.
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).split(b'\0')
for raw in tracked:
 if raw:
  rel=raw.decode();source=REPO/rel
  if source.is_file():
   target=pack/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
(pack/'impact-pack.ini').write_text('[default]\ncustom_wildcards = '+str(pack/'tests/wildcards'/a.samples)+'\nwildcard_cache_limit_mb = '+str(a.wildcard_limit)+'\nsam_editor_model = sam_vit_b_01ec64.pth\nsam_editor_cpu = True\n')
plugins=['comfyui-inspire-pack','comfyui-impact-subpack','comfyui_essentials','comfyui_controlnet_aux','comfyui-animatediff-evolved','comfyui-videohelpersuite','comfyui_ipadapter_plus']
for name in plugins:
 if not (custom/name).exists():(custom/name).symlink_to(COMFY/'custom_nodes'/name,target_is_directory=True)
if not (runtime/'user').exists():shutil.copytree(ROOT.parent/'fixed-mode/runtime/user',runtime/'user')
(runtime/'input').mkdir(exist_ok=True)
for name in ['ComfyUI_00156_.png','ComfyUI_temp_xltgv_00001_.png','ComfyUI_00289_.png','ComfyUI_temp_fyzuo_00038_.png','example.png','chunli.png']:
 if not (runtime/'input'/name).exists():shutil.copy2(COMFY/'input'/name,runtime/'input'/name)
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',a.port))!=0,'Port already occupied'
command=[sys.executable,'main.py','--listen','127.0.0.1','--port',str(a.port),'--disable-auto-launch','--base-directory',str(runtime),'--models-directory',str(COMFY/'models'),'--disable-all-custom-nodes','--whitelist-custom-nodes','comfyui-impact-pack',*plugins]
command+=['--cuda-device','0','--deterministic','--disable-xformers','--use-pytorch-cross-attention'] if a.gpu else ['--cpu']
if a.fp32:command+=['--force-fp32','--fp32-vae','--disable-dynamic-vram']
command+=['--cache-'+a.cache]
if a.cache=='lru':command+=['10']
env=os.environ|{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0','HF_HUB_OFFLINE':'1'}
(ROOT/('server-'+a.profile+'.json')).write_text(json.dumps({'pid':os.getpid(),'command':command,'runtime':str(runtime),'source_sha256':hashlib.sha256((pack/'js/impact-pack.js').read_bytes()).hexdigest()},indent=2)+'\n')
os.chdir(COMFY);os.execvpe(command[0],command,env)
