from pathlib import Path
import os,subprocess,json,time,sys
root=Path(__file__).resolve().parent
source=root/'cpu-source';env=os.environ|{'IMPACT_COMFY_ROOT':'/mnt/teratera/git/ComfyUI','COMFYUI_ROOT':'/mnt/teratera/git/ComfyUI','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
commands=[
('configured-unit',[sys.executable,'-m','pytest','-c','pytest.ini','-q','--tb=short',f'--junitxml={root}/configured-unit.xml']),
('browser-harness',[sys.executable,'-m','pytest','-c','tests/pytest.ini','tests/test_e2e_ui.py','-q',f'--junitxml={root}/browser-harness.xml']),
('cache-hook',[sys.executable,'tests/integration_wildcard_cache.py','--comfy-root','/mnt/teratera/git/ComfyUI','--report',str(root/'cache-hook.json')]),
('cache-tensors',[sys.executable,'tests/integration_wildcard_cache_tensors.py','--comfy-root','/mnt/teratera/git/ComfyUI','--report',str(root/'cache-tensors.json')]),
('failure-detection',[sys.executable,str(root/'check_failure_detection.py')]),
('sam-mask-unit',[sys.executable,'-m','pytest','-c','tests/pytest.ini','tests/test_sam_masks.py','tests/test_mask_to_segs.py','-q','--tb=short',f'--junitxml={root}/sam-mask-unit.xml']),
('animatediff-unit',[sys.executable,'-m','pytest','-c','tests/pytest.ini','tests/test_animatediff_assembly.py','-q','--tb=short',f'--junitxml={root}/animatediff-unit.xml']),
('mask-blur',[sys.executable,'-m','pytest','-c','tests/pytest.ini','tests/test_mask_blur.py','-q','--tb=short',f'--junitxml={root}/mask-blur.xml']),
('legacy-sam-js',['node','--experimental-vm-modules','--test','tests/legacy_sam.test.mjs']),
('lora-source-js',['node','--test','tests/lora_wildcard_source.test.mjs']),
('lora-tensors',[sys.executable,'tests/integration_lora_wildcard_tensors.py','--comfy-root','/mnt/teratera/git/ComfyUI']),
('lora-validation',[sys.executable,'tests/integration_lora_wildcard_validation.py','--comfy-root','/mnt/teratera/git/ComfyUI']),
('tutorial-tensors',[sys.executable,'tests/integration_wildcard_tutorial_tensors.py','--comfy-root','/mnt/teratera/git/ComfyUI','--tutorials-root','/mnt/teratera/git/ComfyUI-extension-tutorials','--inspire-root','/mnt/teratera/git/ComfyUI/custom_nodes/comfyui-inspire-pack','--report',str(root/'tutorial-tensors.json')])]
results=[]
for label,args in commands:
 start=time.time()
 with (root/(label+'.log')).open('w') as log:r=subprocess.run(args,cwd=source/"tests" if label=="configured-unit" else source,env=env,stdout=log,stderr=subprocess.STDOUT)
 results.append({'label':label,'command':args,'exit_code':r.returncode,'seconds':time.time()-start});(root/'units-result.json').write_text(json.dumps(results,indent=2)+'\n');print(label,r.returncode,flush=True)
