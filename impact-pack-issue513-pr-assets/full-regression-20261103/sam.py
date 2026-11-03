from runlib import *
from urllib.request import urlopen
for i in range(90):
 try:
  with urlopen('http://127.0.0.1:18197/object_info',timeout=1) as response:
   if 'SAMLoader' in json.load(response):break
 except OSError:time.sleep(.5)
else:raise RuntimeError('SAM server not ready')
env={'CUDA_VISIBLE_DEVICES':'','IMPACT_E2E_BASE_URL':'http://127.0.0.1:18197','IMPACT_E2E_SAM_IMAGE':str(COMFY/'input/ComfyUI_00156_.png'),'IMPACT_E2E_SAM_MODEL':str(COMFY/'models/sams/sam_vit_b_01ec64.pth'),'IMPACT_E2E_TUTORIAL_IMAGE':str(COMFY/'input/ComfyUI_temp_xltgv_00001_.png')}
pytest('sam-browser-cpu-reference','tests/test_e2e_sam_masks.py','tests/test_e2e_sam_tutorials.py','tests/test_e2e_legacy_sam.py','tests/test_sam_api.py',env=env,timeout=3600)
