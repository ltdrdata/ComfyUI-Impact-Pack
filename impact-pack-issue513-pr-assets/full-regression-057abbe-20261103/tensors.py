from runlib import *
run('wildcard-cache-tensors', [PY,'tests/integration_wildcard_cache_tensors.py','--comfy-root',COMFY,'--report',ROOT/'wildcard-cache-tensors-report.json'])
run('wildcard-tutorial-tensors', [PY,'tests/integration_wildcard_tutorial_tensors.py','--comfy-root',COMFY,'--tutorials-root',ENV['IMPACT_TUTORIALS_DIR'],'--inspire-root',COMFY/'custom_nodes/comfyui-inspire-pack','--report',ROOT/'wildcard-tutorial-tensors-report.json'])
run('lora-source-tensors', [PY,'tests/integration_lora_wildcard_tensors.py','--comfy-root',COMFY])
