from pathlib import Path
import json,os,shutil,subprocess,sys
root=Path(__file__).resolve().parent
out=root/'legacy-python-with-data';out.mkdir(exist_ok=True)
source=out/'source';shutil.copytree(root/'legacy-data-source',source,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.hypothesis'))
(source/'impact-pack.ini').write_text(f'[default]\ncustom_wildcards = {source}/custom_wildcards\nwildcard_cache_limit_mb = 50\n')
bootstrap=out/'bootstrap.py'
bootstrap.write_text('''from pathlib import Path
import runpy,sys
source=Path(sys.argv[1]);target=source/sys.argv[2]
sys.path[:0]=[str(source),str(source/'modules'),'/mnt/teratera/git/ComfyUI']
from comfy.cli_args import args
args.cpu=True
sys.argv=[str(target)]
runpy.run_path(str(target),run_name='__main__')
''')
results=[]
for filename in ['tests/wildcards/test_wildcard_lazy_loading.py']:
 label=Path(filename).stem
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.run([sys.executable,str(bootstrap),str(source),filename],cwd=source,env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT)
 results.append({'file':filename,'exit_code':p.returncode})
 (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
 print(label,p.returncode,flush=True)
