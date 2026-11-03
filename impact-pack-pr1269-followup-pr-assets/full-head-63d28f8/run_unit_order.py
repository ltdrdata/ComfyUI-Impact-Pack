from pathlib import Path
import configparser,json,subprocess,sys
r=Path(__file__).resolve().parent;cwd=r/'source/tests';cfg=configparser.ConfigParser();cfg.read(cwd/'pytest.ini')
first=['wildcards/test_lazy_load_verification.py','wildcards/test_progressive_loading.py']
paths=first+[x for x in cfg['pytest']['testpaths'].split() if x not in first]
cmd=[sys.executable,'-m','pytest','-c','pytest.ini',*paths,'-q','--tb=short',f'--junitxml={r}/unit-reordered.xml']
with (r/'unit-reordered.log').open('w') as log:result=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
(r/'unit-reordered-result.json').write_text(json.dumps({'command':cmd,'cwd':str(cwd),'exit_code':result.returncode},indent=2)+'\n')
raise SystemExit(result.returncode)
