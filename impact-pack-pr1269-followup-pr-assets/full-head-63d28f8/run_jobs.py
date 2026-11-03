from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent;results=[]
for name in sys.argv[1:]:
 result=subprocess.run([sys.executable,str(root/'run_server_suite.py'),str(root/(name+'.json'))])
 results.append({'job':name,'exit_code':result.returncode});(root/('pipeline-'+sys.argv[1]+'.json')).write_text(json.dumps(results,indent=2)+'\n')

raise SystemExit(any(row["exit_code"] for row in results))
