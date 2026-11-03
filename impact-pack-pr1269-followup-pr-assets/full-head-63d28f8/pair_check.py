from pathlib import Path
import json,subprocess,sys,time
import requests
r=Path(__file__).resolve().parent;marker=r/'api-complete.json'
if sys.argv[1]=='full':
 deadline=time.monotonic()+180
 while not marker.exists():
  assert time.monotonic()<deadline,'API comparison did not finish'
  time.sleep(.5)
 raise SystemExit(0)
urls={name:'http://127.0.0.1:'+str(json.loads((r/(('pair-ondemand' if name=='ondemand' else name)+'.json')).read_text())['port']) for name in ['full','ondemand']}
code=1
try:
 deadline=time.monotonic()+60
 while True:
  try:
   response=requests.get(urls['full']+'/system_stats',timeout=1);response.raise_for_status()
   assert str(r/'full/runtime') in response.json()['system']['argv'];break
  except requests.ConnectionError:
   assert time.monotonic()<deadline;time.sleep(.5)
 states={name:requests.get(url+'/impact/wildcards/list/loaded',timeout=5).json() for name,url in urls.items()}
 assert states['full']['on_demand_mode'] is False and states['ondemand']['on_demand_mode'] is True
 cmd=[sys.executable,str(r/'source/tests/wildcards/test_wildcard_final.py'),'--full-url',urls['full'],'--on-demand-url',urls['ondemand']]
 result=subprocess.run(cmd);code=result.returncode
 (r/'api-result.json').write_text(json.dumps({'command':cmd,'exit_code':code,'initial_states':states},indent=2)+'\n')
finally:marker.write_text(json.dumps({'exit_code':code})+'\n')
raise SystemExit(code)
