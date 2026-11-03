from pathlib import Path
import json,os,socket,subprocess,sys,time
from urllib.request import urlopen

root=Path(__file__).resolve().parent
job=json.loads(Path(sys.argv[1]).read_text())
name=job['name'];out=root/name;out.mkdir(exist_ok=True)
port=job['port'];source=Path(job['source'])
# Catalog is launched early in this fresh run; the later pipeline visit must not execute it twice.
if job.get('skip_completed_same_run') and (out/'result.json').exists():
 prior=json.loads((out/'result.json').read_text())
 assert prior.get('server_stopped') and prior['source']==str(source)
 assert len(prior['commands'])==len(job['commands']) and all(c['exit_code']==0 for c in prior['commands'])
 print(name+': already completed in this fresh run',flush=True)
 raise SystemExit(0)
env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IMPACT_E2E_BASE_URL':f'http://127.0.0.1:{port}',
 'IMPACT_TUTORIALS_DIR':'/mnt/teratera/git/ComfyUI-extension-tutorials',
 'IMPACT_E2E_SAM_IMAGE':'/mnt/teratera/git/ComfyUI/input/ComfyUI_00156_.png',
 'IMPACT_E2E_SAM_MODEL':'/mnt/teratera/git/ComfyUI/models/sams/sam_vit_b_01ec64.pth',
 'IMPACT_E2E_TUTORIAL_IMAGE':'/mnt/teratera/git/ComfyUI/input/ComfyUI_temp_xltgv_00001_.png'}|job.get('env',{})
command=[sys.executable,'main.py','--listen','127.0.0.1','--port',str(port),'--base-directory',str(out/'runtime'),
 '--models-directory','/mnt/teratera/git/ComfyUI/models','--disable-all-custom-nodes','--whitelist-custom-nodes',
 'comfyui-impact-pack',*job.get('plugins',[]),'--disable-auto-launch',*job['server_options']]
result={'server_command':command,'source':str(source),'commands':[]}
with socket.socket() as check:
 assert check.connect_ex(('127.0.0.1',port)) != 0, f'Port {port} already in use'
with (out/'server.log').open('w') as log:
 server=subprocess.Popen(command,cwd='/mnt/teratera/git/ComfyUI',env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(240):
   assert server.poll() is None,'Server stopped during startup'
   try:
    with urlopen(env['IMPACT_E2E_BASE_URL']+'/object_info',timeout=1) as response:
     if 'ImpactWildcardEncode' in json.load(response):
      with urlopen(env['IMPACT_E2E_BASE_URL']+'/system_stats',timeout=2) as stats_response:
       stats=json.load(stats_response)
      assert str(out/'runtime') in stats['system']['argv'], 'Connected server is not this isolated runtime'
      assert server.poll() is None, 'Own server stopped during readiness check'
      result['verified_system_stats']=stats
      break
   except OSError:pass
   time.sleep(.5)
  else:raise TimeoutError('Server startup')
  print(name+': server ready',flush=True)
  for case in job['commands']:
   label=case['label'];case_env=env|{'IMPACT_E2E_ARTIFACTS':str(out/label)}|case.get('env',{})
   args=[part.replace('{out}',str(out)).replace('{url}',env['IMPACT_E2E_BASE_URL']) for part in case['args']]
   started=time.time()
   with (out/(label+'.log')).open('w') as testlog:
    completed=subprocess.run(args,cwd=source,env=case_env,stdout=testlog,stderr=subprocess.STDOUT)
   result['commands'].append({'label':label,'command':args,'exit_code':completed.returncode,'seconds':time.time()-started})
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
   print(f'{name}/{label}: exit {completed.returncode}',flush=True)
 finally:
  server.terminate()
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  result['server_stopped']=True
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(any(case['exit_code'] for case in result['commands']))
