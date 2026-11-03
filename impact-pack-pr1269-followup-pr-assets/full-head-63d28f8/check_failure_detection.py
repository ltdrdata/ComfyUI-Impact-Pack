"""Fault injection for the reviewed test guards; no production source mutation."""
import ast,contextlib,importlib.util,json,subprocess,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import pytest
r=Path(__file__).resolve().parent;script=r/'source/tests/wildcards/test_wildcard_final.py'
scenario='';observed=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def respond(self,status,body):
  self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(body).encode())
 def do_GET(self):self.respond(200,{'on_demand_mode':False if scenario=='wrong-mode' else self.server.on_demand,'data':[]})
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']));self.server.posts+=1
  status=503 if scenario=='http-error' else 200
  text={'empty-text':'','error-text':'ERROR: connection failed','unexpanded':'__colors__'}.get(scenario,'blue' if scenario=='unequal-results' and self.server.on_demand else 'red')
  self.respond(status,{} if scenario=='missing-text' else {'text':text})
servers=[]
try:
 for mode in [False,True]:
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.on_demand=mode;server.posts=0;threading.Thread(target=server.serve_forever,daemon=True).start();servers.append(server)
 cmd=[sys.executable,str(script),'--full-url',f'http://127.0.0.1:{servers[0].server_port}','--on-demand-url',f'http://127.0.0.1:{servers[1].server_port}']
 for scenario in ['http-error','missing-text','empty-text','error-text','unexpanded','wrong-mode','unequal-results','valid-responses']:
  for s in servers:s.posts=0
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
  expected=scenario=='valid-responses';assert (result.returncode==0)==expected,(scenario,result.stdout,result.stderr)
  if expected:assert [s.posts for s in servers]==[7,7]
  observed.append({'scenario':scenario,'exit_code':result.returncode,'posts':[s.posts for s in servers],'expected_success':expected,'last_line':(result.stderr or result.stdout).strip().splitlines()[-1]})
finally:
 for s in servers:s.shutdown();s.server_close()
spec=importlib.util.spec_from_file_location('api_diagnostic',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for name,value in [('no-results',{}),('incomplete-results',{'Simple wildcard':'red'})]:
 with patch.object(module,'run_test',return_value=value),patch.object(sys,'argv',cmd[1:]):
  try:module.main()
  except AssertionError:observed.append({'scenario':name,'rejected':True})
  else:raise AssertionError(name+' was accepted')
p=r/'source/tests/test_e2e_wildcard_controls.py';tree=ast.parse(p.read_text());test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_control_bridge_mode_change_requeues')
outer=next(n for n in ast.walk(test) if isinstance(n,ast.If) and any(isinstance(x,ast.Assert) for x in n.orelse) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=='behavior == \'Mute\'')
code=compile(ast.fix_missing_locations(ast.Module(body=outer.orelse,type_ignores=[])),str(p),'exec');guards=[]
for node,bypass,broken in [('ImpactWildcardProcessor',False,True),('ImpactWildcardEncode',False,True),('ImpactWildcardProcessor',True,True),('ImpactWildcardEncode',False,False)]:
 scope={'pytest':pytest,'node_type':node,'bypass_input':bypass,'preview_id':'p','entries':[None,{'prompt':[None,None,{'p':{'inputs':{} if broken else {'source':['s',0]}}}],'outputs':{} if broken else {'p':{'text':['{red|green|blue}']}}}]}
 try:exec(code,scope);outcome='passed'
 except pytest.xfail.Exception:outcome='xfail'
 except (KeyError,AssertionError):outcome='failed'
 expected='xfail' if node=='ImpactWildcardProcessor' and not bypass and broken else 'failed' if broken else 'passed'
 assert outcome==expected;guards.append({'node':node,'first_input_connected':bypass,'broken_preview':broken,'outcome':outcome})
(r/'failure-detection.json').write_text(json.dumps({'api':observed,'bypass':guards,'scope':'Local HTTP fault responses and the exact Bypass assertion block; separate from real Chrome executions.'},indent=2)+'\n')
print('PASS: API faults rejected; only the known Processor failure becomes xfail.')
