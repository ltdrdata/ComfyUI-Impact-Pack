"""Accept only fresh results for the pushed head; retain rejected invocations separately."""
from pathlib import Path
import datetime,hashlib,json,socket,subprocess,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent;A=R.parent;repo=A.parent/'impact-pack-pr1269-followup';meta=json.loads((R/'source.json').read_text())
def load(p):return json.loads((R/p).read_text())
def junit(path):
 cases=ET.parse(R/path).findall('.//testcase');out={'passed':0,'failed':[],'errors':[],'xfail':[],'skipped':[],'report':path,'case_ids':[]}
 for c in cases:
  key=c.get('classname')+'::'+c.get('name');out['case_ids'].append(key)
  if c.find('failure') is not None:out['failed'].append(key)
  elif c.find('error') is not None:out['errors'].append(key)
  elif c.find('skipped') is not None:out['xfail' if c.find('skipped').get('type')=='pytest.xfail' else 'skipped'].append(key)
  else:out['passed']+=1
 return out
spec={'configured_units':('configured-unit.xml',210),'browser_harness':('browser-harness.xml',3),'classic_cache_controls':('classic/cache-controls.xml',97),'catalog':('catalog/catalog.xml',14),'native_tutorials':('catalog/tutorial-native.xml',8),'switches':('catalog/switches.xml',55),'sam_mask_units':('sam-mask-unit.xml',109),'animatediff_assembly':('animatediff-unit.xml',44),'mask_blur':('mask-blur.xml',9),'mask_native':('cpu/mask-native.xml',4),'legacy_sam':('cpu/legacy-sam.xml',6),'sam_native':('gpu/sam.xml',10),'sam_api':('gpu/sam-api.xml',3),'lora_source':('lora/lora-source.xml',2),'lbw_parity':('lbw-candidate/lbw-parity.xml',1)}
groups={};seen=set()
for label,(path,count) in spec.items():
 g=junit(path);assert g['passed']==count,(label,g);assert not any(g[k] for k in ['failed','errors','skipped']),(label,g)
 assert len(g['xfail'])==(2 if label=='classic_cache_controls' else 0),(label,g)
 assert not seen.intersection(g['case_ids']),label;seen.update(g['case_ids']);groups[label]=g
assert sum(g['passed'] for g in groups.values())==575
reruns={k:junit(p) for k,p in [('lru64','verified-lru/cache.xml'),('loader_first','unit-reordered.xml')]}
assert reruns['lru64']['passed']==32 and reruns['loader_first']['passed']==210
for g in reruns.values():assert not any(g[k] for k in ['failed','errors','skipped','xfail']);assert set(g['case_ids'])<=seen
baseline=junit('baseline-controls/bypass.xml');assert baseline['xfail']==groups['classic_cache_controls']['xfail'];assert baseline['passed']==0
names=['classic','verified-lru','catalog','constant','cpu','gpu','lora','ondemand-corrected','wildcard-full','lbw-baseline','lbw-candidate','inference-corrected','full','pair-ondemand','baseline-controls']
jobs={name:load(name+'/result.json') for name in names}
for name,j in jobs.items():
 assert j['server_stopped'] and j['commands'],name
 for c in j['commands']:
  assert c['exit_code']==0 or (name=='cpu' and c['label'] in ['wildcard-ui','encoding-api'] and c['exit_code']==1),(name,c)

red=load('baseline-mixed-list/result.json');assert red['server_stopped'] and red['commands'][0]['exit_code']==1
before=load('baseline-mixed-list/observations.json');after=load('mixed-list/observations.json')
assert before[1]['requested_seed']==5 and before[1]['execution_seed']==2 and before[1]['actual']==['red','plain'] and not before[1]['matches_expected']
assert after[1]['requested_seed']==after[1]['execution_seed']==5 and after[1]['actual']==['blue','plain'] and all(x['matches_expected'] for x in after)
units=load('units-result.json');assert len(units)==13 and all(c['exit_code']==0 for c in units)
assert '# pass 3\n# fail 0' in (R/'legacy-sam-js.log').read_text();assert '# fail 0' in (R/'lora-source-js.log').read_text()
legacy=load('legacy-with-data/results.json');assert len(legacy)==11 and all(x.get('exit_code')==0 for x in legacy)
legacypy=load('legacy-python-with-data/result.json');assert len(legacypy)==1 and legacypy[0]['exit_code']==0
api=load('api-result.json');assert api['exit_code']==0 and api['initial_states']['full']['on_demand_mode'] is False and api['initial_states']['ondemand']['on_demand_mode'] is True
api_lines=(R/'pair-ondemand/pair.log').read_text().splitlines()
assert 'PASS: 7 matching results; both server modes verified' in api_lines
assert sum(line.startswith('PASS: ') for line in api_lines)==8
assert not load('ruff.json')['new_findings']
assert load('cache-hook.json')['test_groups']==20 and load('cache-hook.json')['successful']
assert load('cache-tensors.json')['tests_run']==5 and not load('cache-tensors.json')['failures']
assert len(load('inference-corrected/lora-inference/results.json'))==20
assert load('wildcard-ui-modes.json')['ondemand-corrected']['on_demand_mode'] is True
# Match every tracked file in every candidate source, allowing only recorded environment changes.
source_hashes=meta['source_hashes'];allowed={(a['source'],a['file']):a['sha256'] for a in meta['adaptations']}
for row in legacy:
 src=R/'legacy-with-data'/Path(row['file']).stem/'source';allowed[(str(src),row['file'])]=hashlib.sha256((src/row['file']).read_bytes()).hexdigest()
integrity=[]
sources=[R/'source',*R.glob('*-source'),*R.glob('*/source'),*R.glob('legacy-with-data/*/source')]
for src in sorted(set(sources)):
 is_base=src.name in ['baseline-source','baseline-controls-source'];checked=source_hashes if not is_base else {p:h for p,h in source_hashes.items() if p.startswith(('modules/','js/')) or p in ['__init__.py','install.py']}
 deltas=[]
 for name,h in checked.items():
  if is_base:
   expected=subprocess.check_output(['git','show',meta['baseline']+':'+name],cwd=repo);h=hashlib.sha256(expected).hexdigest()
  actual=hashlib.sha256((src/name).read_bytes()).hexdigest()
  if actual!=h:
   assert allowed.get((str(src),name))==actual,(src,name)
   assert name.startswith('tests/'),name;deltas.append(name)
 integrity.append({'source':str(src),'files_checked':len(checked),'baseline':is_base,'test_only_adaptations':deltas,'production_identical':True})
# Verify source branch and trusted native fixtures have not moved during execution.
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==meta['head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
for file,digest in source_hashes.items():assert hashlib.sha256((repo/file).read_bytes()).hexdigest()==digest,file
ports={load(n+'.json')['port'] for n in names+['baseline-mixed-list','inference','ondemand']}|{x['port'] for x in legacy}
for port in sorted(ports):
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,port
cleanup={'all_owned_servers_stopped':True,'closed_ports':sorted(ports)};(R/'cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
report={'head':meta['head'],'tree':meta['tree'],'baseline':meta['baseline'],'core':meta['core'],'started_utc':meta['started_utc'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'all_results_fresh':True,'unique_pytest_passed':575,'existing_xfails':groups['classic_cache_controls']['xfail'],'groups':groups,'repetitions_excluded_from_unique_total':reruns,'baseline_xfail':baseline,'jobs':jobs,'unit_and_integration_commands':units,'legacy_shell':legacy,'legacy_python':legacypy,'api':api,'baseline_red':{'result':'baseline-mixed-list/result.json','observations':before},'candidate_green':after,'source_integrity':integrity,'cleanup':cleanup,'new_product_failures':[],'rejected_invocations':load('runner-corrections.json'),'coverage_boundaries':['Encode in the hook/cache integration matrix uses a CLIP stand-in; separate tensor and browser sampling checks use installed models.','AnimateDiff coverage is assembly and native restoration, not video inference.','Legacy diagnostic exit codes are not asserted case counts. Seven full/on-demand pairs are checked separately with both actual modes verified.','This run tests pushed head 63d28f8, not a synthetic merge with current Main. Rebase validation is required after #1269 lands.']}
(R/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('ACCEPTED: 575 unique passed, 2 existing xfails; 32 LRU and 210 reordered repetitions excluded.')
