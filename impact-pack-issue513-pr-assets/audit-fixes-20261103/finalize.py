from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;R=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
source=json.loads((P/'candidate-source.json').read_text());assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==s for f,s in source['sha256'].items());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert head==source['parent_head']=='3452e437c982c7578cc4dd4a04ff748eb4484df6'
expected={'wildcard-unit':205,'other-python':168,'ordinary-browser':62,'catalog-browser':14,'mode-browser':4,'mode-lru-browser':4};results={}
for label,count in expected.items():
 d=json.loads((P/'results'/(label+'.json')).read_text());assert d['exit_code']==0 and d['counts']=={'tests':str(count),'failures':'0','errors':'0','skipped':'0'},(label,d);results[label]=d
matrix={}
for cache,focused in [('classic','mode-browser'),('lru','mode-lru-browser')]:
 initial=json.loads((P/'results'/('popup-promoted-'+cache+'.json')).read_text());doc=ET.parse(P/('popup-promoted-'+cache+'.xml'));base=[c for c in doc.iter('testcase') if 'promoted-mode' not in c.get('name')];assert len(base)==20 and all(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in base)
 final=ET.parse(P/(focused+'.xml'));cases=list(final.iter('testcase'));assert len(cases)==4 and len({c.get('name') for c in [*base,*cases]})==24
 matrix[cache]={'validated_scenarios':24,'unchanged_cases_passed':20,'updated_mode_cases_passed':4,'initial_popup_run':initial,'final_mode_run':results[focused],'explanation':'Only the promoted-mode interaction changed after the initial matrix; all 20 other paths are unchanged. The four affected cases were rerun with a single native arrow press, no retries.'}
node=json.loads((P/'node-runtime-comparison.json').read_text());assert all(row['exit_code']==0 and row['passed_60'] for row in node['runtimes']['24.12.0']['results']);assert len(node['runtimes']['24.12.0']['results'])==10
unit=json.loads((P/'unit-rgr.json').read_text());assert unit['candidate_js_sha256']==source['sha256']['js/impact-pack.js'];assert len(unit['results']['red']['failures'])==4 and unit['results']['green']['exit_code']==0 and unit['results']['red']['failures']==unit['results']['red-again']['failures']
artifacts=json.loads((P/'artifacts-rgr.json').read_text());assert artifacts['results']['green']['exit_code']==0 and len(artifacts['results']['red']['failures'])==2 and artifacts['results']['red']['failures']==artifacts['results']['red-again']['failures']
browser=json.loads((P/'browser-mode-rgr.json').read_text());assert browser['candidate_js_sha256']==source['sha256']['js/impact-pack.js'];assert browser['results']['green']['exit_code']==0 and len(browser['results']['red']['failures'])==2 and browser['results']['red']['failures']==browser['results']['red-again']['failures']
for phase in ('red','red-again'):
 for case in ET.parse(P/f'browser-mode-rgr-{phase}.xml').iter('testcase'):
  failure=case.find('failure');assert failure is not None and 'assert promoted_widget_state(ui, host) == before' in failure.text and 'TimeoutError' not in failure.text
sampling=json.loads((P/'sampling/comparison.json').read_text());assert sampling['successful'] and sampling['candidate_source_js_sha256']==source['sha256']['js/impact-pack.js']
for phase in ('candidate','baseline'):
 for q in (1,2,3):
  d=json.loads((P/f'sampling/{phase}/queue-{q}-widget-state.json').read_text());vals=lambda widgets:{w['name']:w['value'] for w in widgets};assert vals(d['root']['20']['widgets'])['mode']=='fixed' and vals(d['interior']['widgets'])['mode']=='populate'
assert '# pass 4\n' in (P/'original-audit-probes.log').read_text();assert json.loads((P/'adapter-disabled.json').read_text())['failed_tests']==4
servers=[]
for f in P.glob('server-*.json'):
 d=json.loads(f.read_text());proc=Path(f"/proc/{d['pid']}/cmdline");assert not (proc.exists() and d['runtime'] in proc.read_text()),f;servers.append({'profile':f.stem,'stopped':True})
assert not subprocess.check_output(['git','diff','--name-only','origin/Main','--','modules','__init__.py','tests/workflows'],cwd=R)
assert not subprocess.check_output(['git','diff','--cached'],cwd=R)
new='tests/test_wildcard_cache_artifacts.py';addition=subprocess.run(['git','diff','--no-index','--','/dev/null',new],cwd=R,stdout=subprocess.PIPE);assert addition.returncode==1
for label,base in [('review-fixes.patch','HEAD'),('pr-candidate.patch','origin/Main')]:
 (P/label).write_bytes(subprocess.check_output(['git','diff',base],cwd=R)+addition.stdout)
resolutions={
 'F1':{'status':'fixed','evidence':['unit-rgr.json','original-audit-probes.log'],'change':'Restore a known shared control before mode resolution can decline the override.'},
 'F2':{'status':'fixed','evidence':['unit-rgr.json','original-audit-probes.log'],'change':'An existing but unreadable mode projection returns unsupported, not the interior default.'},
 'F3':{'status':'fixed','evidence':['adapter-disabled.json','adapter-disabled.log'],'change':'Two duplicate tests replaced by four pending-state tests; disabling the adapter now makes all four fail.'},
 'F4':{'status':'fixed','evidence':['mode-browser.xml','mode-lru-browser.xml','browser-mode-rgr.json','sampling/comparison.json'],'change':'Host fixed versus interior populate is asserted in positional and named serialization, queued three times and rendered. Ignoring host mode is detected in after timing. Before timing retains the separate native core issue.'},
 'F5':{'status':'fixed','evidence':['artifacts-rgr.json'],'change':'Capture/save the original POST before checking backend output or UI preview; three focused regressions include failure paths and normalized history.'},
 'F6':{'status':'fixed','change':'Three fixed repetitions per matrix scenario: 144 planned fixed submissions over both caches, down from 480; native reload and manual invalidation assertions retained.'}}
(P/'findings-resolution.json').write_text(json.dumps(resolutions,indent=2)+'\n')
summary={'completed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'state':'uncommitted local fixes; no commit, push, PR edit or evidence publication','parent_head':head,'base':subprocess.check_output(['git','rev-parse','origin/Main'],cwd=R,text=True).strip(),'candidate_source':source,'findings_resolution':resolutions,'node_tests':{'passed':60,'validated_runtime':'24.12.0','full_suite_runs':10,'runtime_comparison':node,'boundary':'Node20 SIGSEGV root cause is not claimed fixed; Node24 is the bounded validated runner, with no global runtime changes.'},'python_passed':373,'results':results,'promoted_matrix':matrix,'unit_rgr':unit,'artifact_rgr':artifacts,'promoted_mode_browser_rgr':browser,'sampling':sampling,'servers':servers,'trusted_fixtures_unchanged':True,'diagnostics':['Popup-based initial mode failures and the first wrong mode-index assertion are retained; final code uses one native arrow action and mode index 2 in both node types.','Ruff was once incorrectly run on JavaScript; corrected Python lint and Node syntax checks passed.','Initial Node20 RGR SIGSEGV and an initial four-case mutation expectation are retained separately. The final browser mutation RGR selects after timing to isolate the existing before-mode core limitation.'],'remaining_limits':['Independent Inspire GlobalSeed may change upstream inputs.','Native promoted populate/before projection bookkeeping remains a separate frontend issue.','Artifact regressions cover output/preview assertion failures after successful UI execution; they do not claim every earlier submission/backend failure path.','Broader historical model/SAM regression is not presented as rerun here.']}
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Six findings resolved; all scoped checks, source hashes, original counterexamples, real renders and stopped owned servers verified. No remote changes.')
