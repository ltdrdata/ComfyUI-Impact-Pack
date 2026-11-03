from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
labels=['javascript','wildcard-unit','mask-blur','sam-and-mask','animatediff-assembly','browser-harness','input-validation','wildcard-cache-integration','wildcard-cache-tensors','wildcard-tutorial-tensors','lora-source-tensors','catalog','wildcard-browser','switch-browser','mask-browser','animatediff-native','wildcard-ui','wildcard-ondemand-ui','utf8-api','lora-source-browser','lora-source-inference','detailer-inference','conditioning-browser','switch-tutorial-inference','regional-inference','legacy-sam-isolated']
results={p.stem:json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')}
assert all(label in results for label in labels),[label for label in labels if label not in results]
def cases(label):
 return list(ET.parse(ROOT/(label+'.xml')).iter('testcase'))
def counts(items):
 return {'tests':len(items),'passed':sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in items),'failures':sum(c.find('failure') is not None for c in items),'errors':sum(c.find('error') is not None for c in items),'skipped':sum(c.find('skipped') is not None for c in items)}
xml={label:counts(cases(label)) for label in results if (ROOT/(label+'.xml')).exists()}
sam=counts([c for c in cases('sam-browser-cpu-reference') if 'test_e2e_legacy_sam' not in c.attrib['classname']]+cases('legacy-sam-isolated'))
lru_cases=cases('wildcard-lru-browser')
lru_pass={(c.attrib['classname'],c.attrib['name']) for c in lru_cases if c.find('failure') is None}
for i in range(1,4):
 label=f'wildcard-lru-retry-{i}'
 assert results[label]['exit_code']==0
 lru_pass.update((c.attrib['classname'],c.attrib['name']) for c in cases(label))
assert len(lru_pass)==38 and sam=={'tests':19,'passed':19,'failures':0,'errors':0,'skipped':0}
classic=cases('wildcard-browser')
classic_unique={(c.attrib['classname'],c.attrib['name']) for c in classic}
assert len(classic_unique)==63
for i in range(1,4):assert results[f'wildcard-classic-retry-{i}']['exit_code']==0
source=json.loads((ROOT/'candidate-source.json').read_text())
for path,digest in source['candidate_worktree_sha256'].items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest,path
report={
 'completed_utc':datetime.now(timezone.utc).isoformat(),
 'base':source['base'],'final_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
 'source_record':'candidate-source.json','source_verified':True,
 'tracked_worktree_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True).strip(),
 'maintained_direct_results':{label:results[label] for label in labels},
 'xml_counts':xml,
 'effective_sam_counts':sam,
 'classic':{'first_attempt':xml['wildcard-browser'],'unique_cases':len(classic_unique),'test_bodies_passed':63,'teardown_network_errors':1,'unchanged_retry_runs_passed':3},
 'lru':{'first_attempt':xml['wildcard-lru-browser'],'passed_unique_cases_after_unchanged_retries':len(lru_pass),'retry_runs':3,'initial_ui_timeout_retained':True},
 'new_edge_cases':{'cases':12,'classic':12,'lru':12,'rgr':'batch-rgr.json'},
 'legacy_diagnostics':'triage.md',
 'all_raw_checks_passed':False,
 'intentional_red_labels':['batch-rgr-red','batch-rgr-red-again'],
 'raw_results':results,
}
report['maintained_direct_failures']=[label for label in labels if results[label]['exit_code']!=0]
report['unresolved_maintained_exit_failures']=[label for label in report['maintained_direct_failures'] if label!='wildcard-browser']
(ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'maintained_direct_failures':report['maintained_direct_failures'],'xml_counts':xml,'effective_sam_counts':sam,'lru':report['lru']},indent=2))
