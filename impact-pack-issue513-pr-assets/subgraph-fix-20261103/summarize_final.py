from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;R=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();base=subprocess.check_output(['git','rev-parse','origin/Main'],cwd=R,text=True).strip()
frozen=json.loads((P/'final-main-frozen.json').read_text())
assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==s for f,s in frozen['sha256'].items())
assert subprocess.check_output(['git','rev-list','--count','origin/Main..HEAD'],cwd=R,text=True).strip()=='1'
expected={'latest-wildcard-unit':205,'latest-other-python':165,'latest-ordinary-browser':62,'latest-catalog-browser':14,'latest-connection-browser':30,'settled-mode-classic':4,'settled-mode-lru':4}
rows={}
for label,count in expected.items():
 d=json.loads((P/'results'/(label+'.json')).read_text());assert d['exit_code']==0,(label,d)
 assert d['counts']=={'tests':str(count),'failures':'0','errors':'0','skipped':'0'},(label,d)
 rows[label]=d
node=json.loads((P/'results/latest-node-all.json').read_text());assert node['exit_code']==0
assert '# tests 54\n' in (P/'latest-node-all.log').read_text() and '# pass 54\n' in (P/'latest-node-all.log').read_text();rows['latest-node-all']=node|{'tests':54}
lru=json.loads((P/'results/latest-promoted-lru.json').read_text());assert lru['counts']=={'tests':'24','failures':'4','errors':'0','skipped':'0'}
doc=ET.parse(P/'settled-mode-lru.xml');passed=[c.attrib['name'] for c in doc.iter('testcase')];assert sorted(passed)==sorted(lru['failures'])
assert all('promoted-mode' in f for f in lru['failures']);rows['latest-promoted-lru-before-link-normalization']=lru
classic=json.loads((P/'results/final-promoted-classic.json').read_text());assert classic['counts']['tests']=='24' and classic['counts']['errors']=='0' and classic['counts']['skipped']=='0'
assert all('promoted-mode' in name for name in classic['failures']);rows['final-promoted-classic-before-host-redraw']=classic
redraw=json.loads((P/'results/final-promoted-lru-retry.json').read_text());assert redraw['counts']=={'tests':'4','failures':'2','errors':'0','skipped':'0'};rows['final-promoted-lru-before-host-redraw']=redraw
for label in ('final-mode-classic','final-mode-lru','arrow-mode-classic','arrow-mode-lru'):
 rows[label+'-dropdown-diagnostic']=json.loads((P/'results'/(label+'.json')).read_text())
rgr=json.loads((P/'final-browser-rgr.json').read_text());assert rgr['green']['exit_code']==0 and len(rgr['red']['failures'])==4 and rgr['red']['failures']==rgr['red-again']['failures']
unit=json.loads((P/'latest-unit-rgr.json').read_text());assert len(unit['phases']['red']['failed'])==8 and unit['phases']['green']['exit_code']==0 and unit['phases']['red']['failed']==unit['phases']['red-again']['failed']
sampling=json.loads((P/'sampling/comparison.json').read_text());assert sampling['successful'] and sampling['candidate_source_js_sha256']==frozen['sha256']['js/impact-pack.js']
servers=[]
for f in P.glob('server-*.json'):
 d=json.loads(f.read_text());proc=Path(f"/proc/{d['pid']}/cmdline");running=proc.exists() and d['runtime'] in proc.read_text()
 servers.append({'profile':f.stem,'pid':d['pid'],'stopped':not running});assert not running,f
source={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in frozen['sha256']}
commit={'head':head,'base':base,'commits':1,'source_sha256':source,'sampling_candidate_source_matches_final_commit':True,'sampling_capture_parent_head':sampling['capture_parent_head'],'sampling_baseline':sampling['base'],'trusted_fixtures_unchanged':True}
(P/'commit.json').write_text(json.dumps(commit,indent=2)+'\n')
report={'completed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'head':head,'base':base,'commits':1,'source_sha256':source,'results':rows,'promoted_lru':{'scenarios':24,'initial_passed':20,'initial_native_link_order_failures':4,'corrected_comparison_passed':4,'correction':'Sort complete subgraph link records by ID, then redraw after manual editing and use the native combo arrow for mode selection; preserve all records, arrays and assertions.'},'unit_rgr':unit,'browser_rgr':rgr,'sampling':sampling,'servers':servers,'source_boundaries':{'production_js_unchanged_during_all_latest_runs':True,'ordinary_tests_ran_before_order_only_subgraph_link_comparison_correction':True,'classic_full_and_browser_rgr_precede_redraw_only_mode_step':True,'native_mode_arrow_classic_and_lru_use_final_test_source':True,'all_four_promoted_mode_cases_reran_on_classic_and_lru_with_final_source':True,'earlier_broad_full_regression':'Historical previous implementation; not rerun as part of promoted-seed change.'},'remaining_limitation':'Independent upstream Inspire GlobalSeed may still change fixed inputs; manual/upstream edits intentionally invalidate.','diagnostics':'review-notes.md; initial native link-order failures, an interrupted/occupied-port attempt and one Node vm SIGSEGV plus transient promoted dropdown failures are retained and excluded from successful runs.'}
(P/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('Verified source hashes, one commit, final suites, RGR, render parity, unchanged fixtures and stopped owned servers.')
