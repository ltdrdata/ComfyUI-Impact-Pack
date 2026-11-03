from pathlib import Path
import subprocess,json,hashlib,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;R=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls');PY='/home/rho/venv/bin/python'
def node_rgr():
 d=P/'unit-rgr';(d/'js').mkdir(parents=True,exist_ok=True)
 for f in ['tests/wildcard_seed_control.test.mjs','tests/workflows/all-impact-nodes.widgets.json']:
  (d/f).parent.mkdir(parents=True,exist_ok=True);(d/f).write_bytes((R/f).read_bytes())
 old=(P/'before/js/impact-pack.js').read_bytes();new=(R/'js/impact-pack.js').read_bytes();rows={}
 for phase,data in [('red',old),('green',new),('red-again',old)]:
  (d/'js/impact-pack.js').write_bytes(data);cmd=['/home/rho/.nvm/versions/node/v24.12.0/bin/node','--experimental-vm-modules','--test-reporter=tap','--test','tests/wildcard_seed_control.test.mjs'];r=subprocess.run(cmd,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(P/('unit-rgr-'+phase+'.log')).write_bytes(r.stdout);lines=r.stdout.decode().splitlines();rows[phase]={'command':cmd,'exit_code':r.returncode,'counts':[x for x in lines if x.startswith(('# tests ','# pass ','# fail '))],'failures':[x for x in lines if x.startswith('not ok ')]}
 assert len(rows['red']['failures'])==4 and rows['green']['exit_code']==0 and rows['red']['failures']==rows['red-again']['failures']
 (P/'unit-rgr.json').write_text(json.dumps({'candidate_js_sha256':hashlib.sha256(new).hexdigest(),'test_sha256':hashlib.sha256((R/'tests/wildcard_seed_control.test.mjs').read_bytes()).hexdigest(),'results':rows},indent=2)+'\n');print('Node RGR:',[r['counts'] for r in rows.values()],flush=True)
def artifact_rgr():
 d=P/'artifact-rgr/tests';d.mkdir(parents=True,exist_ok=True)
 for f in ['pytest.ini','test_wildcard_cache_artifacts.py','e2e_ui.py','test_e2e_widget_catalog.py']:(d/f).write_bytes((R/'tests'/f).read_bytes())
 old=(P/'before/tests/test_e2e_wildcard_cache.py').read_bytes();new=(R/'tests/test_e2e_wildcard_cache.py').read_bytes();rows={}
 for phase,data in [('red',old),('green',new),('red-again',old)]:
  (d/'test_e2e_wildcard_cache.py').write_bytes(data);xml=P/('artifacts-rgr-'+phase+'.xml');cmd=[PY,'-m','pytest','test_wildcard_cache_artifacts.py','-q','--tb=short','--junitxml='+str(xml)]
  with (P/('artifacts-rgr-'+phase+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT)
  doc=ET.parse(xml);rows[phase]={'command':cmd,'exit_code':r.returncode,'counts':doc.find('testsuite').attrib,'failures':[c.get('name') for c in doc.iter('testcase') if c.find('failure') is not None]}
 assert len(rows['red']['failures'])==2 and rows['green']['exit_code']==0 and rows['red']['failures']==rows['red-again']['failures']
 (P/'artifacts-rgr.json').write_text(json.dumps({'test_sha256':hashlib.sha256((R/'tests/test_wildcard_cache_artifacts.py').read_bytes()).hexdigest(),'results':rows},indent=2)+'\n');print('Artifact RGR: 2 failures / 1 pass -> 3 pass -> same 2 failures / 1 pass',flush=True)
node_rgr();artifact_rgr()
