from pathlib import Path
import subprocess,json,hashlib
P=Path(__file__).resolve().parent;R=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls');D=P/'unit-rgr-final';(D/'tests').mkdir(parents=True,exist_ok=True);(D/'js').mkdir(exist_ok=True)
for f in ['tests/wildcard_seed_control.test.mjs','tests/workflows/all-impact-nodes.widgets.json']:
 (D/f).parent.mkdir(parents=True,exist_ok=True);(D/f).write_bytes((R/f).read_bytes())
baseline=subprocess.check_output(['git','show','HEAD:js/impact-pack.js'],cwd=R);candidate=(R/'js/impact-pack.js').read_bytes();rows={}
for phase,data in [('red',baseline),('green',candidate),('red-again',baseline)]:
 (D/'js/impact-pack.js').write_bytes(data)
 cmd=['node','--experimental-vm-modules','--test','tests/wildcard_seed_control.test.mjs']
 r=subprocess.run(cmd,cwd=D,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(P/('latest-unit-rgr-'+phase+'.log')).write_bytes(r.stdout)
 lines=r.stdout.decode().splitlines();failed=[x for x in lines if x.startswith('not ok ')];rows[phase]={'command':cmd,'exit_code':r.returncode,'counts':[x for x in lines if x.startswith(('# tests ','# pass ','# fail '))],'failed':failed};print(phase,rows[phase]['counts'],flush=True)
assert rows['green']['exit_code']==0 and rows['red']['failed'] and rows['red']['failed']==rows['red-again']['failed']
(P/'latest-unit-rgr.json').write_text(json.dumps({'baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'candidate_js_sha256':hashlib.sha256(candidate).hexdigest(),'test_sha256':hashlib.sha256((R/'tests/wildcard_seed_control.test.mjs').read_bytes()).hexdigest(),'phases':rows},indent=2)+'\n')
