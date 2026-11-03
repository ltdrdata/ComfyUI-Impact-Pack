from pathlib import Path
import hashlib,json,subprocess,tempfile

ROOT=Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
OUT=Path(__file__).resolve().parent
BASE='244e93ef9883aaf241d5d0e54ebb57e3b2e840d3'
SOURCE=ROOT/'js/impact-pack.js'
fixed=SOURCE.read_text()
baseline=subprocess.check_output(['git','show',BASE+':js/impact-pack.js'],cwd=ROOT,text=True)
testfile=ROOT/'tests/wildcard_seed_control.test.mjs'
oldtest=subprocess.check_output(['git','show','aaec21a:tests/wildcard_seed_control.test.mjs'],cwd=ROOT,text=True)
results={}

def run(label,source,test_source):
    with tempfile.TemporaryDirectory(prefix='impact-513-unit-') as directory:
        root=Path(directory)
        (root/'js').mkdir()
        (root/'tests/workflows').mkdir(parents=True)
        (root/'js/impact-pack.js').write_text(source)
        (root/'tests/wildcard_seed_control.test.mjs').write_text(test_source)
        (root/'tests/workflows/all-impact-nodes.widgets.json').write_bytes((ROOT/'tests/workflows/all-impact-nodes.widgets.json').read_bytes())
        command=['node','--experimental-vm-modules','--test','tests/wildcard_seed_control.test.mjs']
        p=subprocess.run(command,cwd=root,capture_output=True,text=True)
        log=p.stdout+p.stderr
        (OUT/(label+'.log')).write_text(log)
        import re
        counts={k:int(re.search(r'^# '+k+r' (\d+)$',log,re.M)[1]) for k in ['tests','pass','fail','cancelled','skipped']}
        failures=re.findall(r'^not ok \d+ - (.+)$',log,re.M)
        results[label]={'command':command,'exit_code':p.returncode,'counts':counts,'failures':failures,
                        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'test_sha256':hashlib.sha256(test_source.encode()).hexdigest()}
        assert counts['cancelled']==counts['skipped']==0
        assert 'SyntaxError' not in log and 'ERR_MODULE' not in log
        return p.returncode

for phase,source in [('red',baseline),('green',fixed),('red-again',baseline)]:
    assert run(phase,source,testfile.read_text())==(0 if phase=='green' else 1)
assert results['red']['failures']==results['red-again']['failures']
assert results['red']['counts']['fail']==8
# Mutate only the fixed-mode branch; non-fixed delegation remains intact.
marker="\t\t\t\t\t\t\treturn original.apply(this, args);"
assert fixed.count(marker)==1
mutations={
    'drop-fixed-arguments':fixed.replace(marker,marker.replace('this, args','this, []')),
    'drop-fixed-return':fixed.replace(marker,marker.replace('return original','original')),
    'wrong-fixed-receiver':fixed.replace(marker,marker.replace('this, args','null, args')),
    'skip-core-fixed':fixed.replace(marker,'\t\t\t\t\t\t\treturn;'),
    'lose-selected-policy':fixed.replace('this.value = selected;','this.value = "increment";'),
}
for name,source in mutations.items():
    run('old-tests-'+name,source,oldtest)
    assert run('mutation-'+name,source,testfile.read_text())==1
assert SOURCE.read_text()==fixed
(OUT/'unit-review.json').write_text(json.dumps(results,indent=2)+'\n')
for label,result in results.items(): print(label,result['counts'])
