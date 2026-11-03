from runlib import *
import re

temporary = ROOT / 'legacy-api-temp'
temporary.mkdir(exist_ok=True)
scripts = {}
for name in ['test_dynamic_prompts_full.sh', 'wildcards/run_quick_test.sh']:
    text = (REPO / 'tests' / name).read_text()
    text = re.sub(r'\b(?:8188|819[0-9])\b', '18506', text).replace('/tmp/', str(temporary) + '/')
    assert 'pkill ' not in text and 'killall ' not in text
    path = temporary / Path(name).name
    path.write_text(text)
    scripts[Path(name).stem] = path

results = {}
corpus = ROOT / 'legacy-isolated/runtime/custom_nodes/comfyui-impact-pack/custom_wildcards'
for revision, options in [('candidate', []), ('baseline', ['--baseline'])]:
    with server('legacy-api-' + revision, 18506, '--samples', str(corpus), '--wildcard-limit', '0.000001', *options) as url:
        for name, path in scripts.items():
            label = 'legacy-api-' + revision + '-' + name
            record = run(label, ['bash', path], cwd=REPO,
                         env={'PATH': '/home/rho/venv/bin:' + os.environ.get('PATH', '')})
            text = (ROOT / (label + '.log')).read_text(errors='replace')
            text = re.sub(r'(?:\x1b|\\033)\[[0-9;]*m', '', text)
            found = [line for line in text.splitlines() if re.search(r'❌|✗|\[FAIL\]|\bFAILED\b|FAIL:|Failed:\s*[1-9]', line)]
            results[label] = {'record': record, 'failure_markers': found}
(ROOT / 'legacy-api-diagnostics.json').write_text(json.dumps({
    'candidate': META['head'], 'base': META['base'], 'records': results,
    'adaptation': 'Only local server port and temporary paths; assertions unchanged. Both revisions use the same copied corpus in on-demand mode.',
}, indent=2) + '\n')
