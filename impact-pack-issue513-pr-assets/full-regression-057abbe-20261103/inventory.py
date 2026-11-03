from runlib import *
import configparser

excluded = {
    'tests/e2e_ui.py': 'Shared browser interaction library; tested by test_e2e_ui.py.',
    'tests/restart_test_server.sh': 'Server administration helper, not a regression suite.',
    'tests/wildcards/find_transitive_wildcards.sh': 'Corpus discovery helper, not assertions.',
}
entries = []
for path in sorted((REPO / 'tests').rglob('*')):
    if not path.is_file():
        continue
    name = path.name
    if name.endswith('.test.mjs') or path.suffix == '.sh' or (
        path.suffix == '.py' and name.startswith(('test_', 'integration_', 'e2e_'))):
        entries.append(str(path.relative_to(REPO)))
records = [json.loads(path.read_text()) for path in (ROOT / 'results').glob('*.json')]
parser = configparser.ConfigParser()
parser.read(REPO / 'tests/pytest.ini')
defaults = ['tests/' + item for item in parser['pytest']['testpaths'].split()]
coverage = {}
for name in entries:
    if name in excluded:
        coverage[name] = {'status': 'auxiliary', 'reason': excluded[name]}
        continue
    invocations = []
    for record in records:
        command = record['command']
        explicit = any(Path(argument).name == Path(name).name for argument in command)
        default = name in defaults and record['label'] == 'wildcard-unit'
        if explicit or default:
            invocations.append({'label': record['label'], 'exit_code': record['exit_code']})
    coverage[name] = {'status': 'executed' if invocations else 'pending', 'invocations': invocations}
report = {'head': META['head'], 'base': META['base'], 'entrypoints': coverage,
          'executed': sum(item['status'] == 'executed' for item in coverage.values()),
          'pending': [name for name, item in coverage.items() if item['status'] == 'pending'],
          'auxiliary': excluded}
(ROOT / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'executed': report['executed'], 'pending': report['pending'], 'auxiliary': len(excluded)}, indent=2))
