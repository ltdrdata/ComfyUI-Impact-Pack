from runlib import *
import hashlib
import xml.etree.ElementTree as ET

candidate = (REPO / 'js/impact-pack.js').read_bytes()
baseline = subprocess.check_output(['git', 'show', META['base'] + ':js/impact-pack.js'], cwd=REPO)
results = {}
with server('bypass-compare', 18517) as url:
    target = ROOT / 'runtime-bypass-compare/custom_nodes/comfyui-impact-pack/js/impact-pack.js'
    try:
        for name, source in [('main', baseline), ('candidate', candidate)]:
            target.write_bytes(source)
            result = pytest('bypass-' + name, 'tests/test_e2e_wildcard_controls.py',
                '-k', 'Bypass-populated-only and ImpactWildcardProcessor', env={'IMPACT_E2E_BASE_URL': url})
            suite = ET.parse(ROOT / ('bypass-' + name + '.xml')).find('testsuite')
            counts = {key: int(suite.get(key)) for key in ('tests', 'failures', 'errors', 'skipped')}
            assert counts == {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 2}, counts
            results[name] = {'source_js_sha256': hashlib.sha256(source).hexdigest(), 'result': result, 'counts': counts}
    finally:
        target.write_bytes(candidate)
(ROOT / 'bypass-comparison.json').write_text(json.dumps({
    'main': META['base'], 'same_two_xfail_on_main_and_candidate': True, 'results': results,
    'symptom': 'Processor Bypass with only populated_text connected drops PreviewAny.source; positional and named restoration.',
}, indent=2) + '\n')
