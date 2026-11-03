from runlib import *
import hashlib

candidate = (REPO / 'js/impact-pack.js').read_bytes()
baseline = subprocess.check_output(['git', 'show', META['base'] + ':js/impact-pack.js'], cwd=REPO)
reports = {}
with server('bridge-rgr', 18516) as url:
    target = ROOT / 'runtime-bridge-rgr/custom_nodes/comfyui-impact-pack/js/impact-pack.js'
    try:
        for phase, source in [('red', baseline), ('green', candidate), ('red-again', baseline)]:
            target.write_bytes(source)
            report = pytest('bridge-rgr-' + phase, 'tests/test_e2e_wildcard_controls.py',
                '-k', 'control_bridge_stop_with_constant and before_wildcard and positional',
                env={'IMPACT_E2E_BASE_URL': url})
            report['served_js_sha256'] = hashlib.sha256(source).hexdigest()
            reports[phase] = report
    finally:
        target.write_bytes(candidate)
import xml.etree.ElementTree as ET
def failed(phase):
    root = ET.parse(ROOT / ('bridge-rgr-' + phase + '.xml'))
    return [(test.get('name'), test.find('failure').get('message'))
            for test in root.iter('testcase') if test.find('failure') is not None]
assert reports['red']['exit_code'] == reports['red-again']['exit_code'] == 1
assert reports['green']['exit_code'] == 0 and len(failed('red')) == 2
assert failed('red') == failed('red-again')
assert all('731' in message and '732' in message for _, message in failed('red'))
(ROOT / 'bridge-rgr.json').write_text(json.dumps({'base': META['base'],
    'candidate_js_sha256': hashlib.sha256(candidate).hexdigest(), 'results': reports,
    'red_failures': failed('red'), 'intentional_red': True,
    'scope': 'Same fixed-mode seed and selected-policy assertions on Main, candidate, Main; all backend files and test bytes unchanged.'}, indent=2) + '\n')
