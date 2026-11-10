"""Extract submitted prompts and outputs from existing, unmodified Playwright traces."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent
comparisons = []
for run in ('feedback-fixes/red-shared', 'feedback-fixes/green-chrome'):
    for kind in ('PreviewBridge', 'PreviewBridgeLatent'):
        case = f'test_queued_bridge_results_preserve_the_latest_saved_edit[shared_file-{kind}]'
        trace = ASSETS / run / 'browser' / case / 'trace.zip'
        histories = {}
        with zipfile.ZipFile(trace) as archive:
            for line in archive.read('trace.network').decode().splitlines():
                record = json.loads(line)['snapshot']
                if '/history/' not in record['request']['url']:
                    continue
                content = record['response']['content'].get('_sha1')
                if content:
                    histories.update(json.loads(archive.read('resources/' + content)))
        stages = []
        for entry in histories.values():
            prompt = entry['prompt'][2]
            bridge_id, bridge = next((key, node) for key, node in prompt.items() if node['class_type'] == kind)
            source = prompt[bridge['inputs']['images' if kind == 'PreviewBridge' else 'latent'][0]]['inputs']
            conversion = next(key for key, node in prompt.items() if node['class_type'] == 'MaskToImage')
            output_id = next(key for key, node in prompt.items()
                             if node['class_type'] == 'PreviewImage' and node['inputs']['images'][0] == conversion)
            output, = entry['outputs'][output_id]['images']
            path = ASSETS / run / 'runtime' / output['type'] / output['subfolder'] / output['filename']
            with Image.open(path) as image:
                size = list(image.size)
                nonzero = int(np.count_nonzero(np.array(image.convert('L'))))
            stages.append({'prompt_id': entry['prompt'][1], 'input_size': [source['width'], source['height']],
                           'restore_mask': bridge['inputs']['restore_mask'], 'image_reference': bridge['inputs']['image'],
                           'mask_size': size, 'mask_nonzero': nonzero, 'status': entry['status']['status_str'],
                           'output': str(path.relative_to(ASSETS)), 'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        assert len(stages) == (3 if run.endswith('red-shared') else 4), stages
        assert stages[1]['image_reference'] == stages[2]['image_reference']
        assert stages[1]['input_size'] == [512, 512]
        assert stages[2]['input_size'] == [256, 256]
        comparisons.append({'run': run, 'node': kind, 'trace': str(trace.relative_to(ASSETS)),
                            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(), 'stages': stages})

(ASSETS / 'shared-reference-sequence.json').write_text(json.dumps({
    'method': 'Read-only extraction of actual HTTP history responses from Playwright traces. '
              'Bridge execution feedback is held until both prompts finish. Executions themselves remain sequential.',
    'comparisons': comparisons,
}, indent=2) + '\n')
for row in comparisons:
    print(row['run'], row['node'], [(s['prompt_id'], s['input_size'], s['mask_size'], s['mask_nonzero']) for s in row['stages']])
