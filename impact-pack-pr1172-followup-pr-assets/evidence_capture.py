"""Record the existing E2E assertions without changing their actions or outcomes."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


@pytest.fixture(autouse=True)
def record_mask_assertions(request, monkeypatch):
    module = request.module
    if module.__name__ != 'test_e2e_preview_bridge':
        return
    original = module.mask_output
    root = Path(os.environ['IMPACT_E2E_ARTIFACTS']) / request.node.name / 'comparisons'
    root.mkdir(parents=True, exist_ok=True)
    records = []

    def recorded(ui, entry, node_id, expected):
        index = len(records) + 1
        prefix = f'{index:02d}'
        passed = False
        try:
            result = original(ui, entry, node_id, expected)
            passed = True
            return result
        finally:
            meta, = entry['outputs'][node_id]['images']
            actual_bytes = module.file_bytes(ui, meta)
            (root / f'{prefix}-actual.png').write_bytes(actual_bytes)
            expected.save(root / f'{prefix}-expected.png')
            with Image.open(io.BytesIO(actual_bytes)) as image:
                actual = np.array(image.convert('L'))
                size = list(image.size)
            wanted = np.array(expected.convert('L'))
            error = int(np.abs(actual.astype(np.int16) - wanted.astype(np.int16)).max()) if actual.shape == wanted.shape else None
            screenshot = root / f'{prefix}-screen.png'
            ui.page.screenshot(path=str(screenshot))
            graph = ui.app.evaluate('''app => app.graph._nodes.map(node => ({
                id: String(node.id), type: node.type,
                bounds: [...app.canvasPosToClientPos(node.pos), node.size[0] * app.canvas.ds.scale, node.size[1] * app.canvas.ds.scale],
                widgets: (node.widgets ?? []).map(w => ({name: w.name, value: w.value, y: w.last_y}))
            }))''')
            record = {'step': index, 'prompt_id': entry['prompt'][1], 'node_id': node_id,
                      'assertion_passed': passed, 'actual_size': size, 'expected_size': list(expected.size),
                      'actual_nonzero': int(np.count_nonzero(actual)),
                      'expected_nonzero': int(np.count_nonzero(wanted)), 'max_error': error,
                      'output': meta, 'graph': graph, 'viewport': ui.page.viewport_size,
                      'screen': screenshot.name, 'actual': f'{prefix}-actual.png', 'expected': f'{prefix}-expected.png',
                      'screen_sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest(),
                      'actual_sha256': hashlib.sha256(actual_bytes).hexdigest()}
            (root / f'{prefix}-history.json').write_text(json.dumps(entry, indent=2) + '\n')
            records.append(record)
            (root / 'records.json').write_text(json.dumps(records, indent=2) + '\n')

    monkeypatch.setattr(module, 'mask_output', recorded)
