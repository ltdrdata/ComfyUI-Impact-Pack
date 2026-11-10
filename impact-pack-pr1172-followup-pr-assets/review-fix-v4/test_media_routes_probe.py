"""Recorded probe for an isolated browser_checks.py server, never the shared server."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

from PIL import Image
import requests


def test_view_upload_registration_and_raw_masks_agree():
    url = os.environ['IMPACT_E2E_BASE_URL']
    run = Path(os.environ['IMPACT_E2E_ARTIFACTS']).parent
    runtime = run / 'runtime'
    assert runtime.is_dir()
    other_disk = run / 'other-disk'
    other_disk.mkdir()
    data = io.BytesIO()
    Image.new('RGBA', (64, 64), (255, 0, 0, 128)).save(data, format='PNG')
    png = data.getvalue()
    observations = []

    def record(row):
        observations.append(row)
        (run / 'observations.json').write_text(json.dumps({
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows': observations,
        }, indent=2) + '\n')

    for storage in ['input', 'output', 'temp']:
        directory = runtime / storage
        directory.mkdir(exist_ok=True)
        outside = other_disk / storage
        outside.mkdir()
        target = outside / 'mask.png'
        target.write_bytes(png)
        for link_kind in ['file', 'directory']:
            subfolder = 'probe-' + link_kind
            link = directory / subfolder
            if link_kind == 'directory':
                link.symlink_to(outside, target_is_directory=True)
            else:
                link.mkdir()
                (link / target.name).symlink_to(target)
            item = {'filename': target.name, 'subfolder': subfolder, 'type': storage}
            viewed = requests.get(url + '/view', params=item, timeout=15)
            assert viewed.status_code == 200
            assert viewed.content == png
            uploaded = requests.post(url + '/upload/image',
                data={'type': storage, 'subfolder': subfolder, 'overwrite': 'true'},
                files={'image': (target.name, png, 'image/png')}, timeout=15)
            assert uploaded.status_code == 200
            assert target.read_bytes() == png
            registered = requests.get(url + '/impact/set/pb_id_image',
                                      params=item | {'node_id': '2'}, timeout=15)
            assert registered.status_code == 200
            metadata = requests.get(url + '/impact/get/pb_id_image',
                                    params={'id': registered.text}, timeout=15)
            assert metadata.json() == item
            record({'storage': storage, 'link': link_kind, 'view_status': viewed.status_code,
                    'upload_status': uploaded.status_code, 'registration_status': registered.status_code,
                    'metadata': metadata.json(), 'file_sha256': hashlib.sha256(viewed.content).hexdigest()})
            references = {'registered': registered.text}
            if storage == 'input':
                references['raw'] = f'{subfolder}/{target.name} [input]'
            for route, reference in references.items():
                # Raw restoration must not inherit the registered node's last-mask cache.
                bridge_id = '2' if route == 'registered' else str(100 + len(observations))
                prompt = {
                    '1': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 64, 'height': 64, 'batch_size': 1}},
                    bridge_id: {'class_type': 'PreviewBridgeLatent', 'inputs': {
                        'latent': ['1', 0], 'image': reference, 'preview_method': 'Latent2RGB-SD15',
                        'restore_mask': 'always', 'block': False}},
                    '3': {'class_type': 'MaskToImage', 'inputs': {'mask': [bridge_id, 1]}},
                    '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': 'media-path-probe'}},
                }
                submitted = requests.post(url + '/prompt', json={'prompt': prompt}, timeout=15)
                assert submitted.status_code == 200, submitted.text
                prompt_id = submitted.json()['prompt_id']
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    history = requests.get(url + '/history/' + prompt_id, timeout=15).json()
                    if prompt_id in history:
                        break
                    time.sleep(.1)
                else:
                    raise AssertionError('Prompt did not finish')
                entry = history[prompt_id]
                assert entry['status']['status_str'] == 'success', entry['status']
                output, = entry['outputs']['4']['images']
                actual = requests.get(url + '/view', params=output, timeout=15)
                assert actual.status_code == 200
                with Image.open(io.BytesIO(actual.content)) as image:
                    size = image.size
                    extrema = image.convert('L').getextrema()
                (run / 'masks').mkdir(exist_ok=True)
                (run / 'masks' / f'{storage}-{link_kind}-{route}.png').write_bytes(actual.content)
                record({'storage': storage, 'link': link_kind, 'route': route,
                        'prompt_id': prompt_id, 'bridge_id': bridge_id, 'mask_size': size,
                        'mask_extrema': extrema, 'expected_mask_value': 127,
                        'output_sha256': hashlib.sha256(actual.content).hexdigest()})
                assert size == (64, 64), size
                assert all(abs(value - 127) <= 1 for value in extrema), extrema
        for escape, subfolder in [('absolute', str(outside)), ('parent', '../../' + other_disk.name + '/' + storage)]:
            params = {'filename': target.name, 'subfolder': subfolder, 'type': storage}
            response = requests.get(url + '/impact/set/pb_id_image', params=params | {'node_id': '2'}, timeout=15)
            record({'storage': storage, 'escape': escape, 'registration_status': response.status_code})
            assert response.status_code == 400
