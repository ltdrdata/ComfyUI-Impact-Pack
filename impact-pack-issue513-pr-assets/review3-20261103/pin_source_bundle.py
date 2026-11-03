"""Pin the focused test bundle to canonical blobs after the code commit exists."""
from pathlib import Path
import hashlib
import json
import subprocess

P = Path(__file__).resolve().parent
R = P.parent.parent / 'impact-pack-pr-issue513-fixed-seed-controls'
mapping = json.loads((P / 'commit.json').read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip() == mapping['head']
manifest = json.loads((P / 'source-bundle.json').read_text())
(P / 'validated-source/.gitattributes').write_text('* -text\n')
for name in manifest:
    data = subprocess.check_output(['git', 'show', mapping['head'] + ':' + name], cwd=R)
    (P / 'validated-source' / name).write_bytes(data)
    manifest[name] = hashlib.sha256(data).hexdigest()
(P / 'source-bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
