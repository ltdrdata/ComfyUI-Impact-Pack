"""Check completed local validation before amending or publishing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
R = Path('/mnt/teratera/git/impact-pack-pr-issue513-fixed-seed-controls')
BASE = '56a34b9c439470d07440f9104204a084e5fdb38c'
BEFORE = '6d1a48803313c0b38193366bcef8f31dd552e70d'
candidate = json.loads((P / 'candidate-source.json').read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip() == BEFORE
for name, digest in candidate['source_sha256'].items():
    assert hashlib.sha256((R / name).read_bytes()).hexdigest() == digest, name

labels = {'wildcard-unit': 205, 'other-python': 168, 'mode-browser': 4,
          'ordinary-browser': 62, 'catalog-browser': 14,
          'promoted-classic': 24, 'promoted-lru': 24}
results = {}
for label, count in labels.items():
    item = json.loads((P / 'results' / (label + '.json')).read_text())
    assert item['exit_code'] == 0 and item['counts'] == {
        'tests': str(count), 'failures': '0', 'errors': '0', 'skipped': '0'}, (label, item)
    results[label] = item
node = json.loads((P / 'results/node-all.json').read_text())
node_log = (P / 'node-all.log').read_text()
assert node['exit_code'] == 0 and re.search(r'^# tests 62$', node_log, re.M)
assert re.search(r'^# pass 62$', node_log, re.M) and re.search(r'^# fail 0$', node_log, re.M)
before_log = (P / 'node-before-refactor.log').read_text()
assert re.search(r'^# pass 33$', before_log, re.M) and re.search(r'^# fail 0$', before_log, re.M)
node['passed'] = 62
results['node-all'] = node

fixtures = {
    'tests/workflows/all-impact-nodes.json': 'a711b1eb0b2b4c62801a91ffc1dfae2d38d37b3adb587325e2da3f08aa08f7be',
    'tests/workflows/all-impact-nodes.widgets.json': '2cae64fdad3e82209eacb9a237b6fcdbad47968d1b7d516e6cddec1e4b44b3e9',
}
for name, digest in fixtures.items():
    assert hashlib.sha256((R / name).read_bytes()).hexdigest() == digest
    assert (R / name).read_bytes() == subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=R)
assert not subprocess.check_output(['git', 'diff', BASE, '--', 'modules', 'install.py'], cwd=R)
servers = []
for path in sorted(P.glob('server-*.json')):
    item = json.loads(path.read_text())
    cmdline = Path('/proc') / str(item['pid']) / 'cmdline'
    assert not cmdline.exists() or item['runtime'].encode() not in cmdline.read_bytes(), item
    if 'sampling-baseline' not in path.name:
        assert item['source_sha256'] == candidate['source_sha256']['js/impact-pack.js']
    servers.append({'profile': path.name, 'pid': item['pid'], 'owned_server_stopped': True})
assert len(servers) == 7
for path in sorted((P / 'initial-settings-race').glob('server-*.json')):
    item = json.loads(path.read_text())
    cmdline = Path('/proc') / str(item['pid']) / 'cmdline'
    assert not cmdline.exists() or item['runtime'].encode() not in cmdline.read_bytes(), item
    servers.append({'profile': 'initial-settings-race/' + path.name, 'pid': item['pid'], 'owned_server_stopped': True})
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError):
        continue
    assert not (b'--base-directory' in args and any(str(P / 'runtime-').encode() in value for value in args)), proc

comparison = json.loads((P / 'sampling/comparison.json').read_text())
overlay = json.loads((P / 'sampling/overlays/manifest.json').read_text())
assert comparison['successful'] and comparison['candidate_source_js_sha256'] == candidate['source_sha256']['js/impact-pack.js']
assert overlay['candidate_source_js_sha256'] == comparison['candidate_source_js_sha256']
assert len(overlay['images']) == 6
for image in overlay['images']:
    assert hashlib.sha256((P.parent / image['source']).read_bytes()).hexdigest() == image['source_sha256']

notes = '''# 남은 리뷰 지적 검토 및 보완

검토 기준은 기존 PR HEAD `6d1a488`이다. 전용 worktree에서 보완했고, 검증한 파일은 `candidate-source.json`, 최종 단일 커밋과의 대응은 `commit.json`에 기록한다.

| 지적 | 판정 및 반영 |
| --- | --- |
| 1. 중첩 방어 | 수용. 세 헬퍼의 try/catch와 API 접근의 중복 optional chaining을 제거하고 getWidgetFromSlot 어댑터의 한 경계에서만 선택적 바인딩 실패를 처리한다. 원본 코어 호출은 catch 밖에 둔다. 공유 정책은 mode 조회 전에 복원한다. 원본 오류 전달 테스트를 두 노드에 추가했다. |
| 2. 증적 헬퍼 테스트 분리 | note로 유지. 이 세 테스트는 앞선 감사에서 실제 발견한 원본 POST 유실 회귀를 막는다. 요청한 증적의 신뢰성과 직접 연결되어 있어 삭제하지 않는다. 사용자의 단일 PR 커밋 요청을 유지한다. 기존 Playwright 하네스를 import하지만 이 파일 자체는 Chrome이나 서버를 실행하지 않는다. 기본 standalone pytest 수집 대상에는 포함되지 않는다. Playwright가 필요하다는 범위는 README에 명시했다. |
| 3. 고정 화면 좌표 | 수용. 1400,450 빈 영역 클릭 대신 Escape로 제스처를 끝낸다. 기존 UI에서 읽은 네이티브 화살표 좌표를 한 번 클릭한다. 클릭 재시도나 app 평가로 위젯 값을 대입하지 않는다. 렌더 증빙 recorder의 동일한 고정 좌표도 제거했다. |
| 4. Node 버전 안내 | 수용. tests/README.md 상단에 전체 tests/*.test.mjs의 공통 Node 24 요구와 명령을 모으고 E2E 문서는 해당 절을 연결한다. 전역 Node 설정, 설치와 CI는 변경하지 않았다. |

현재 AGENTS.md에는 위젯/저장값 호환성과 카탈로그 절차가 명시되어 있다. 리뷰가 인용한 예외 처리 문구는 현재 파일에 없지만, 중복 방어를 줄이는 기술적 근거가 타당해 반영했다.

## 이번 소스로 직접 실행한 결과

- 같은 33개 seed 행동 검사를 기존 production 코드와 보완 후에 실행해 모두 통과했다. 동작을 보존하는 리팩터링 검증이며 새 버그의 Red/Green이라고 부르지 않는다. API 드리프트 12형태, 기존 공유 정책 복원, unreadable 승격 mode, 원본 코어 오류 전달을 포함한다.
- Node 24.12.0 전체 62건, Python 373건(205 + 168, 증적 검사 3건 포함) 통과. 이번 Node 실행을 과거의 60건 × 10회와 합치지 않는다.
- Chrome 집중 mode 전환 4건, 일반 캐시/튜토리얼 62건, 승격 classic 24건 + LRU 24건, 신뢰 카탈로그 14건/198 레코드가 최종 소스에서 모두 통과했다. 최종 행렬은 각 24건 전체를 실행했다. 클릭 재시도 루프나 조건부 재실행은 없다.
- 위젯 이름·순서·저장값을 native Export/load에서 검사했다. 신뢰 fixture 두 파일과 백엔드 변경은 없다.
- 최종 JS로 baseline/보완 각각 실제 모델 렌더 3회 수행했다. 보완 seed 513 유지 및 2·3회차 캐시 적중, 대조 seed 513/514/515, 독립 sampler 9001/9002/9003과 서로 다른 RGB 출력, 대응 독립 출력 및 첫 fixed 출력의 정확한 RGB 일치를 확인했다. 3회차 전 native 재로드도 검증했다.
- 원본 캡처 6개/출력 PNG 12개와 별도 박스 overlay 6개를 보존한다. 코어 투영 객체와 산술을 대체하지 않는다. 모든 소유 테스트 서버를 종료했다.

프론트엔드는 설치된 1.57.6에서 검증했다. 다른 프론트엔드 버전 전체를 보장한다고 주장하지 않는다. 독립 upstream GlobalSeed와 코어 promoted populate/before projection bookkeeping은 기존 한계로 남는다. 이전 감사의 RGR 및 광범위 SAM/모델 실행은 과거 소스의 자료로 보존하며 이번 실행 수에 합산하지 않는다.

## 준비 실패 및 진단 이력

최초 행렬에서 classic 22/24, LRU 23/24가 통과했고, 나머지 세 건은 노드 생성 전 Widget control mode 설정의 option 클릭에서 타임아웃됐다. 로그와 trace를 보면 검색 뒤 설정 패널이 교체되는 구간에서 드롭다운이 닫힌 상태였다. 검색 결과 제목을 기다린 뒤 열도록 수정하고 실제 선택값도 확인한다. 원본 로그/스크린샷/trace는 `initial-settings-race`에 보존했다. 처음 넣은 combobox 개수 검사는 검색 입력까지 세어 실패했으며 폐기했다. 이 중단된 진단은 `diagnostic-readiness-count`에 따로 보존하고 제품 회귀 또는 RGR의 Red로 세지 않는다. 보완 후에는 집중 mode 4건을 먼저 통과시킨 뒤 classic/LRU 각 24건과 일반 62건 전체를 새로 실행했다. 이 자료를 최초 실행부터 전부 성공한 것처럼 표현하지 않는다.
'''
(P / 'review-notes.md').write_text(notes)
(P / 'sampling/README.md').write_text('''# Three rendered queues per revision

Final source is mapped to the amended code commit in ../commit.json. Host mode is fixed and interior mode is populate. Each revision uses three actual Chrome Run queues; queue 3 follows native Export/reload. comparison.json records source hashes, real prompt IDs, cache events and RGB hashes. Original screenshots, twelve outputs and separate SVG/HTML overlays remain available. The recorder uses Escape instead of a hardcoded blank-canvas click. SD1.5 epicrealism + installed foodphoto LoRA, 512x512, deterministic CUDA; no assets downloaded.
''')

support = {name for name in candidate['source_sha256'] if Path(name).suffix != '.md'}
support.update(str(f.relative_to(R)) for f in (R / 'tests').glob('*.test.mjs'))
support.update(['js/impact-sam-editor.js', 'js/impact-wildcard.js', 'tests/pytest.ini',
                'tests/test_e2e_widget_catalog.py', *fixtures])
source = {}
for name in sorted(support):
    target = P / 'validated-source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R / name, target)
    source[name] = hashlib.sha256(target.read_bytes()).hexdigest()
(P / 'validated-source/README.md').write_text('''# Source-pinned focused tests

These files match the code commit in ../commit.json. From this directory, Node 24 with `node --experimental-vm-modules --test tests/*.test.mjs` runs all 62 Node tests. From tests/, `python -m pytest test_wildcard_cache_artifacts.py -v` runs the three POST artifact checks with pytest-timeout, Pillow and Playwright installed; it does not start Chrome or ComfyUI. This bounded source bundle is not a full ComfyUI installation. Exact commands and limits are in ../summary.json and ../review-notes.md.
''')
(P / 'source-bundle.json').write_text(json.dumps(source, indent=2) + '\n')
summary = {'completed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
           'validation_state': 'Matching working tree before amend; publication tracked separately',
           'parent_head': BEFORE, 'base': BASE,
           'source_sha256': candidate['source_sha256'],
           'runtime': {'node': '24.12.0', 'frontend': importlib.metadata.version('comfyui-frontend-package'),
                       'chrome': subprocess.check_output(['google-chrome', '--version'], text=True).strip(),
                       'python': sys.version.split()[0],
                       'comfy_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/mnt/teratera/git/ComfyUI', text=True).strip()},
           'results': results, 'python_passed': 373, 'node_passed': 62,
           'chrome_scenario_coverage': 124, 'chrome_case_executions_including_focused_check': 128,
           'fixture_sha256': fixtures, 'catalog_records': 198, 'servers': servers,
           'render_comparison': 'sampling/comparison.json', 'overlay_manifest': 'sampling/overlays/manifest.json',
           'notes': 'review-notes.md', 'refactor_equivalence': 'refactor-equivalence.json',
           'limitations': ['Independent upstream GlobalSeed may change fixed inputs.',
                           'Core promoted populate/before projection bookkeeping remains outside this fix.']}
(P / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Validation complete: Node 62, Python 373, Chrome 128 final executions / 124 scenario coverage, six real captures, unchanged fixtures and stopped owned servers.')
