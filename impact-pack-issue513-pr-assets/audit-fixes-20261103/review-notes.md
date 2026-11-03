# 코드·테스트 감사 보완과 재검증

게시된 HEAD `3452e43` 위의 로컬 미커밋 변경이다. 전용 worktree만 수정했으며 commit, push, 원격 PR 수정, 외부 이슈 등록은 하지 않았다.

| 항목 | 보완 | 반증·재검증 |
| --- | --- | --- |
| F1: 보류 정책 복원 누락 | seed의 원본 control을 먼저 복원한 다음 mode를 조회한다. 다른 host가 mode 조회에 실패해도 직전 host의 fixed를 읽지 않는다. | 기존 감사 반례 포함 4건 통과. 새 테스트는 decrement/randomize를 사용해 increment 상수 복원도 구분한다. |
| F2: 읽을 수 없는 승격 mode를 내부 fixed로 대체 | 승격 input이 없는 경우와 input은 있으나 값을 읽을 수 없는 경우를 구분한다. 후자는 override를 포기한다. | 기존 코드에서 새 단위 검사 4건 실패 → 로컬 31건 통과 → 기존 코드에서 같은 4건 실패. |
| F3: 승격 경로를 타지 않는 오류 테스트 | 기존 중복 2건을 실제 정책 보류 후의 실패 경로 4건으로 교체했다. | 승격 어댑터를 통째로 끈 변형은 이제 4건이 실패한다. |
| F4: host/interior mode가 같은 E2E | host=fixed, interior=populate로 값을 구분하고 native positional/named 저장값에도 이를 단언한다. | classic·LRU의 해당 4건이 모두 통과. host mode를 무시하는 변형은 after timing의 두 노드에서 seed가 5→6이 되어 실패 → 정상 코드 2건 통과 → 같은 변형 2건 실패. |
| F5: 출력 실패 시 POST 증적 누락 | UI 실행 후 원본 POST를 먼저 저장하고 공통 출력/preview 검사를 호출한다. | 출력 불일치, preview 단언 실패, 정규화된 history와 원본 seed 구분을 검사한다. RGR은 2 실패/1 통과 → 3 통과 → 같은 2 실패/1 통과. |
| F6: 동일 경로 10회 반복 | 일반 행렬은 3회로 줄였다. native 재로드, 위젯 이름/순서/값, 수동 seed 변경 시 cache miss 검사는 유지했다. | 양쪽 캐시 24개 시나리오 각각 검증. 정규 행렬의 fixed 제출은 480→144회다. |

## 결과 범위

- Node 24.12.0에서 전체 60건을 10회 연속 실행해 모두 통과했다. 이 비교는 실패할 때만 재실행하는 retry가 아니며 모든 호출 결과를 보존했다.
- Python 373건 통과: wildcard 205 + 기타/하네스 168(신규 증적 검사 3건 포함).
- Chrome 일반 캐시/튜토리얼 62건, 신뢰 카탈로그 14건/198 레코드 통과.
- 승격 행렬은 캐시마다 변경되지 않은 20건과 수정한 mode 4건으로 24개 시나리오를 검증했다. 첫 행렬의 팝업 타임아웃은 원본을 보존했고, 영향을 받은 4건만 각 캐시에서 단일 네이티브 화살표 조작으로 다시 실행했다. 클릭 재시도 루프나 app 평가를 통한 위젯 대입은 없다. 이를 하나의 최초 24/24 all-green 원시 실행으로 표현하지 않는다.
- core before-mode의 projection/첫 큐 문제 때문에 mode 무시 변형의 반증은 after timing으로 분리했다. 최종 Red는 모두 popup/준비 오류가 아닌 실제 seed 6 대 기대값 5의 단언 실패다. 해당 코어 문제를 이 PR에서 우회하지 않는다.
- 최종 production JS로 실제 모델 렌더를 전후 3회씩 새로 수행했다. host=fixed/interior=populate 상태를 모든 큐에서 확인했다. 패치 seed 513 유지, 2·3회차 캐시 적중, 독립 sampler 9001/9002/9003 진행, 서로 다른 독립 출력 3개, 대응 독립 출력 및 최초 fixed 출력의 전후 정확한 RGB 일치를 확인했다. 3회차 전 native 재로드도 검증했다.
- 두 신뢰 fixture와 백엔드는 변경하지 않았다. 소유 테스트 서버는 모두 종료했다.

## 런타임과 준비 오류

[Node 문서](https://github.com/nodejs/node/blob/main/doc/api/vm.md#class-vmmodule)는 VM 모듈을 experimental로 분류한다. 로컬 Node 20.19.5에서는 전체 10회 중 1회가 SIGSEGV로 중단됐고, Node 24.12.0의 동일 소스 10회는 모두 60건을 통과했다. Node 20 크래시의 근본 원인을 수정했다고 주장하지 않는다. 검증 런타임을 Node 24로 명시했으며 전역 Node 설정이나 설치를 바꾸지 않았다. `node-runtime-comparison.json`에 모든 결과가 있다. Node 24의 기본 reporter가 TAP과 달라 첫 카운터가 0을 표시했던 집계 오류는 exit code와 60 pass 원문으로 확인해 바로잡았다.

초기 mode-index 단언에서 Encode의 wildcard picker 위치를 mode로 잘못 읽은 오류, popup-based mode 전환 타임아웃, Node 20 첫 RGR 중단, JavaScript를 Python 린트에 잘못 포함한 호출은 진단 원본으로 남겼다. 최종 린트·문법 검사와 수정 범위의 성공 결과에 이 실패를 합산하지 않았다. 과거 전체 모델/SAM 회귀를 현재 코드로 새로 실행했다고 주장하지 않는다.

원본 POST 보존 강화의 범위는 성공한 UI 실행 뒤 output/preview 단언 실패다. 제출 또는 backend 실행 자체가 더 일찍 실패하는 모든 경로의 증적 보존까지 검증했다고 주장하지 않는다.

상세 결과/명령은 `summary.json`, 소스 해시는 `candidate-source.json`, 보완 diff는 `review-fixes.patch`와 `pr-candidate.patch`에 있다.
