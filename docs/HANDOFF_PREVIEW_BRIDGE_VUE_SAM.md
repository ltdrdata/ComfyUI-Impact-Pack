# Preview Bridge와 Vue MaskEditor SAM 확장 인계

작성일: 2026-11-01. 수신 대상: 진행 중인 Vue MaskEditor SAM 확장 작업 담당자.

Preview Bridge의 일반 마스크 편집은 Vue MaskEditor의 저장 흐름에 맞춰 이식했다.
SAM 기능은 사용자 설명에 따라 별도 작업에서 Vue MaskEditor를 확장한다.
다음 연결 지점은 **SAM 검출 결과를 Vue 편집기의 마스크 레이어에 반영하고,
기존 Save 경로로 저장한 뒤 Preview Bridge의 IMAGE·MASK·LATENT 출력에서 확인하는 것**이다.

## 작업 상태와 가져갈 파일

저장소는 `/mnt/teratera/git/ComfyUI/custom_nodes/comfyui-impact-pack`,
브랜치는 `fix/switch-workflow-compatibility`, 기준 HEAD는
`7c1fd27f3f5dbdc2b4adc1d9ee613cc53677a33e`이다.
아래 변경은 작성 시점에 **커밋되지 않은 working tree 변경**이다.
다른 checkout으로 옮길 때 새 파일도 함께 가져가야 한다.

| 파일 | 역할 |
| --- | --- |
| [js/preview-bridge.js](../js/preview-bridge.js) | 새 파일. Vue 위젯 유지, 실행 전 이미지 등록, 실행 결과 동기화, 기존 워크플로 복원 |
| [js/impact-image-util.js](../js/impact-image-util.js) | Bridge의 구형 setter·Clipspace 호출 스택 감지를 제거하고 새 모듈 연결 |
| [tests/preview_bridge.test.mjs](../tests/preview_bridge.test.mjs) | 새 파일. Vue store만 갱신되는 경우, 비동기 등록, 실패 후 재시도, 기존 ID 복원 검사 |
| [tests/test_e2e_preview_bridge.py](../tests/test_e2e_preview_bridge.py) | 새 파일. 실제 Chrome에서 편집·재로딩·출력·복원·SAM 검사 |
| [docs/E2E_TEST_STRATEGY.md](E2E_TEST_STRATEGY.md), [tests/README.md](../tests/README.md) | 실행 방법과 검증 범위 |

`js/impact-sam-editor.js`와 Python backend는 이번 이식에서 변경하지 않았다.
신규 노드 등록이나 기존 카탈로그 fixture 변경도 없다.

## Preview Bridge가 받는 데이터

Preview Bridge는 중간 이미지를 편집한 IMAGE와 MASK로 후단에 전달한다.
PreviewBridgeLatent는 latent samples를 유지하면서 편집한 마스크를 `noise_mask`와 MASK로 전달한다.
사용 의도는 튜토리얼의 반복 편집 및 후단 inpainting이다.

참고 원본은 `/mnt/teratera/git/ComfyUI-extension-tutorials/ComfyUI-Impact-Pack` 아래의
`tutorial/previewbridge.md`, `tutorial/sam_with_preview_bridge.md`,
`workflow/preview-bridge-image.png`, `workflow/preview-bridge-latent.png`이다.
PNG의 내장 워크플로에는 후단 결과를 Bridge로 다시 붙여 반복 편집하는 설명도 있다.

SAM 확장은 다음 저장 규약을 유지하면 현재 Bridge 연동을 이용할 수 있다.

1. 검출 결과를 Vue MaskEditor의 마스크 레이어에 반영한다. 편집기 마스크의 alpha는
   선택 강도이고, 저장 PNG의 alpha는 그 반대이다. Backend는 `MASK = 1 - alpha / 255`로 읽는다.
   마스크 아래의 RGB도 보존해야 한다. 기존 Vue Save 경로가 이 변환과 레이어 업로드를 담당한다.
2. Save 완료 후 native `image` 위젯에는 실제 업로드된 파일명을 유지한다.
   형식은 `subfolder/filename.png [input]`이며, subfolder가 없으면 생략한다.
   `[temp]`, `[output]`도 지원한다. 업로드 응답의 경로와 타입을 사용하고,
   `clipspace/`를 고정으로 붙이지 않는다.
3. 저장할 때마다 새 파일명을 사용한다. 현재 Bridge는 같은 파일명에 대한 등록 ID를 재사용한다.
   같은 경로를 덮어쓰는 별도 저장 방식은 등록 및 실행 캐시 갱신을 추가로 설계해야 한다.
   기존 Vue Save는 저장 시각을 포함한 새 파일명을 생성한다.
4. 이미지 출력 metadata와 preview도 새 파일을 가리켜야 한다. Vue의 기존 저장 경로를
   이용하면 위젯·출력·preview를 함께 갱신한다. 마스크 저장 중 IMAGE의 RGB나 paint 레이어를 잃지 않아야 한다.

현재 흐름은 다음과 같다.

```text
Vue MaskEditor Save
  → 파일 업로드와 native image 위젯·preview 갱신
  → Run 또는 API prompt 생성
  → image.serializeValue가 /impact/set/pb_id_image 등록 완료를 기다림
  → API prompt에 $node-id 형태의 서버 참조 전달
  → Bridge 실행 결과의 파일명을 다음 편집 입력으로 반영
```

Native workflow의 기존 `image` 위치에는 파일명이 저장되고, API prompt에만 서버 ID를 쓴다.
등록 실패 시 prompt 제출을 중단하고 오류를 표시한다. 이후 같은 편집 결과로 재시도할 수 있다.
기존 `$...` 값은 그대로 지원하며, 로드 시 `/impact/get/pb_id_image`로 preview를 복원한다.
등록되지 않은 기존 ID에서 파일을 복구하는 기능은 아니다.

이 모듈은 Vue 내부 store를 직접 import하지 않는다. native 위젯의 값 연결을 보존하므로
frontend가 store만 갱신하는 경우에도 동작한다. SAM 쪽에서 `image.value` accessor를 덮어쓰거나
저장값을 `$...`로 바꾸는 별도 처리를 추가하지 않는다.
`onNodeOutputsUpdated`는 MaskEditor 저장마다 호출되는 훅이 아니므로 저장 알림으로 가정하면 안 된다.

## SAM 작업에서 이어받을 부분

현재 [js/impact-sam-editor.js](../js/impact-sam-editor.js)는 구형 별도 대화상자이다.
`IMAGE` 또는 `MASK` 출력 노드에 `Open in SAM Detector` 메뉴를 추가하므로
두 Preview Bridge 모두 원래 대상이다.
Frontend 1.57.6에서는 `/extensions/core/clipspace.js` import가 404로 실패해 메뉴가 나타나지 않았다.
이는 실제 Detect 실행 이전의 로딩 실패이다.

진행 중인 Vue SAM 확장의 UI와 등록 API에 맞춰
`test_interactive_sam_saves_detected_mask_to_bridge`를 갱신해야 한다.
이 테스트는 현재 구형 메뉴, `#pointsCanvas`, `#samEditorMaskCanvas`, `Save to node`를 사용한다.
새 UI로 옮길 때 구형 메뉴를 복구하는 것을 완료 조건으로 삼지 않는다.

연결 후 확인할 흐름은 다음과 같다.

1. 실제 Preview Bridge를 실행하고 Vue MaskEditor 안에서 SAM 도구를 연다.
2. positive·negative 점을 지정하고 실제 검출을 실행한다. 검출 응답의 마스크를 독립적인 기대값으로 보관한다.
3. 검출 결과가 마스크 레이어에 표시되는지 확인하고, Vue Save로 저장한다.
4. Run 후 Bridge의 MASK와 기대값을 픽셀 단위로 비교한다.
   이미지 Bridge에서는 IMAGE가 보존되고, latent Bridge에서는 samples가 유지되며
   다음 Bridge에도 `noise_mask`가 전달되는지 확인한다.
5. 다시 열어 수동 마스크 편집과 SAM 재검출을 수행하고, 두 번째 저장이 첫 번째 결과를 대체하는지 확인한다.
   paint 레이어도 유지해야 한다.
6. Native Export·Import를 거쳐 같은 편집 결과를 복원하고, 위치 기준과 이름 기준으로
   위젯 이름·순서·값이 유지되는지 확인한다.

기존 E2E의 `edit_mask`, `editor_mask`, `editor_image`, `mask_output`, `latent_samples`,
`reload_workflow` 헬퍼와 latent 연결 사례를 재사용할 수 있다.
그래프 생성·연결·편집·Export·Import는 Chrome UI로 수행한다. JS는 상태와 픽셀을 읽는 용도이다.
SAM 테스트는 `IMPACT_E2E_SAM_IMAGE`가 없을 때만 건너뛴다.
검출 모델을 지정한 실행에서 메뉴나 모델이 없으면 실패로 남겨야 한다.

## 검증 결과와 남은 실패

| 대상 | 결과 |
| --- | --- |
| Frontend 1.57.6의 Bridge E2E | 24건 중 20 통과, 4 실패, skip 없음 |
| Frontend 1.57.6의 위젯 카탈로그와 보호 검사 | 12건 모두 통과 |
| Frontend 1.53.7의 반복 편집·native 재로딩·Clipspace 검사 | 5건 모두 통과 |
| Frontend 1.53.7의 동일 실제 UI RGR | 기존 코드 실패 → 이식 코드 통과 → 기존 코드 실패 |
| JS 단위 테스트 | 6건 모두 통과 |

RGR은 `test_edit_mask_repeat_and_native_reload[positional-PreviewBridge]`로 수행했다.
기존 코드는 편집 결과 대신 64×64 기본 마스크를 반환했고, 이식 코드는 256×256 편집 마스크와
반복 편집·native 재로딩 검사를 통과했다.

1.57.6에서는 기존 Clipspace 반복 편집 실패가 해결됐다. 이전 23건의 결과는 18 통과·5 실패였고,
현재는 페인트·마스크 재로딩 검사를 1건 추가했다. 등록 실패 검사는 잘못된 출력 실행을 허용하는 대신
prompt 제출이 차단되고 재시도가 성공하는지를 검사하도록 바꿨다.

남은 실패는 아래 네 건이다. 이번 이식으로 생긴 새 실패는 없다.

- `test_interactive_sam_saves_detected_mask_to_bridge`: 구형 SAM 모듈 로딩 실패.
  진행 중인 Vue SAM 확장에 맞춰 위의 새 UI 흐름으로 전환할 대상이다.
- `test_restored_latent_mask_reaches_the_next_bridge[same_size-always]`
- `test_restored_latent_mask_reaches_the_next_bridge[same_size-if_same_size]`
- `test_restored_latent_mask_reaches_the_next_bridge[new_size-always]`

마지막 세 건은 [modules/impact/bridge_nodes.py](../modules/impact/bridge_nodes.py)의 latent 갱신 분기에서
복원 마스크로 preview는 만들지만 `res_latent = latent`로 반환해 후단 `noise_mask`에 반영하지 않는 문제이다.
SAM UI 이식과 구분해 처리해야 한다.

PR #1236은 callback에서 setter를 실행하는 패치였다. 검증한 1.53.7의 저장 경로는 callback을 호출하지
않았고, 1.57.6은 이미 setter를 호출했다. 이번 변경은 native 위젯 연결을 유지하는 별도 구현이다.
#1236의 callback wrapper를 다시 붙이지 않는다. 해당 PR도 `tests/preview_bridge.test.mjs`를 추가하므로
합칠 경우 현재 6개 테스트와 파일 충돌을 확인해야 한다.

## 재검증 방법과 산출물

아래 명령은 저장소 루트에서 실행한다. 수정 중인 SAM 확장과 현재 Bridge 코드가 함께 설치된
격리 ComfyUI 서버를 먼저 준비한다. 카탈로그 검사는 Full Cache 설정이 필요하다.
자세한 서버 조건은 [E2E 테스트 전략](E2E_TEST_STRATEGY.md)을 따른다.

```bash
node --experimental-vm-modules --test tests/preview_bridge.test.mjs

IMPACT_E2E_BASE_URL=http://127.0.0.1:18140 \
IMPACT_E2E_ARTIFACTS=/tmp/impact-vue-sam-handoff-e2e \
IMPACT_E2E_SAM_IMAGE=/mnt/teratera/git/ComfyUI-extension-tutorials/ComfyUI-Impact-Pack/images/interactive-sam-original.png \
/home/rho/venv/bin/python -m pytest \
  tests/test_e2e_preview_bridge.py tests/test_e2e_widget_catalog.py -v --tb=short
```

SAM만 빠르게 확인하려면 같은 환경변수로 `tests/test_e2e_preview_bridge.py -k interactive_sam -v`를 실행한다.
기존 검증 서버는 `sam_vit_b_01ec64.pth`와 `sam_editor_cpu = True`로 구성했다.
새 확장이 요구하는 검출 backend와 모델 설정은 해당 작업의 설정에 맞춘다.

로컬 산출물은 `/tmp/impact-vue-bridge-glnyseed`에 있다. 저장소에 포함되지 않은 임시 파일이다.

- `validation.json`: 결과 요약, 남은 실패, 검증한 코드와 테스트의 SHA256.
- `full.xml`, `full.log`, `full/`: 1.57.6 전체 결과와 테스트별 screenshot·trace·workflow.
- `vue1537-final.xml`, `vue1537-final/`: 1.53.7의 5개 검사 결과.
- `rgr1537-{red,green,red-again}.xml`과 같은 이름의 디렉터리: 실제 UI RGR 결과.
- `server`, `server1537`: 당시 격리 서버 복사본. 사용한 18140·18141 서버 프로세스는 모두 종료했다.
  복사본을 다시 사용할 때 현재 변경 파일과 새 SAM 확장을 동기화해야 한다.

[AGENTS.md](../AGENTS.md)의 위젯 호환성 규칙을 유지한다.
`tests/workflows/all-impact-nodes.json`과 `all-impact-nodes.widgets.json`은 변경하지 않았다.
실패를 통과시키기 위해 이 기준 fixture를 재생성하거나 완화하지 않는다.
