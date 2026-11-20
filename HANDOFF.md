# 인수인계 (2026-11-20)

## 현재 상태

- Streamlit 웹 앱: 업로드 → VLM 분석 → 체크리스트 → 모자이크 렌더 → 다운로드
- live: `https://gong000000--mosaic.us-east.modal.run`
- 얼굴: YuNet + MediaPipe 478점 외곽, 추적/sticky box, 원거리 사람 크롭 보강
- 번호판: 전용 YOLO 검출 + 개별 트래커 + 차량 게이팅
- 주소·문서: VLM + EAST 글자 정제 + EAST-only 보완
  (`MOSAIC_ADDRESS_EAST=1` 기본)
- 간판: `signboard` 별도 카테고리. VLM이 간판을 찾은 프레임에서만 EAST 보완
- UI: 얼굴/번호판/텍스트·문서/간판 **체크박스 다중 선택**,
  전체 선택/해제. 차량동승자는 UI에서 제거(얼굴이 동승자 얼굴 포함)
- 렌더: SQLite 지속형 큐 + Streamlit과 분리된 워커 프로세스,
  계정별 대기 순번, 최대 2계정 동시 렌더, 작업별 카드/진행률/취소/재시도
- 계정/기록: `users.db`, `outputs/jobs.db`(SQLite), 결과 보관 24시간/500개/20GB
- 배포: Modal, `gpu=["L4", "T4"]` (L4 우선, 용량 없으면 T4)

## 검증 결과

- `pytest tests`: **54 passed**
- `scripts/apptest.py`: **APPTEST_PASS**
  - 얼굴, 번호판, 못 찾은 영역, **그룹 전환**, 2작업 등록/두 번째 취소
- `scripts/qualitytest.py`: **QUALITY_TEST_PASS**
- `scripts/robustness.py`: **ROBUSTNESS_PASS**
- live E2E (실제 Gemini)
  - face · 얼굴만: 분석 32.6초 / 렌더 19.2초
  - NY · 얼굴+번호판: 분석 21.4초 / 렌더 195.0초
  - live T4 얼굴/queue-cancel: PASS (그룹 UI 적용 전)
  - 새 그룹 UI: local AppTest + 실제 Gemini 프레임 검사로 확인
  - 2026-11-20 live 큐 복구: `12345678`과 `asd123` 작업이 동시에 running,
    진행률 증가 확인. `asd123`의 다음 작업은 계정 순번 2로 대기.
  - 같은 날 GPU preemption으로 컨테이너가 재시작된 뒤, 새 컨테이너에서
    running 작업을 queued로 복구하고 다시 처리하는 것을 확인.
- 카테고리 분리 확인
  - 합성 택배 송장: `address`로 검출, `signboard` 오검출 없음
  - street 프레임: `signboard` 검출, 문서 텍스트와 분리
- T4 전체 렌더 (`face.mp4` 72프레임, `yolo11m-seg`, 960):
  - **19.31초 / 3.73 fps / 피크 230MB**
- CI: `unit`, `lint`, `apptest` green
- `master` 브랜치 보호: `unit`, `lint`, `apptest` 필수, force-push/삭제 금지

## 큐 멈춤 원인과 수정 (2026-11-20)

원인:
- 워커가 Streamlit 프로세스 안의 스레드였고, 예외로 한 번 죽으면 다시
  뜨지 않았다.
- DB에 running 작업이 남으면 `claim()`이 전역으로 새 작업을 막아, 뒤 작업이
  계속 "대기 2번째/3번째"로 보였다.
- 브라우저 세션이 없거나 컨테이너가 scale-down되면 워커 자체가 없었다.
- Modal GPU 컨테이너는 preemption될 수 있다. preemption 후 브라우저 요청이
  없으면 워커가 다시 뜨지 않아 running 작업이 그대로 남았다.
- 대기 순번이 전 사용자 합계라 다른 아이디 작업이 내 순번에 섞였다.

수정:
- `src/worker.py`를 Streamlit과 별도 프로세스로 실행하고, supervisor가
  죽은 워커를 재시작한다.
- `queue_worker` heartbeat 리스로 중복 워커를 막고, 시작 시 running 작업을
  queued로 복구한다.
- 작업 중 self health 요청 + 5분 주기 `queue_keepalive` 함수로 컨테이너를
  유지/재기동한다.
- 15분 이상 진행률이 멈춘 작업은 워커 프로세스를 재시작해 고착을 푼다
  (`MOSAIC_JOB_STALE_SEC`).
- 순번은 계정별로 계산하고, 계정당 1개·서버 전체 최대 2개를 동시에
  처리한다.

## 운영 제약

- **GPU**: L4 우선, 없으면 T4 자동 대체. L4 전체 렌더 FPS/피크 VRAM은
  용량 부족으로 아직 실측하지 못했다. `scripts/bench_gpu.py`로 측정한다.
- **단일 컨테이너**: `max_containers=1` 전제. SQLite 큐와 heartbeat 리스는
  이 컨테이너 안에서 동작한다. 여러 컨테이너로 스케일 아웃하려면 외부
  DB/분산 락이 필요하다.
- **테스트 영상**: 실제 얼굴이 있어 저장소에 없다. `samples/` 또는
  `MOSAIC_TEST_FACE`/`MOSAIC_TEST_STREET` 환경변수로 지정한다.
- **보관**:
  - 결과: 24시간 + 500개 + 20GB 상한
  - 완료/취소 입력: 즉시 삭제
  - 실패 입력: 재시도용 24시간 보관 후 삭제
- **개인정보**: 원본은 `outputs/job_inputs/`에 임시 복사되며 작업 종료 시
  삭제된다. VLM에는 샘플 프레임 이미지만 전송하고 영상 전체는 보내지 않는다.

## 주요 파일

- `app.py`: Streamlit UI, 작업 카드/취소/다운로드
- `src/pipeline.py`: 2패스 렌더, 얼굴/번호판/영역 처리
- `src/jobqueue.py`: SQLite 큐, 계정별 스케줄러, 리스/heartbeat, 취소/재시도
- `src/worker.py`: Streamlit과 분리된 큐 워커 프로세스/감시
- `src/faces.py`, `src/face_mesh.py`: 얼굴 검출/추적/외곽
- `src/plates.py`, `src/texts.py`: 번호판, EAST 주소/간판
- `src/history.py`, `src/retention.py`: 기록/보관 정책
- `scripts/apptest.py`, `scripts/bench_gpu.py`: E2E/GPU 측정
- `modal_app.py`: Modal 배포 설정
