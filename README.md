# 영상 속 개인정보 보호 모자이크

영상에서 개인정보(얼굴, 차량번호판, 주소·문서 텍스트, 간판, 명찰, 신분증,
휴대폰 화면)를 찾아 **목록으로 보여주고**, 사용자가 고른 항목만 실제
모자이크 영상으로 렌더링하는 Streamlit 앱입니다. GPU 서버(Modal)에 배포되어
설치 없이 웹 브라우저로 사용합니다.

## 웹에서 사용

- 서비스 주소는 `https://gong000000--mosaic.us-east.modal.run` 입니다.
  (`modal_app.py`의 `label="mosaic"`로 지정한 값이 URL에 쓰입니다.
  Modal 대시보드에서도 확인할 수 있습니다.)
- 주소만 열면 되고, 사용자 PC에 설치할 것은 없습니다.
- 첫 접속은 서버를 깨우느라 10~30초 걸릴 수 있습니다.
- 회원가입 → 로그인 → 영상 업로드 순서로 사용합니다.
- 사용 후 브라우저 탭을 닫으면 약 5분 뒤 GPU가 자동으로 꺼집니다.
- 계정과 결과는 서버 볼륨에 저장되어 재배포해도 유지됩니다.
- 서비스 주소는 팀 내부에만 공유하세요. `MOSAIC_INVITE_CODE`를 Secret에
  설정하면 초대 코드를 아는 사람만 가입할 수 있습니다. 설정하지 않으면
  링크를 아는 사람이 가입해 크레딧을 사용할 수 있습니다.

## 동작 흐름

0. **회원가입 · 로그인**
   - 앱을 열면 먼저 로그인 화면이 나옵니다. 처음이면 `회원가입` 탭에서
     아이디(영문/숫자/_ 3~20자)와 비밀번호(8자 이상)를 등록합니다.
   - 계정은 서버의 `users.db`(SQLite)에 저장되며, 비밀번호는
     PBKDF2-SHA256 + 사용자별 salt 해시로만 저장됩니다.
   - 로그인한 아이디별로 결과 파일과 `지난 작업` 기록이 분리됩니다.

1. **업로드 → VLM 분석**
   - 균등 8장에 더해 **장면 전환 직후**, **컷 직전**, **움직임 피크** 프레임을
     섞어 최대 20장을 Gemini VLM에 보내 개인정보 후보와 위치를 찾습니다.
     6개 워커로 병렬 호출하되 분당 90회 상한을 둡니다.
   - 화면에는 자동 모자이크가 아니라 **체크리스트**만 표시됩니다.
   - 분석은 항상 모든 개인정보 후보를 찾습니다(프라이버시 우선). 사이드바의
     그룹 체크박스는 분석 후 어떤 항목을 자동으로 체크할지만 정합니다.
   - 분석 전 원본 프레임 일부가 Gemini로 전송된다는 고지를 업로드 화면에
     표시합니다. 원본 영상 파일과 오디오는 전송하지 않습니다.

2. **가릴 항목 선택**
   - 그룹 체크박스: `얼굴`, `번호판`, `텍스트·문서`, `간판`. 여러 개를 동시에
     선택할 수 있고, `전체 선택`/`전체 해제` 버튼도 있습니다.
   - `텍스트·문서`: 주소·택배 송장·신분증·영수증·휴대폰 화면·명찰
   - `간판`: 매장·건물·광고판·도로 표지판
   - 그룹을 바꾸면 체크리스트 기본 선택이 즉시 바뀌고 VLM을 다시 부르지 않습니다.
   - `요청한 항목`과 `추가로 찾은 개인정보`를 나눠 보여줍니다.
   - 그룹이 요구했지만 샘플에서 못 찾은 항목은 `못 찾음`으로 표시됩니다.
   - 얼굴은 선택하면 프레임마다 로컬 얼굴 검출로 적용해 누락을 줄입니다.

3. **모자이크 시작**
   - 버튼을 눌러야만 전체 영상을 렌더링합니다. 자동 렌더는 하지 않습니다.
   - 같은 설정의 이전 결과가 있어도 새로 렌더링합니다.
   - 렌더 단계에서는 VLM을 다시 호출하지 않습니다. 분석 priors,
     YOLO11-seg/ByteTrack 추적, 얼굴 전용 경로의 OpenCV YuNet 검출을 사용합니다.
   - 얼굴만 선택하면 얼굴형 폴리곤으로 가려 목·상체·손으로 퍼지지 않습니다.
     눈 사이 기울기로 고개 각도를 따라가고, 검출이 끊긴 프레임은 광학 흐름과
     속도 예측으로 이어서 가립니다. YuNet이 놓친 프로필·고개 숙인 얼굴은
     포즈 키포인트로 보완하고, **옆으로 누운 얼굴은 프레임을 90/270도
     회전해 다시 검출**합니다. 트래킹은 검출이 끊겨도 약 0.8초 유지합니다.
   - `MOSAIC_FACE_MESH=1`(기본값)이면 MediaPipe 랜드마크 478점의 외곽으로
     사람마다 다른 얼굴 실루엣을 그립니다. 단, 랜드마크 외곽이 이마·머리카락을
     충분히 덮지 못하면 **여백을 넓힌 박스형 안전 마스크**로 자동 대체합니다.
     모델이 없거나 실패해도 박스형으로 대체합니다. 끄려면 0으로 하세요(빠름).
   - 렌더가 끝나면 입력/출력의 같은 프레임을 비교해 **얼굴 마스크 누락을
     자동 점검**합니다. 작업 카드에 `얼굴 마스크 점검: 348/348`처럼 표시하고,
     고신뢰 누락 프레임은 강한 사각형 모자이크로 자동 보정합니다
     (`MOSAIC_FACE_AUDIT`, 기본 켜짐).
   - 상반신/전체 모드는 1패스에서 사람 박스·마스크 윤곽을 기록하고 2패스에서
     재생하므로 트랙 ID가 바뀌지 않습니다. 검출이 끊긴 짧은 구간은
     앞뒤 박스로 보간합니다.
   - 번호판·간판 박스는 VLM 앵커 사이를 광학 흐름으로 이동시키고, 이동 거리에
     비례한 여유를 붙여 샘플 사이에서도 빠지지 않게 합니다. 마지막 앵커에서
     너무 오래 지나면 고스트를 남기지 않도록 렌더를 멈춥니다.
   - **번호판 전용 검출기**: YOLO11n 번호판 모델이 2프레임마다 직접 찾아
     광학 흐름으로 이어 붙입니다. VLM 앵커는 백업으로 유지합니다.
     `MOSAIC_PLATE_DET=0`이면 VLM만 씁니다.
   - `veryfast` H.264 MP4로 인코딩하며 원본 오디오와 영상 길이를 유지합니다.
     사이드바의 `오디오 제거`를 켜면 목소리도 제거합니다.
   - 렌더 작업은 SQLite 큐(`render_jobs`)에 저장되고, Streamlit과 **별도
     프로세스**로 실행되는 워커가 처리합니다. 브라우저 창을 닫거나 접속이
     끊겨도 작업이 계속되고, 워커가 죽으면 자동으로 다시 뜹니다. 서버가
     재시작되거나 Modal GPU가 preemption되면 남아 있던 running 작업을
     queued로 되돌려 다시 처리합니다. 5분마다 도는 keepalive 함수가 큐를
     확인해 컨테이너를 깨우므로, 아무도 접속하지 않아도 작업이 영원히
     멈춰 있지 않습니다.
     입력 영상은 볼륨의 `job_inputs/`에 복사한 뒤 완료/실패/취소 시
     삭제합니다.
   - 대기 순번은 **계정별**로 계산합니다. 같은 아이디의 앞선 작업만 순번에
     포함되므로 다른 아이디의 작업 때문에 내 순번이 밀리지 않습니다.
     같은 계정 안에서는 작업이 등록 순서대로 하나씩 실행되고, 서버 전체로는
     최대 5개 계정 작업을 동시에 처리합니다(`MOSAIC_MAX_PARALLEL`).
     진행률이 15분 이상 갱신되지 않는 작업은 워커를 재시작해 고착을
     풉니다(`MOSAIC_JOB_STALE_SEC`).
   - 긴 영상은 해상도에 따라 자동으로 키프레임 단위 구간을 나눠
     렌더링하고, 끝난 구간은 체크포인트에 저장합니다. preemption/재시작 시
     완료 구간을 건너뛰고 남은 구간부터 이어서 렌더합니다
     (`MOSAIC_CHUNK_SEC`로 구간 길이 지정, `0`이면 끔).
   - `모자이크 시작`은 작업을 큐에 넣고 즉시 돌아옵니다. 여러 번 누르면
     `작업 진행 · 결과`에 1번째/2번째/3번째 카드가 생기고, 각 카드에서 진행률을
     보거나 **취소**할 수 있습니다. 완료된 카드 아래에 미리보기와 다운로드가
     표시됩니다.
   - 실패한 작업은 입력 영상을 24시간 보관하고 카드의 **다시 시도** 버튼으로
     같은 설정을 다시 큐에 넣을 수 있습니다.

4. **미리보기 · 다운로드**
   - 사진이 아니라 **모자이크가 실제로 적용된 동영상**을 재생해 확인합니다.
   - 결과 파일명은 `아이디--원본이름_시각_토큰.mp4` 형식이라 계정별로
     구분됩니다. `지난 작업`은 로그인 직후 업로드 전에도 볼 수 있고,
     현재 로그인한 아이디의 결과만 표시됩니다. 진행 중 작업은 대기 순번/
     진행률로 함께 표시됩니다.
   - 확인 후 다운로드합니다. `받으면 결과 자동 삭제`를 켜면 다운로드 시
     서버 결과 파일과 기록을 지웁니다.

## 고정 설정

속도 모드/샘플 수/모델 선택 UI는 없습니다. 항상 아래 설정으로 동작합니다.

| 설정 | 값 |
|---|---|
| 사람 검출/추적 | `yolo11m-seg.pt` (테스트만 `yolo11n-seg.pt`) |
| 추론 해상도 | `imgsz=960` |
| 분석 샘플 수 | 균등 8장 + 최대 12장(컷·컷 직전·움직임 피크) = 최대 20장 |
| VLM 병렬 워커 | 6 |
| VLM 분당 호출 상한 | 90회 (`MOSAIC_VLM_RPM`으로 조절) |
| 번호판 확대 탐지 | 영상당 최대 28회 (프레임당 최대 2회) |
| 키프레임 간격 | `15` |
| 영상 최대 길이 | `MOSAIC_MAX_SECONDS` 기본 900초 |
| 인코딩 프리셋 | `veryfast` |
| Ultra 모드 | 사용 안 함 |

무거운 `yolo11m-seg.pt`/`960px`는 L4 GPU에서 충분히 빠르고, 사람 마스크
경계가 `yolo11n`/`640px`보다 훨씬 정밀합니다. 가중치는 Modal 이미지 빌드
때 미리 내려받아 첫 요청 지연을 줄입니다. 로컬 테스트는
`MOSAIC_MODEL=yolo11n-seg.pt`, `MOSAIC_IMGSZ=640`으로 낮춰 실행합니다.

## 샘플링 (분석 정확도)

분석에 보낼 프레임은 균등 배치만 쓰지 않습니다(`src/sampling.py`).

| 종류 | 개수 | 목적 |
|---|---|---|
| 균등 기본 샘플 | 6 | 영상 전체를 고르게 훑는다 |
| 장면 전환 샘플 | 최대 4 | 화면이 크게 바뀐 지점 직후 |
| 컷 직전 샘플 | 최대 2 | 전환 순간 사라지는 물체를 잡는다 |
| 움직임 피크 샘플 | 최대 4 | 컷 없는 CCTV에서 사람이 스치는 지점 |

- 영상 전체를 저해상도(64×36) 서명으로 최대 900프레임까지 훑어 움직임
  변화량을 구합니다.
- 컷/피크 임계값은 영상마다 중앙값·MAD로 적응적으로 정합니다. 카메라
  흔들림을 컷으로 오탐하지 않기 위한 것입니다.
- 10초 영상에서 2.2초에만 스친 신분증처럼, 균등 샘플이 놓치는 구간을
  잡아냅니다. 총 호출 수는 16장으로 제한해 시간과 비용을 예측 가능하게
  유지합니다.
- 6개 워커로 병렬 호출하되 분당 90회 상한을 두어 429를 예방합니다.

## 모자이크 품질

| 항목 | 처리 |
|---|---|
| 모자이크 강도 | 슬라이더 강도가 높을수록 블록이 커집니다. 기본 30에서 짧은 축 약 5칸으로, 모자이크 후에도 눈·코·입이 재검출되지 않는 크기를 실제 얼굴 영상으로 확인했습니다 |
| 사람 마스크 | YOLO `masks.xy` 폴리곤을 원본 좌표에 직접 래스터화하고 1px 안쪽 침식. 바깥 팽창·바깥 블러를 제거해 사람 밖 배경으로 번지지 않습니다 |
| 마스크 시간 안정화 | 직전 프레임과 75:25로 섞어 경계 떨림만 줄이고, 바깥으로 커지지 않게 임계값을 둡니다 |
| 얼굴 | YuNet 타이트 박스를 넓힌 뒤 MediaPipe 랜드마크 외곽(사람마다 다름)으로 가림. 랜드마크가 이마·머리를 충분히 덮지 못하면 여백을 키운 박스형 안전 마스크로 대체한다 |
| 얼굴 누락 | 다중 스케일 YuNet + **90/270도 회전 재검출** + 포즈 키포인트 머리 보완 + 3프레임 간격 재검출. 검출이 끊기면 광학 흐름/속도 예측으로 약 0.8초 유지 |
| 근접 얼굴 | 화면의 50%까지 허용하되, 큰 박스는 높은 신뢰도에서만 사용 |
| 번호판 | YOLO11n 번호판 모델이 2프레임마다 직접 검출하고 광학 흐름으로 잇는다. 여백을 물체 크기 비례(최대 12px)로 주고, 차량 안에 있을 때만 그려 벽보 오검출을 줄인다. VLM 앵커는 백업으로 유지한다. |
| 영역 박스 | VLM이 넓게 잡은 주소·문서 박스를 EAST 글자 영역에 맞춰 조인다. 텍스트·문서 그룹에서는 사람/차량/번호판 내부를 제외한 EAST-only 글자 영역도 추가하고, 간판 그룹은 VLM이 간판을 하나라도 찾은 프레임에서만 큰 글자 영역을 보완한다(`MOSAIC_ADDRESS_EAST=1`). 번호판 박스는 에지 영역에 맞춰 조임(최소 60% 유지). 글자가 안 보이면 VLM 박스를 유지한다. |
| 시간 보간 | 샘플 간격이 길어져도 최대 180프레임까지 보간하고, 이동 거리에 비례해 박스를 넓혀 누락 방지 |

## 로그인 · 계정

- 앱을 열면 로그인 화면이 먼저 표시됩니다. `회원가입` 탭에서 계정을 만든 뒤
  `로그인` 탭에서 접속합니다.
- 아이디는 영문/숫자/_ 3~20자, 비밀번호는 8자 이상입니다.
- 계정은 서버 볼륨의 `users.db`(SQLite)에 자동 생성·저장됩니다.
- 비밀번호는 PBKDF2-SHA256(20만 회) + 사용자별 salt 해시로 저장되고
  평문으로 저장하지 않습니다.
- 5회 연속 실패하면 60초 동안 로그인이 잠깁니다.
- 로그인 후 사이드바에서 현재 아이디를 확인하고 `로그아웃`할 수 있습니다.
- 결과 파일은 `아이디--원본이름_시각_토큰.mp4`로 저장되며, `지난 작업`은
  로그인만 하면 볼 수 있고 현재 아이디로 시작하는 파일만 보여줍니다.

## 운영 · 재배포 (Modal, GPU)

가입 후 매월 $30 크레딧이 제공되고, 접속이 없으면 컨테이너가 자동으로
내려가 GPU 비용이 계속 나가지 않습니다. 단, 렌더 작업이 도는 동안에는
컨테이너를 유지합니다.

최초 1회(배포 PC에서):

```powershell
pip install modal python-dotenv
modal setup
```

배포·재배포:

```powershell
modal deploy modal_app.py
```

- 첫 배포는 CUDA용 torch 이미지를 빌드하므로 10분 정도 걸릴 수 있습니다.
- `users.db`와 `outputs/`는 Modal Volume `mosaic-data`에 저장됩니다.
  컨테이너가 재시작·재배포돼도 계정과 결과가 유지됩니다.
- 렌더 워커는 Streamlit과 별도 프로세스(`src/worker.py`)로 뜨고, SQLite
  heartbeat 리스로 중복 실행을 막습니다. 작업 중에는 서버가 스스로 health
  요청을 보내고, 5분 주기 `queue_keepalive` 함수가 큐를 확인해 컨테이너를
  깨웁니다. 브라우저를 닫아도, GPU가 preemption돼도 렌더가 다시 이어집니다.
- `MOSAIC_MAX_PARALLEL`(기본 3)은 동시에 렌더할 수 있는 **계정 수**입니다.
  같은 계정의 작업은 등록 순서대로 하나씩 처리됩니다. 3개를 넘는 계정은
  빈 슬롯이 생길 때까지 `대기 중`으로 표시됩니다.
  - T4 · 1080p · 12초 얼굴-only 측정: 1개 157초(2.29fps),
    2개 244초(합계 2.95fps), 3개 355초(합계 3.04fps),
    5개 661초(합계 2.71fps). 3개가 총 처리량이 가장 높고, 5개는
    개별 작업이 약 4배 느려져 기본값을 3으로 둡니다.
- `MOSAIC_JOB_STALE_SEC`(기본 900초) 동안 진행률이 갱신되지 않으면 워커
  프로세스를 재시작해 멈춘 작업을 정리합니다.
- `MOSAIC_CHUNK_SEC`는 긴 영상을 나눠 렌더링하는 구간 길이입니다. 미설정
  시 해상도별 자동(4K 20초, 1440p 30초, 1080p 45초, 720p 60초, 그 이하
  90초)이고, `0`이면 구간 분할/체크포인트를 끕니다. 끝난 구간은 결과 옆
  `.chunks/`에 저장되며, 성공 시 자동 삭제됩니다.
- `MOSAIC_FACE_ROTATE`(기본 1)는 옆으로 누운 얼굴을 90/270도 회전해 다시
  검출합니다. `0`이면 끕니다. `MOSAIC_FACE_ROTATE_EVERY`(기본 3)는 회전
  재검출을 수행하는 프레임 간격입니다.
- `MOSAIC_FACE_AUDIT`(기본 1)는 렌더 후 얼굴 마스크 누락을 자동 점검합니다.
  `MOSAIC_FACE_AUDIT_SAMPLES`(기본 120)는 점검 프레임 수,
  `MOSAIC_FACE_AUDIT_DIFF`(기본 4.0)는 입력/출력 차이 임계값,
  `MOSAIC_FACE_AUDIT_FIX`(기본 1)는 고신뢰 누락 프레임 자동 보정입니다.
- `MOSAIC_ADMIN_USERS`에 쉼표로 구분한 아이디를 넣으면 관리자 계정 상단에
  `🛠 관리자 · 전체 작업/강퇴` 패널이 표시됩니다. 모든 계정의 렌더/대기
  작업 목록과 개별·전체 취소, 계정별 강퇴, 30분 사이트 잠금(비관리자
  로그인/가입 차단)을 할 수 있습니다. 관리자 본인은 강퇴/잠금에서 제외됩니다.
- `GEMINI_API_KEY`는 Modal Named Secret `mosaic-gemini`에서 읽습니다.
  `.env`를 들고 있는 PC 말고도 **Modal 로그인만 되어 있으면 배포할 수
  있습니다.** `.env`는 이미지에 포함되지 않고 커밋되지 않습니다.
  모델명은 Secret의 `VLM_MODEL`로 바꿉니다(기본 `gemini-3-flash-preview`).
- 기본 GPU는 L4(8 CPU, 16GB RAM)이고, L4 용량이 없으면 T4로 자동 대체됩니다.
  `modal_app.py`의 `gpu=`, `cpu=`, `memory=`에서 바꿉니다.
  L4는 `yolo11m-seg`/960 추론 약 23ms, T4는 약 55ms입니다.
- Modal 리버스 프록시에서 파일 업로드가 403으로 막히지 않도록 Streamlit
  CORS/XSRF는 기본적으로 꺼져 있습니다. 사내 프록시 없이 직접 노출하는
  환경이라면 `MOSAIC_ENABLE_XSRF=1`로 켤 수 있습니다.
- 가입을 제한하려면 Secret에 `MOSAIC_INVITE_CODE`를 추가하세요.
  `modal secret create mosaic-gemini --from-dotenv .env --force`로 갱신하면
  같은 Secret에 함께 들어갑니다.
- 로그: `modal app logs mosaic-web --tail 100` (계속 보려면 `-f`)
- 중지: `modal app stop mosaic-web`
- 탭을 열어둔 채 방치하면 GPU가 계속 과금됩니다.

### 서비스 주소 바꾸기

Modal URL은 `https://<워크스페이스>--<label>.<리전>.modal.run` 형태입니다.
`<label>`은 `modal_app.py`의 `modal.web_server(..., label="...")`로 정합니다.

```python
@modal.web_server(PORT, startup_timeout=600, label="mosaic")
```

`label`을 바꾸면 **URL도 바뀝니다.** 이전 주소는 404가 되므로, 공유 사이트에
올린 링크가 있다면 함께 고쳐야 합니다. 바꾸고 나서 `modal deploy modal_app.py`
로 다시 배포해야 적용됩니다.

Modal 자체의 커스텀 도메인(`https://app.내도메인.com`)은 Team 플랜
($250/월) 전용 기능입니다. Starter 플랜에서는 `modal.run` 주소만 쓸 수
있습니다.

### Named Secret 만들고 갱신하기

Secret은 Modal 워크스페이스에 있으므로 한 번 만들면 모든 배포 PC가
공유합니다. 값은 화면에 출력되지 않습니다.

최초 1회(키가 있는 PC에서):

```powershell
modal secret create mosaic-gemini --from-dotenv .env
```

키를 바꾸었을 때 갱신:

```powershell
modal secret create mosaic-gemini --from-dotenv .env --force
```

확인(키 값은 안 보입니다):

```powershell
modal secret list
```

### 다른 컴퓨터에서 배포하기

`.env` 없이 가능합니다. 같은 Modal 계정(`gong000000`)으로 로그인해야
기존 URL과 `mosaic-data` 볼륨을 그대로 씁니다.

```powershell
git clone https://github.com/gong000000/mosaic-web.git
cd mosaic-web
pip install modal python-dotenv
modal setup          # gong000000 워크스페이스 계정으로 로그인
modal deploy modal_app.py
```

- 다른 Modal 계정으로 배포하면 URL이 바뀌고 기존 계정·결과를 볼 수 없습니다.
- `modal deploy`는 앱을 다시 켜는 동작입니다. 링크를 공개하고 싶지 않으면
  배포하지 말고 `modal app stop mosaic-web -y`로 중지 상태를 유지합니다.

### 로컬에서 실행 (개발용, 선택)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-gpu.txt   # CPU만 쓸 때는 requirements.txt
python -m streamlit run app.py
```

`http://127.0.0.1:8501`에서 확인합니다. `.env`의 키를 그대로 사용합니다.

## 테스트

로컬 환경에서 VLM 스텁으로 키 없이 실행할 수 있습니다.

```powershell
python -m pytest tests                    # 핵심 25개, 모델/영상 없이 실행
python scripts/smoke.py
python scripts/authtest.py
python scripts/checklist.py
python scripts/qualitytest.py             # 마스크/강도/보간 회귀
python scripts/sampletest.py
python scripts/freeform.py
python scripts/regression.py              # 샘플 영상 필요
python scripts/robustness.py
python scripts/apptest.py                 # 샘플 영상 필요(없으면 make_ci_video.py)
python scripts/make_ci_video.py /tmp/mosaic_ci.mp4
python scripts/live_apptest.py /data/_smoke/face.mp4  # 배포 컨테이너 안에서 실행
python scripts/golden_run.py --out golden.json
python scripts/leak_metrics.py 결과.mp4   # YuNet 재검출 대리 지표
```

- `authtest.py`는 임시 DB로 회원가입/로그인/비밀번호 해시 저장과 로그인
  화면 노출을 확인합니다(실제 `users.db`는 건드리지 않습니다).
- `MOSAIC_MODEL`, `MOSAIC_IMGSZ`를 지정하지 않은 로컬 테스트는
  `yolo11m-seg.pt`/960을 사용하므로 느릴 수 있습니다. 빠른 검사는
  `MOSAIC_MODEL=yolo11n-seg.pt`, `MOSAIC_IMGSZ=640`을 지정하세요.
- `apptest.py`는 로그인 세션을 주입한 뒤 업로드 → 분석 목록 →
  `모자이크 시작` → 결과 파일 생성 → 다운로드 버튼 노출까지 확인합니다.
  실제 영상이 없으면 `scripts/make_ci_video.py`로 합성 MP4를 만들 수 있습니다.
- `qualitytest.py`는 마스크가 사람 밖으로 번지지 않는지, 강도가 높을수록
  블록이 커지는지, 기울어진 얼굴·긴 샘플 간격 보간이 유지되는지 봅니다.
- `address_eval.py`는 `tests/fixtures/address_predictions.json`(VLM) 또는
  `address_predictions_east.json`(VLM+EAST-only)과 `address_labels.json`으로
  주소·문서/간판 precision/recall(EAST reference)을 계산합니다. 로컬 테스트
  영상 경로는 `--video-dir`로 넘깁니다.
- 테스트 영상은 `MOSAIC_TEST_FACE`, `MOSAIC_TEST_STREET` 환경변수나
  `samples/face.mp4`, `samples/street.mp4`로 지정합니다. 저장소에는 포함하지
  않으며, 없으면 해당 테스트는 건너뜁니다.
- `python scripts/live_vlm.py <영상 경로>`는 실제 Gemini 분석 결과를
  확인합니다(키 필요).
- CI(`.github/workflows/ci.yml`)는 `tests/`와 정적 검사, 그리고 합성 영상
  `apptest`를 실행합니다. `pip check`와 Dependabot도 설정되어 있습니다.

## 개인정보 처리

- 원본 영상은 실행 환경의 임시 폴더에서만 처리됩니다. 렌더 큐에 등록된
  작업은 처리 동안 볼륨의 `job_inputs/`에 임시 복사되고, 완료/취소 시
  즉시 삭제됩니다. 실패한 작업만 재시도용으로 24시간 보관 후 삭제됩니다.
- VLM에는 **샘플 프레임 이미지**와 선택적 차량 크롭만 전송하며 영상 파일
  전체를 보내지 않습니다. 업로드 화면에 이 사실을 고지합니다.
- 결과 파일은 `outputs/`에 저장되며, 기본 24시간이 지난 파일은 자동 정리됩니다.
  전체 결과가 500개 또는 20GB를 넘으면 오래된 것부터 지워집니다
  (`MOSAIC_MAX_OUTPUT_FILES`, `MOSAIC_MAX_OUTPUT_GB`로 조정). 보관이 필요한
  결과는 다운로드해 두세요.
- `오디오 제거`를 켜지 않으면 목소리·대화는 결과에 남습니다. 시각 정보만
  자동으로 가리므로 공개 전에 전체 구간을 직접 검수해야 합니다.
- API 키는 환경변수/Secret에서만 읽고 화면·로그·코드에 남기지 않습니다.

## 프로젝트 구조

```text
app.py                      Streamlit UI (로그인 → 분석 목록 → 렌더 → 미리보기/다운로드)
modal_app.py                Modal 배포 설정 (GPU, 영속 볼륨)
src/auth.py                 회원가입/로그인 (SQLite users.db, PBKDF2 해시)
src/config.py               카테고리/UI 그룹/.env 로더
src/preview.py              VLM 샘플 분석 + 체크리스트 항목
src/vlm_package.py          VLM 호출/파싱/재시도
src/select.py               VLM 판정 인터페이스
src/faces.py                YuNet 얼굴 검출/추적/각도 + 회전 검출 + 사람 크롭 확대
src/face_mesh.py            MediaPipe 얼굴 외곽(사람마다 다른 실루엣)
src/plates.py               번호판 전용 검출 + 차량 게이팅
src/texts.py                EAST 글자 영역 + VLM 주소 박스 정제
src/flow.py                 번호판·간판 광학 흐름 추적
src/offline.py              2패스 역방향 채움·평활·ID 전환 보간
src/pipeline.py             YOLO 추적 + 모자이크 렌더 파이프라인
src/render.py               모자이크/FFmpeg H.264 유틸
src/render_service.py       체크리스트 → 렌더 요청 변환/서명/캐시
src/render_queue.py         (구) 세션 내 렌더 직렬화
src/jobqueue.py             지속형 렌더 작업 큐(SQLite) + 계정별 스케줄러
src/worker.py               Streamlit과 분리된 큐 워커 프로세스/감시
src/chunk_render.py         긴 영상 구간 렌더 + 완료 구간 체크포인트
src/face_audit.py           렌더 후 얼굴 마스크 누락 점검/자동 보정
src/history.py              사용자별 결과 기록 (SQLite jobs.db)
src/retention.py            보관/삭제 정책 (24시간·개수·용량)
requirements.txt            CPU/headless 의존성
requirements-gpu.txt        CUDA GPU 의존성
requirements-dev.txt        pytest/ruff
.env.example                Gemini 키 템플릿 (.env로 복사해 사용)
configs/bytetrack_stable.yaml  ByteTrack 설정
weights/
  yolo11n-seg.pt            로컬 테스트용 기본 모델
  face_detection_yunet_2023mar.onnx  얼굴 전용 검출 모델
  (배포 이미지는 /opt/weights에 yolo11m/s-seg, yolo11n-pose를 빌드 때 받음)
assets/style.css            앱 스타일
tests/                       pytest 단위·통합 테스트
scripts/
  smoke/authtest/checklist/qualitytest/sampletest/freeform/regression/
  robustness/apptest/golden_run/leak_metrics/live_apptest/address_eval  자동 검사/측정
  make_ci_video.py          CI용 합성 테스트 영상 생성
  bench_gpu.py              GPU 전체 렌더 FPS/피크 VRAM 측정
  live_vlm.py               실제 Gemini 분석 확인용
HANDOFF.md                  인수인계/검증 결과/운영 제약
```
