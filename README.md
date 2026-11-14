# 영상 속 개인정보 보호 모자이크

영상에서 개인정보(얼굴, 차량번호판, 주소·간판, 명찰, 신분증, 휴대폰 화면,
차량동승자)를 찾아 **목록으로 보여주고**, 사용자가 고른 항목만 실제
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
- 서비스 주소는 팀 내부에만 공유하세요. 회원가입이 열려 있어 링크를 아는
  사람이 크레딧을 사용할 수 있습니다.

## 동작 흐름

0. **회원가입 · 로그인**
   - 앱을 열면 먼저 로그인 화면이 나옵니다. 처음이면 `회원가입` 탭에서
     아이디(영문/숫자/_ 3~20자)와 비밀번호(8자 이상)를 등록합니다.
   - 계정은 서버의 `users.db`(SQLite)에 저장되며, 비밀번호는
     PBKDF2-SHA256 + 사용자별 salt 해시로만 저장됩니다.
   - 로그인한 아이디별로 결과 파일과 `지난 작업` 기록이 분리됩니다.

1. **업로드 → VLM 분석**
   - 균등 4장에 더해 **장면 전환 직후**와 **움직임 피크** 프레임을 섞어
     최대 10장을 Gemini VLM에 보내 개인정보 후보와 위치를 찾습니다.
   - 화면에는 자동 모자이크가 아니라 **체크리스트**만 표시됩니다.
   - 업로드/구간이 바뀌면 다시 분석합니다. 프롬프트만 바꾸면 VLM을
     다시 호출하지 않고 체크 항목만 갱신합니다.

2. **가릴 항목 선택**
   - `요청한 항목`과 `추가로 찾은 개인정보`를 나눠 보여줍니다.
   - 프롬프트가 요구했지만 샘플에서 못 찾은 항목은 `못 찾음`으로 표시됩니다.
   - 얼굴은 선택하면 프레임마다 로컬 얼굴 검출로 적용해 누락을 줄입니다.

3. **모자이크 시작**
   - 버튼을 눌러야만 전체 영상을 렌더링합니다. 자동 렌더는 하지 않습니다.
   - 같은 설정의 이전 결과가 있어도 새로 렌더링합니다.
   - 렌더 단계에서는 VLM을 다시 호출하지 않습니다. 분석 priors,
     YOLO11-seg 추적, 얼굴 전용 경로의 OpenCV YuNet 검출을 사용합니다.
   - 얼굴만 선택하면 얼굴 박스만 타원형으로 가려 목·상체로 퍼지지 않습니다.
     YuNet이 놓친 프로필·고개 숙인 얼굴은 사람 박스의 머리 영역으로 보완합니다.
   - 번호판은 VLM priors를 시간 방향으로 보간하고, 물체 크기에 비례한
     작은 여백만 붙여 번짐을 줄입니다.
   - `veryfast` H.264 MP4로 인코딩하며 원본 오디오와 영상 길이를 유지합니다.

4. **미리보기 · 다운로드**
   - 사진이 아니라 **모자이크가 실제로 적용된 동영상**을 재생해 확인합니다.
   - 결과 파일명은 `아이디--원본이름_시각_토큰.mp4` 형식이라 계정별로
     구분됩니다. `지난 작업`은 로그인 직후 업로드 전에도 볼 수 있고,
     현재 로그인한 아이디의 결과만 표시됩니다.
   - 확인 후 다운로드합니다. `받으면 결과 자동 삭제`를 켜면 다운로드 시
     서버 결과 파일과 기록을 지웁니다.

## 고정 설정

속도 모드/샘플 수/모델 선택 UI는 없습니다. 항상 아래 설정으로 동작합니다.

| 설정 | 값 |
|---|---|
| 사람 검출/추적 | `yolo11n-seg.pt` |
| 추론 해상도 | `imgsz=640` |
| 분석 샘플 수 | 균등 4장 + 최대 6장(장면 전환·움직임 피크) = 최대 10장 |
| VLM 분당 호출 상한 | 12회 (무료 티어 분당 한도 보호) |
| 번호판 확대 탐지 | 영상당 최대 10회 |
| 키프레임 간격 | `15` |
| 인코딩 프리셋 | `veryfast` |
| Ultra 모드 | 사용 안 함 |

무거운 `yolo11m-seg.pt`/`yolo11x-seg.pt`는 앱 고정 설정에 쓰지 않아
로딩/메모리를 줄이기 위해 제거했습니다. 코드 기본값도 `yolo11n-seg.pt`로
맞춰 두어 구 호출 경로가 무거운 모델을 로딩하지 않습니다.

## 샘플링 (분석 정확도)

분석에 보낼 프레임은 균등 배치만 쓰지 않습니다(`src/sampling.py`).

| 종류 | 개수 | 목적 |
|---|---|---|
| 균등 기본 샘플 | 4 | 영상 전체를 고르게 훑는다 |
| 장면 전환 샘플 | 최대 4 | 화면이 크게 바뀐 지점 직후 |
| 움직임 피크 샘플 | 최대 3 | 컷 없는 CCTV에서 사람이 스치는 지점 |

- 영상 전체를 저해상도(64×36) 서명으로 훑어 움직임 변화량을 구합니다.
- 컷/피크 임계값은 영상마다 중앙값·MAD로 적응적으로 정합니다. 카메라
  흔들림을 컷으로 오탐하지 않기 위한 것입니다.
- 10초 영상에서 2.2초에만 스친 신분증처럼, 균등 샘플이 놓치는 구간을
  잡아냅니다. 총 호출 수는 10장으로 제한해 시간과 비용을 예측 가능하게
  유지합니다.

## 모자이크 품질

| 항목 | 처리 |
|---|---|
| 모자이크 블록 크기 | ROI 크기가 아니라 '작은 축의 블록 개수'(기본 10개)로 계산. 예전처럼 ROI 크기에 비례해 크면 얼굴이 몇 개 덩어리로 뭉개집니다 |
| 얼굴 | YuNet 타이트 박스를 조금만 넓힌 뒤 타원으로 가림(목·어깨로 번지지 않음) |
| 얼굴 누락 | YuNet이 놓친 프로필/고개 숙인 얼굴은 사람 박스 머리 영역으로 보완 |
| 번호판 | 여백을 고정 24px가 아니라 물체 크기 비례(최대 8px)로 줄여 번짐 방지 |
| 영역 박스 | VLM이 넓게 잡은 번호판 박스를 실제 글자 영역에 맞춰 조임(최소 60% 유지) |
| 사람 마스크 | 모자이크를 '실제로 가리는 영역' 기준으로 만들어 블록이 과하게 굵어지지 않음 |

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
내려가 GPU 비용이 계속 나가지 않습니다.

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
- `GEMINI_API_KEY`는 Modal Named Secret `mosaic-gemini`에서 읽습니다.
  `.env`를 들고 있는 PC 말고도 **Modal 로그인만 되어 있으면 배포할 수
  있습니다.** `.env`는 이미지에 포함되지 않고 커밋되지 않습니다.
  모델명은 Secret의 `VLM_MODEL`로 바꿉니다(기본 `gemini-3-flash-preview`).
- GPU 사양은 `modal_app.py`의 `gpu="T4"`에서 바꿉니다.
- 로그: `modal app logs mosaic-web --tail 100` (실시간은 `-f`)
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
python scripts/smoke.py
python scripts/authtest.py
python scripts/checklist.py
python scripts/freeform.py
python scripts/regression.py
python scripts/robustness.py
python scripts/apptest.py
```

- `authtest.py`는 임시 DB로 회원가입/로그인/비밀번호 해시 저장과 로그인
  화면 노출을 확인합니다(실제 `users.db`는 건드리지 않습니다).
- `apptest.py`는 로그인 세션을 주입한 뒤 업로드 → 분석 목록 →
  `모자이크 시작` → 결과 파일 생성 → 다운로드 버튼 노출까지 확인합니다.
- `python scripts/live_vlm.py <영상 경로>`는 실제 Gemini 분석 결과를
  확인합니다(키 필요).

## 개인정보 처리

- 원본 영상은 실행 환경의 임시 폴더에서만 처리됩니다.
- VLM에는 **샘플 프레임 이미지**만 전송하며 영상 파일 전체를 보내지 않습니다.
- 결과 파일은 `outputs/`에 저장되며, 기본 24시간이 지난 파일은 앱 시작 시
  자동 정리됩니다. 보관이 필요한 결과는 다운로드해 두세요.
- API 키는 환경변수/Secret에서만 읽고 화면·로그·코드에 남기지 않습니다.

## 프로젝트 구조

```text
app.py                      Streamlit UI (로그인 → 분석 목록 → 렌더 → 미리보기/다운로드)
modal_app.py                Modal 배포 설정 (GPU, 영속 볼륨)
src/auth.py                 회원가입/로그인 (SQLite users.db, PBKDF2 해시)
src/config.py               카테고리/프리셋/.env 로더
src/preview.py              VLM 샘플 분석 + 체크리스트 항목
src/vlm_package.py          VLM 호출/파싱/재시도
src/select.py               VLM 판정 인터페이스
src/faces.py                YuNet 얼굴 검출/추적
src/pipeline.py             YOLO 추적 + 모자이크 렌더 파이프라인
src/render.py               모자이크/FFmpeg H.264 유틸
src/render_service.py       체크리스트 → 렌더 요청 변환/서명/캐시
src/history.py              결과 기록
src/retention.py            보관/삭제 정책
requirements.txt            CPU/headless 의존성
requirements-gpu.txt        CUDA GPU 의존성
.env.example                Gemini 키 템플릿 (.env로 복사해 사용)
configs/bytetrack_stable.yaml  ByteTrack 설정
weights/
  yolo11n-seg.pt            앱 고정 person/vehicle 모델
  face_detection_yunet_2023mar.onnx  얼굴 전용 검출 모델
assets/style.css            앱 스타일
scripts/
  smoke/authtest/checklist/freeform/regression/robustness/apptest  자동 검사
  live_vlm.py               실제 Gemini 분석 확인용
```
