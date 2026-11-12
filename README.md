# 영상 속 개인정보 보호 모자이크

영상에서 개인정보(얼굴, 차량번호판, 주소·간판, 명찰, 신분증, 휴대폰 화면,
차량동승자)를 찾아 **목록으로 보여주고**, 사용자가 고른 항목만 실제
모자이크 영상으로 렌더링하는 Streamlit 앱입니다.

## 동작 흐름

0. **회원가입 · 로그인**
   - 앱을 열면 먼저 로그인 화면이 나옵니다. 처음이면 `회원가입` 탭에서
     아이디(영문/숫자/_ 3~20자)와 비밀번호(8자 이상)를 등록합니다.
   - 계정은 프로젝트 루트의 `users.db`(SQLite)에 저장되며, 비밀번호는
     PBKDF2-SHA256 + 사용자별 salt 해시로만 저장됩니다.
   - 로그인한 아이디별로 결과 파일과 `지난 작업` 기록이 분리됩니다.

1. **업로드 → VLM 분석**
   - 고르게 샘플링한 프레임 4장을 Gemini VLM에 보내 개인정보 후보와 위치를 찾습니다.
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
   - 번호판은 VLM priors를 시간 방향으로 보간하고, 이동/오차를 덮도록
     조금 더 넓게 가립니다.
   - `veryfast` H.264 MP4로 인코딩하며 원본 오디오와 영상 길이를 유지합니다.

4. **미리보기 · 다운로드**
   - 사진이 아니라 **모자이크가 실제로 적용된 동영상**을 재생해 확인합니다.
   - 결과 파일명은 `아이디--원본이름_시각_토큰.mp4` 형식이라 계정별로
     구분됩니다. `지난 작업`에는 현재 로그인한 아이디의 결과만 보입니다.
   - 확인 후 다운로드합니다. `받으면 결과 자동 삭제`를 켜면 다운로드 시
     서버 결과 파일과 기록을 지웁니다.

## 고정 설정

속도 모드/샘플 수/모델 선택 UI는 없습니다. 항상 아래 설정으로 동작합니다.

| 설정 | 값 |
|---|---|
| 사람 검출/추적 | `yolo11n-seg.pt` |
| 추론 해상도 | `imgsz=480` |
| 분석 샘플 수 | `4` |
| 키프레임 간격 | `15` |
| 인코딩 프리셋 | `veryfast` |
| Ultra 모드 | 사용 안 함 |

무거운 `yolo11m-seg.pt`/`yolo11x-seg.pt`는 앱 고정 설정에 쓰지 않아
로딩/메모리를 줄이기 위해 제거했습니다. 코드 기본값도 `yolo11n-seg.pt`로
맞춰 두어 구 호출 경로가 무거운 모델을 로딩하지 않습니다.

## 설치

CUDA GPU 환경:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt
```

CPU만 사용할 때:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## VLM API 키 (Gemini 전용)

VLM은 **Google Gemini 하나만** 사용합니다. 키는 프로젝트 루트의 `.env`
파일에만 넣습니다. **키를 채팅이나 코드에 붙여넣지 마세요.**

`.env.example`은 키를 넣는 파일이 아니라 **템플릿**입니다. 처음 한 번만
`.env`로 복사해서 값을 채우면 됩니다. `.env.example`은 커밋용으로
의도적으로 남겨둡니다. 실제 키는 `.env`에만 두고, `.env`는 `.gitignore`
에 등록되어 있어 커밋되지 않습니다.

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` 내용:

```dotenv
GEMINI_API_KEY=...
```

저장한 뒤 Streamlit을 재시작하면 앱이 키를 자동으로 읽습니다. 앱에는 키
입력창이 없으며, 키 값은 화면과 로그에 출력하지 않습니다. 키가 없으면
앱이 짧은 안내 문구를 보여주고 분석을 시작하지 않습니다.

모델명을 바꾸려면 같은 파일에 `VLM_MODEL=...`을 추가합니다
(기본값 `gemini-3-flash-preview`).

## 로그인 · 계정

- 앱을 열면 로그인 화면이 먼저 표시됩니다. `회원가입` 탭에서 계정을 만든 뒤
  `로그인` 탭에서 접속합니다.
- 아이디는 영문/숫자/_ 3~20자, 비밀번호는 8자 이상입니다.
- 계정은 프로젝트 루트의 `users.db`(SQLite)에 자동 생성·저장되며,
  `users.db`는 `.gitignore`에 등록되어 커밋되지 않습니다.
- 비밀번호는 PBKDF2-SHA256(20만 회) + 사용자별 salt 해시로 저장되고
  평문으로 저장하지 않습니다.
- 5회 연속 실패하면 60초 동안 로그인이 잠깁니다.
- 로그인 후 사이드바에서 현재 아이디를 확인하고 `로그아웃`할 수 있습니다.
- 결과 파일은 `아이디--원본이름_시각_토큰.mp4`로 저장되며, `지난 작업`은
  현재 아이디로 시작하는 파일만 보여줍니다.
- `users.db`를 지우면 모든 계정과 로그인 정보가 초기화됩니다(결과 영상은
  `outputs/`에 그대로 남습니다).

## 실행 · 접속

프로젝트 폴더에서 Streamlit 서버를 실행합니다.

```powershell
cd C:\Users\gong0000\Desktop\mosaic-web
.\.venv\Scripts\python.exe -m streamlit run app.py `
  --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

브라우저에서 **http://127.0.0.1:8501** 을 엽니다.

- 같은 PC에서는 `http://localhost:8501`도 됩니다.
- 서버 콘솔에 표시되는 `Local URL` / `Network URL`을 그대로 사용해도 됩니다.
- 같은 네트워크의 다른 기기에서 접속하려면 `--server.address 0.0.0.0`으로
  실행하고, Windows 방화벽에서 8501 포트를 허용한 뒤
  `http://<서버 PC의 IP>:8501`로 접속합니다.
- 서버를 끄려면 실행 중인 터미널에서 `Ctrl+C`를 누릅니다.
- 코드를 수정한 뒤에는 서버를 다시 시작해야 반영됩니다.
- 로그인 세션은 브라우저 탭에 유지되며, 서버를 재시작해도 계정(`users.db`)은
  유지됩니다.

## 클라우드 배포 (Modal, GPU)

다른 컴퓨터에서 접속할 수 있도록 Modal에 올리는 방법입니다. 가입 후 매월
$30 크레딧이 제공되고, 접속이 없으면 컨테이너가 자동으로 내려가 GPU 비용이
계속 나가지 않습니다.

```powershell
pip install modal python-dotenv
modal setup                      # 브라우저에서 로그인 (최초 1회)
modal deploy modal_app.py
```

배포가 끝나면 출력에 `https://....modal.run` 주소가 표시됩니다. 이 주소를
다른 컴퓨터에서 그대로 열면 됩니다.

- 첫 배포는 CUDA용 torch 이미지를 빌드하므로 10분 정도 걸릴 수 있습니다.
- `users.db`와 `outputs/`는 Modal Volume `mosaic-data`에 저장됩니다.
  컨테이너가 재시작·재배포돼도 계정과 결과가 유지됩니다.
- `GEMINI_API_KEY`는 로컬 `.env`에서 읽어 Secret으로 전달합니다. `.env`는
  이미지에 포함되지 않습니다.
- Streamlit 탭을 닫으면 약 5분 뒤 컨테이너가 내려갑니다. 탭을 열어둔 채
  방치하면 GPU가 계속 과금되므로 사용 후 탭을 닫으세요.
- GPU 사양은 `modal_app.py`의 `gpu="T4"`에서 바꿀 수 있습니다.
- 로그 확인/중지: `modal app logs mosaic-web`, `modal app stop mosaic-web`.

## 실제 VLM 품질 확인

키를 넣은 뒤 샘플 영상으로 분석 결과를 확인할 수 있습니다.

```powershell
python scripts/live_vlm.py C:\Users\gong0000\Downloads\street.mp4
```

찾은 카테고리, 박스 수, 호출 시간, 어노테이션 이미지 저장 경로를 출력합니다.
키가 없으면 실행을 거부하며 키를 출력하지 않습니다.

## 테스트

VLM 스텁으로 키 없이 실행할 수 있습니다.

```powershell
python scripts/smoke.py
python scripts/authtest.py
python scripts/checklist.py
python scripts/freeform.py
python scripts/regression.py
python scripts/robustness.py
python scripts/apptest.py
```

`authtest.py`는 임시 DB로 회원가입/로그인/비밀번호 해시 저장과 로그인
화면 노출을 확인합니다(실제 `users.db`는 건드리지 않습니다).
`apptest.py`는 로그인 세션을 주입한 뒤 업로드 → 분석 목록 →
`모자이크 시작` → 결과 파일 생성 → 다운로드 버튼 노출까지 확인합니다.

## 개인정보 처리

- 원본 영상은 로컬 임시 폴더에서만 처리됩니다.
- VLM에는 **샘플 프레임 이미지**만 전송하며 영상 파일 전체를 보내지 않습니다.
- 결과 파일은 `outputs/`에 저장되며, 기본 24시간이 지난 파일은 앱 시작 시
  자동 정리됩니다. 보관이 필요한 결과는 `outputs/` 밖으로 옮기세요.
  테스트 산출물은 필요 없으면 `outputs/`에서 지워도 됩니다.
- API 키는 환경변수/`.env`에서만 읽고 화면·로그·코드에 남기지 않습니다.

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
configs/bytetrack_stable.yaml  ByteTrack 설정
weights/
  yolo11n-seg.pt            앱 고정 person/vehicle 모델
  face_detection_yunet_2023mar.onnx  얼굴 전용 검출 모델
assets/style.css            앱 스타일
scripts/
  smoke/authtest/checklist/freeform/regression/robustness/apptest  자동 검사
  live_vlm.py               실제 Gemini 분석 확인용
```
