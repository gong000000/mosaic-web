# 영상 속 개인정보 보호 모자이크 시스템

영상에서 개인정보(얼굴, 번호판, 간판, 명찰, 신분증 등)를 찾아 체크리스트로 보여주고,
선택한 항목만 모자이크하는 웹 앱

VLM이 영상 프레임을 직접 보고 개인정보 항목과 위치를 찾는다. 앱은 결과를
카테고리별 체크리스트로 보여주고, 사용자가 체크한 항목만 모자이크한다.
VLM API 키는 필수이며, 키가 없으면 분석을 시작하지 않는다. 로컬 검출로
대신 판정하지 않는다.

## 동작 방식

1. **사람 검출**: YOLO11-seg + ByteTrack이 사람 박스·마스크와 트랙 ID를 만든다.
2. **VLM 분석**: 영상에서 고르게 샘플링한 프레임(4/6/8/12개 선택 가능)을 VLM에 보내고,
   개인정보 항목과 박스를 받아 체크리스트를 만든다.
   - 프롬프트(예: "얼굴과 차량번호판을 모자이크해줘")는 VLM 요청에 그대로 포함된다.
   - 체크리스트는 `요청한 항목`과 `추가로 찾은 개인정보`로 나뉘며 둘 다 기본 체크된다.
   - 프레임은 최대 1920px로 전송해 작은 번호판·명찰도 최대한 읽는다.
   - 번호판은 전체 프레임 탐지 외에 차량 크롭을 확대한 VLM 보조 탐지를 수행한다.
     차량 영역은 YOLO11-seg가 제안하지만, 번호판 여부와 박스는 VLM이 결정한다.
     확대 호출은 영상 전체 최대 6회, 프레임당 최대 2회로 제한된다.
   - 업로드 내용(SHA-256)이 바뀌면 이전 분석 결과를 재사용하지 않는다.
3. **2패스 렌더**: 1패스에서 `(사람 트랙, 카테고리)`별 VLM 판정을 모아 확정하고,
   2패스에서 선택된 카테고리만 렌더한다.
   - 1패스와 2패스는 모든 프레임을 같은 설정으로 추적해 트랙 ID를 일치시킨다.
     VLM 탐지는 1패스의 키프레임에서만 한다.
   - 사람 카테고리(얼굴, 동승자, 명찰)는 사람 마스크/얼굴 영역을 가린다.
     VLM이 찾은 명찰은 `가리는 범위`와 무관하게 박스 위치를 직접 가린다.
     얼굴·동승자·자유형 사람 항목의 직접 박스는 사람 박스의 `가리는 범위`
     안으로 잘라서 적용한다. 그래서 `얼굴만`을 골라도 선택한 명찰은 가려진다.
   - 배경 카테고리(번호판, 주소·간판, 신분증, 휴대폰 화면)는 VLM이 찾은
     사각형 영역을 가린다.
   - 체크리스트에서 VLM이 찾은 박스는 처리 단계의 초기값으로 넘어간다.
     배경 박스는 `region_priors`, 사람 부착 박스(명찰 등)는 `person_priors`다.
     처리 중 VLM이 같은 프레임에서 놓쳐도 체크한 항목은 계속 가려진다.
     체크리스트 샘플보다 이른 프레임에는 해당 박스를 그리지 않는다.
   - 모자이크 픽셀은 VLM 박스/사람 마스크를 받아 OpenCV와 ffmpeg가 그린다.
   - MOV/AVI 등은 브라우저 미리보기와 OpenCV 호환을 위해 H.264 MP4 사본으로
     한 번 변환한 뒤 처리한다(원본 업로드 파일은 보관).
   - 1패스 결과는 캐시 파일로 저장되며, 재실행 시 VLM 호출을 절약할 수 있다.

## 카테고리

| 키 | 표시 | 처리 |
|---|---|---|
| `face` | 얼굴 | 사람 |
| `passenger` | 차량동승자 | 사람 |
| `name_tag` | 명찰 | 사람 |
| `license_plate` | 차량번호판 | 배경 영역 |
| `address` | 주소·간판 | 배경 영역 |
| `id_card` | 신분증·문서 | 배경 영역 |
| `phone_screen` | 휴대폰 화면 | 배경 영역 |

프롬프트에 없는 자유 형식 항목(예: "빨간 옷 입은 사람")은 VLM이
`matches_prompt`와 `region_type`을 함께 주면 사람 또는 배경 항목으로 처리된다.

## 구조

```
app.py                  메인 앱: 업로드, 프롬프트/프리셋, 체크리스트, 모자이크, 다운로드
src/                    라이브러리
  config.py             카테고리/별칭/타입, 프리셋, VLM 제공자, 출력 경로
  detect.py             검출 후처리 (작은 박스 제거, 겹친 박스 필터링, 박스 매칭)
  preview.py            영상 4/6/8/12샘플 분석: 사람 탐지 + VLM 체크리스트 + 번호판 확대
  select.py             VLMSelector: VLM 탐지, 사람 박스 매칭
  vlm_package.py        프레임 base64 전송, 프롬프트, JSON/좌표/카테고리 파싱, VLM 재시도 로직
  vlm_budget.py         VLM 호출 예산 관리(VLMBudget), 재시도 정책(VLMRetryPolicy), 오류 분류
  render.py             사람/배경 모자이크, ffmpeg 믹싱·자르기·MP4 변환
  models.py             YOLO 모델 로딩 캐시 + 동시 로딩/추론 락
  pipeline.py           2패스 파이프라인: 카테고리별 투표, 선택 렌더, region/person
                        priors, 사람 부착 VLM 박스(명찰 등) 직접 렌더, 1패스 결과 캐시
  history.py            작업 히스토리 (outputs/jobs.json, 원자적 저장)
  retention.py          24시간 보관 정책, 다운로드 시 삭제 옵션
configs/                추적기 설정 (bytetrack_stable.yaml)
weights/                모델 가중치 (yolo11n/m/x-seg.pt)
outputs/                결과물 + 작업 기록 (jobs.json)
scripts/
  smoke.py              패키지/프로젝트 모듈 import, 카테고리 설정, conflict marker 검사
  robustness.py         강건성 검사 (예외 입력, VLM 파싱/제공자/키 필수)
  regression.py         회귀 테스트 (face.mp4 트랙 5명 렌더)
  evaluate.py           첫 프레임 VLM 탐지 리포트 (API 키 필요)
  checklist.py          체크리스트/배경·사람 priors/보간/due 모의 VLM 검사
  freeform.py           자유 형식 person 카테고리 + 명찰 직접 마스킹 검사 (모의 VLM)
  apptest.py            Streamlit AppTest E2E (스텁 VLM: 업로드→분석→모자이크)
  bench.py              속도 벤치마크 (모델·해상도별 FPS)
```

## VLM 호출 아키텍처 (신규)

### 호출 예산 관리 (`src/vlm_budget.py`)
- `VLMBudget`: 일일 호출 한도와 영상당 호출 한도를 관리. 무료 등급(Gemini 하루 20회 등)
  초과를 방지하고, 한도 소진 시 보조 호출(번호판 확대 등)을 건너뛴다.
- `VLMRetryPolicy`: 지수 백오프(1→2→4초) 기반 재시도. 429(한도 초과), 401/403(인증)
  오류는 즉시 실패 처리하고 불필요한 재시도를 하지 않는다.

### 재시도 로직 (`src/vlm_package.py`)
- `detect_with_vlm()`: 5xx 서버 오류 시 최대 3회 재시도. 한도 초과나 인증 오류는 재시도하지 않음.
- `classify_vlm_error()`: 오류를 `quota_exceeded`, `auth_error`, `model_not_found`,
  `server_error`, `parse_error`, `unknown_error`로 분류.
- `friendly_vlm_message()`: 사용자에게 보여줄 친절한 오류 메시지 생성.

## 실행 방법

### 1. Python 가상환경 생성 (Python 3.12)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 라이브러리 설치
```powershell
pip install -r requirements.txt
```
(GPU가 없는 저사양 PC는 `requirements-cpu.txt`를 대신 사용)

### 3. 실행
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```
브라우저에서 `http://localhost:8501`로 접속한다.

### 4. VLM 설정 (OpenAI / Google Gemini)

앱 사이드바의 `3. VLM`에서 제공자를 고르고 API 키를 입력한다.
API 키는 필수이며, 입력 전에는 분석이 시작되지 않는다.

**Google Gemini**
1. <https://aistudio.google.com/apikey> 에서 API 키를 만든다.
2. 앱에서 `VLM 제공자`를 `Google Gemini`로 바꾼다.
   모델명 기본값은 `gemini-3-flash-preview`이다.
   (모델이 없다고 나오면 `gemini-flash-lite-latest`를 시도)
3. `API 키`에 복사한 키를 붙여넣는다.

**OpenAI**: <https://platform.openai.com/api-keys> 에서 키를 만들고 제공자
`OpenAI`, 모델 `gpt-4o-mini`(기본)를 사용한다.

키프레임 1장이 요청 1회이므로 30fps 10초 영상 기준 `빠름` 약 20회, `보통` 약 60회,
`정밀` 약 300회 호출된다. 여기에 체크리스트 분석 4~12회와 번호판 확대 호출
(영상 전체 최대 6회)이 더해진다. 제공자별 호출 한도와 약관을 확인한 뒤 사용한다.

## 실제 VLM 검증 결과 (2026-11-05)

| 영상 | 카테고리 | 검출 결과 |
|------|----------|----------|
| `street.mp4` | `license_plate` (차량번호판) | **2개 검출** ✅ (박스: [1244,458,1275,487], [856,677,873,685]) |
| `face.mp4` | `face` (얼굴) | **5개 검출** ✅ (처음 3개: [215,60,396,243], [424,128,526,273], [554,320,645,426]) |
| `face.mp4` | `name_tag` (명찰) | 첫 프레임 미검출 (두 번째 이후 프레임에 있거나 영상에 없을 수 있음) |

실제 Gemini `gemini-3-flash-preview` API 호출로 확인했다.

## 개인정보 처리

- 업로드한 영상 파일은 서버(실행 PC)의 임시 폴더에서만 처리된다.
  서버 시작 시 24시간이 지난 업로드 임시 폴더를 자동으로 지운다.
- VLM 판정에 쓰는 **영상 프레임(샘플 4~12장 + 처리 키프레임)** 이 선택한
  제공자로 전송된다. 영상 파일 자체는 전송되지 않는다.
- 결과물은 `outputs/`에 저장되며 24시간이 지나면 자동 삭제된다.
  "받으면 결과 자동 삭제"를 켜면 다운로드 직후 삭제된다.
- 작업 히스토리(`outputs/jobs.json`)는 파일명·프롬프트·설정 요약만 남기며,
  최대 100건을 유지한다. 저장은 임시 파일 교체 방식이라 중간에 끊겨도
  기존 기록이 깨지지 않는다. 24시간이 지나 파일이 삭제되면 해당 기록도 정리한다.

## 현재 한계

- 체크리스트는 선택한 4/6/8/12개 샘플 프레임을 보고 만든다. 아주 짧게 등장하는
  항목은 목록에서 빠질 수 있다. 처리 단계에서는 키프레임마다 다시 탐지하며,
  체크리스트의 배경·사람 부착 박스는 `region_priors`/`person_priors`로 유지된다.
- 배경 영역은 키프레임 사이에서 앞뒤 박스를 선형 보간하지만, 빠르게 움직이거나
  가려졌다 나타나는 번호판은 순간적으로 어긋날 수 있다.
- VLM 호출이 실패하면 해당 분석/처리는 오류로 안내한다. 로컬 검출로 대신하지 않는다.
- Gemini 무료 등급은 모델별 하루 호출 제한이 있다(모델·지역에 따라 다름).
  기본 12샘플 체크리스트 분석과 키프레임별 VLM 처리(번호판 확대 호출 포함)를
  함께 실행하면 한도를 쉽게 넘을 수 있다. 무료 등급에서는 `체크리스트 샘플`을
  4개, `속도`를 `빠름`으로 두고 짧은 영상으로 호출 수를 줄인다. 한도에 걸리면
  앱이 일일 한도 초과로 안내한다. 내일 다시 시도하거나 유료 키를 사용하세요.
- CPU 환경에서는 긴 영상을 처리하면 느리다. 데모용으로는 짧은 영상(30~60초)을 권장한다.
- `가리는 범위`(전체/상반신/얼굴만)는 사람 마스크와 얼굴·동승자·자유형 사람
  박스의 범위를 정한다. 선택한 명찰은 이 설정과 무관하게 VLM이 찾은 박스
  위치를 별도로 가리므로 `얼굴만`·`상반신`·`전체` 어디서든 명찰이 보이면
  가려진다. 배경 항목도 이 설정과 무관하게 박스 영역만 가린다.
- 외부 URL 배포(HTTPS)는 아직 하지 않았다. 현재는 로컬 Streamlit 실행 단계다.

## 기술 스택

- **Frontend**: Streamlit
- **Object Detection / Segmentation**: YOLO11-seg (사람 박스·마스크, 차량 크롭 제안)
- **VLM**: OpenAI GPT-4o-mini 또는 Google Gemini Flash
  (사이드바에서 선택, OpenAI 호환 API)
- **Multi-Object Tracking**: ByteTrack
- **Video Encoding**: H.264 via ffmpeg (imageio-ffmpeg)
- **언어**: Python 3.12

## 검사

```powershell
.\.venv\Scripts\python.exe scripts\smoke.py
.\.venv\Scripts\python.exe scripts\robustness.py
.\.venv\Scripts\python.exe scripts\regression.py
.\.venv\Scripts\python.exe scripts\checklist.py
.\.venv\Scripts\python.exe scripts\freeform.py
.\.venv\Scripts\python.exe scripts\apptest.py
```
