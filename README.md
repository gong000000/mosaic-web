# 영상 속 개인정보 보호 모자이크 시스템

영상에서 개인정보(얼굴, 번호판, 간판, 명찰, 신분증 등)를 찾아 체크리스트로 보여주고, 선택한 항목만 모자이크하는 웹 앱

VLM이 영상 프레임을 직접 보고 개인정보 항목과 위치를 찾는다. 앱은 결과를
카테고리별 체크리스트로 보여주고, 사용자가 체크한 항목만 모자이크한다.
VLM API 키는 필수이며, 키가 없으면 분석을 시작하지 않는다. 로컬 검출로
대신 판정하지 않는다.

## 동작 방식

1. **사람 검출**: YOLO11-seg + ByteTrack이 사람 박스·마스크와 트랙 ID를 만든다.
2. **VLM 분석**: 영상 전체에서 고르게 12개 프레임(4~12개 선택)을 샘플링해 VLM에 보내고,
   개인정보 항목과 박스를 받아 체크리스트를 만든다.
   - 프롬프트(예: "얼굴과 차량번호판을 모자이크해줘")는 VLM 요청에 그대로 포함된다.
   - 체크리스트는 `요청한 항목`과 `추가로 찾은 개인정보`로 나뉘며 둘 다 기본 체크된다.
   - 프레임은 최대 1920px로 전송해 작은 번호판·명찰도 최대한 읽는다.
   - 번호판은 전체 프레임 탐지 외에 차량 크롭을 확대한 VLM 보조 탐지를 한 번 더
     수행한다. 차량 영역은 YOLO11-seg가 제안하지만, 번호판 여부와 박스는 VLM이 결정한다.
   - 같은 업로드/프롬프트/설정이면 VLM을 다시 호출하지 않고 결과를 재사용한다.
3. **2패스 렌더**: 1패스에서 `(사람 트랙, 카테고리)`별 VLM 판정을 모아 확정하고,
   2패스에서 선택된 카테고리만 렌더한다.
   - 1패스와 2패스는 모든 프레임을 같은 설정으로 추적해 트랙 ID를 일치시킨다.
     VLM 탐지는 1패스의 키프레임에서만 한다.
   - 사람 카테고리(얼굴, 동승자, 명찰)는 사람 마스크/얼굴 영역을 가린다.
<<<<<<< HEAD
     VLM이 찾은 명찰·얼굴 박스는 `가리는 범위`와 별개로 박스 위치를 직접
     가린다. 그래서 `얼굴만`을 골라도 선택한 명찰은 가려진다.
=======
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
   - 배경 카테고리(번호판, 주소·간판, 신분증, 휴대폰 화면)는 VLM이 찾은
     사각형 영역을 가린다.
   - 체크리스트에서 VLM이 찾은 배경 박스는 처리 단계의 초기값(`region_priors`)으로
     넘어간다. 처리 중 VLM이 같은 프레임에서 놓쳐도 체크한 항목은 계속 가려진다.
   - 모자이크 픽셀은 VLM 박스/사람 마스크를 받아 OpenCV와 ffmpeg가 그린다.

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
  preview.py            영상 4~12샘플 분석: 사람 탐지 + VLM 체크리스트 + 번호판 확대
  select.py             VLMSelector: VLM 탐지, 사람 박스 매칭
  vlm_package.py        프레임 base64 전송, 프롬프트, JSON/좌표 파싱
  render.py             사람/배경 모자이크, ffmpeg 오디오 믹싱, 구간 자르기
  models.py             YOLO 모델 로딩 캐시
<<<<<<< HEAD
  pipeline.py           2패스 파이프라인: 카테고리별 투표, 선택 렌더, region_priors,
                        사람 부착 VLM 박스(명찰 등) 직접 렌더
=======
  pipeline.py           2패스 파이프라인: 카테고리별 투표, 선택 렌더, region_priors
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
  history.py            작업 히스토리 (outputs/jobs.json)
  retention.py          24시간 보관 정책, 다운로드 시 삭제 옵션
configs/                추적기 설정 (bytetrack_stable.yaml)
weights/                모델 가중치 (yolo11n/m/x-seg.pt)
outputs/                결과물 + 작업 기록 (jobs.json)
scripts/
  smoke.py              설치 확인 (필수 라이브러리 import)
  robustness.py         강건성 검사 (예외 입력, VLM 파싱/제공자/키 필수)
  regression.py         회귀 테스트 (face.mp4 트랙 5명 렌더)
  evaluate.py           첫 프레임 VLM 탐지 리포트 (API 키 필요)
  checklist.py          체크리스트/배경 영역/region_priors 모의 VLM 검사
<<<<<<< HEAD
  freeform.py           자유 형식 person 카테고리 + 명찰 직접 마스킹 검사 (모의 VLM)
=======
  freeform.py           자유 형식 person 카테고리 검사 (모의 VLM)
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
  apptest.py            Streamlit AppTest E2E (스텁 VLM: 업로드→분석→모자이크)
  bench.py              속도 벤치마크 (모델·해상도별 FPS)
```

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
`py` 대신 가상환경 파이썬을 직접 지정해야 하는 환경도 있다.

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

키프레임 1장이 요청 1회이므로 10초 영상 기준 `빠름` 약 20회, `보통` 약 60회,
`정밀` 약 300회 호출된다. 여기에 체크리스트 분석 12회와 번호판 확대 호출
(차량별 최대 2회)이 더해진다. 제공자별 호출 한도와 약관을 확인한 뒤 사용한다.

## 개인정보 처리

- 업로드한 영상 파일은 서버(실행 PC)의 임시 폴더에서만 처리된다.
  서버 시작 시 24시간이 지난 업로드 임시 폴더를 자동으로 지운다.
- VLM 판정에 쓰는 **영상 프레임(샘플 4~12장 + 처리 키프레임)** 이 선택한
  제공자로 전송된다. 영상 파일 자체는 전송되지 않는다.
- 결과물은 `outputs/`에 저장되며 24시간이 지나면 자동 삭제된다.
  "받으면 결과 자동 삭제"를 켜면 다운로드 직후 삭제된다.
- 작업 히스토리(`outputs/jobs.json`)는 파일명·프롬프트·설정 요약만 남기며,
  최대 100건을 유지한다.

## 현재 한계

- 체크리스트는 기본 12개 샘플 프레임을 보고 만든다. 아주 짧게 등장하는 항목은
  목록에서 빠질 수 있다. 처리 단계에서는 키프레임마다 다시 탐지하며,
  체크리스트의 배경 박스는 `region_priors`로 유지된다.
- 배경 영역은 키프레임 사이에서 박스를 유지하므로, 빠르게 움직이는 번호판은
  순간적으로 어긋날 수 있다.
- VLM 호출이 실패하면 해당 분석/처리는 오류로 안내한다. 로컬 검출로 대신하지 않는다.
- 실제 Gemini 키로 번호판 검출, 체크리스트 표시, 선택한 번호판 모자이크
  적용까지 확인했다.
- Gemini 무료 등급은 모델별 하루 20회 호출이다. 기본 12샘플 체크리스트 분석과
  키프레임별 VLM 처리(번호판 확대 호출 포함)를 함께 실행하면 한도를 쉽게
  넘으므로 유료 키나 한도 상향이 필요할 수 있다. 무료 등급에서는
  `체크리스트 샘플`을 4개, `속도`를 `빠름`으로 두고 짧은 영상으로 호출 수를
  줄인다. 한도에 걸리면 앱이 일일 한도 초과로 안내한다.
- CPU 환경에서는 긴 영상을 처리하면 느리다. 데모용으로는 짧은 영상(30~60초)을 권장한다.
<<<<<<< HEAD
- `가리는 범위`(전체/상반신/얼굴만)는 사람 마스크/얼굴 영역의 범위를 정한다.
  선택한 명찰은 이 설정과 무관하게 VLM이 찾은 박스 위치를 별도로 가리므로
  `얼굴만`·`상반신`·`전체` 어디서든 명찰이 보이면 가려진다. 배경 항목도
  이 설정과 무관하게 박스 영역만 가린다.
=======
- `가리는 범위`(전체/상반신/얼굴만)는 사람 항목 전체에 하나의 값으로 적용된다.
  예를 들어 `얼굴만`을 고르면 명찰이 선택되어 있어도 그 부위가 가려지지 않을 수 있다.
  해당 항목까지 확실히 가리려면 `전체`를 사용한다. 배경 항목은 이 설정과 무관하게
  박스 영역만 가린다.
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
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
