# 영상 속 개인정보 보호 모자이크 시스템

영상에서 개인정보(얼굴, 번호판, 간판, 명찰, 신분증 등)를 찾아 체크리스트로 보여주고,
선택한 항목만 모자이크하는 웹 앱.

VLM이 영상 프레임을 보고 개인정보 항목과 위치를 찾는다. 업로드 후 별도 버튼 없이
분석이 시작되고, 결과 화면에서 타임라인/효과를 바꾸면 미리보기가 즉시 다시 그려지고,
모두 끝나면 미리보기 영상을 바로 다운로드한다. VLM API 키는 필수이며, 키가 없으면
분석을 시작하지 않는다. 로컬 검출로 VLM 판정을 대신하지 않는다.

## 동작 방식

1. **업로드 → 자동 분석**
   - 고르게 샘플링한 프레임 4장 + 번호판 확대(전체 6 / 프레임당 2) VLM 호출로
     체크리스트 항목/박스를 만든다. 프롬프트가 못 찾은 항목은 `못 찾음`으로 표시.
   - 업로드 내용(SHA-256)이 바뀌면 이전 분석 결과를 재사용하지 않는다.

2. **체크리스트 + 미리보기(영상)**
   - `st.video` 재생 미리보기 영상 + 아래 필름스트립 타임라인(균등 썸네일 + 현재 위치
     커서) + 타임라인 슬라이더. 슬라이더를 움직이면 미리보기 영상이 그 위치부터 재생.
   - `원본과 비교`, `가리는 범위`(얼굴만/상반신/전체), `모자이크 모양`(모자이크/흐리게/
     검게), 강도 변경 시 미리보기 즉시 갱신.
   - 미리보기 화면의 프롬프트 수정은 VLM 재호출 없이 체크리스트만 갱신.

3. **모자이크 → 다운로드**
   - 체크 항목에 대해 전체 영상 자동 렌더(2패스) 후 결과/미리보기 영상으로 다운로드.
   - 효과/범위/강도 변경 시 자동 재렌더 옵션(`설정을 바꾸면 전체 영상도 자동으로
     다시 만들기`)이 있으나 끌 수 있다.

## 고정 처리 설정

속도 모드/샘플 수/모델 선택 UI는 없앴다. blur.me 스튜디오처럼 항상 빠른 설정으로
동작하도록 아래 값을 고정한다.

| 설정 | 값 |
|---|---|
| 사람 검출 모델 | `yolo11n-seg.pt` |
| 추론 해상도 | `imgsz=480` |
| 체크리스트 샘플 | `sample_n=4` |
| 키프레임 간격 | `keyframe_stride=15` |
| 인코딩 프리셋 | `veryfast` |
| `ultra` | `False` |
| 처리 단계 VLM 호출 상한 | `vlm_call_limit=3` |
| 번호판 확대 호출 상한 | 영상 전체 6회, 프레임당 2회 |

## 카테고리

| 키 | 표시 | 처리 |
|---|---|---|
| `face` | 얼굴 | 사람 |
| `passenger` | 차량동승자 | 사람 |
| `name_tag` | 명찰 | 사람 (찾은 위치만) |
| `license_plate` | 차량번호판 | 배경 영역 |
| `address` | 주소·간판 | 배경 영역 |
| `id_card` | 신분증·문서 | 배경 영역 |
| `phone_screen` | 휴대폰 화면 | 배경 영역 |

## 구조

```
app.py                  메인 앱: 자동 분석/미리보기, 체크리스트, 필름스트립 영상,
                         효과 튜너, 프롬프트 동기화, 다운로드
src/                    라이브러리
  config.py             카테고리/별칭/타입, 프리셋, VLM 제공자, 출력 경로
  detect.py             검출 후처리 (작은 박스 제거, 겹친 박스 필터링, 박스 매칭)
  preview.py            영상 샘플 분석 + 미리보기 프레임/영상 빌드(보간/필름스트립)
  select.py             VLMSelector: VLM 탐지, 사람 박스 매칭
  vlm_package.py        프레임 base64 전송, 프롬프트, JSON/좌표/카테고리 파싱, 재시도
  vlm_budget.py         VLMBudget, 오류 분류
  render.py             사람/배경 모자이크, FFmpegFrameWriter, ffmpeg 믹싱·자르기·변환
  models.py             YOLO 모델 로딩 캐시 + 동시 로딩/추론 락
  pipeline.py           2패스 파이프라인: 병렬 VLM chunk, 카테고리별 투표, 선택 렌더,
                        priors, blanket 사람 마스크, 캐시, 단일 인코딩
  history.py / retention.py   작업 히스토리, 24시간 보관 정책
scripts/                smoke/robustness/regression/checklist/freeform/apptest/bench
configs/, weights/, outputs/    추적기 설정, 가중치, 결과물/기록
```

## VLM 호출·한도 동작

- 체크리스트 분석: 샘플 4 + 번호판 확대 최대 6 = 영상당 최대 약 10회.
- 처리 단계: 전체 키프레임에 고르게 배분해 최대 3회(`vlm_call_limit=3`).
- 영상당 VLM 최대 약 13회. 429(한도)/401(인증)/모델 없음은 **자동 재시도하지 않는다**.
- 1패스 결과는 `pass1_<hash>.pkl` 캐시(version 2). 효과 변경 시 재사용, 설정 변경 시
  `run_sig`로 자동 재렌더 판단.

## 실행 방법

### 1. Python 3.12 가상환경
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 라이브러리
```powershell
pip install -r requirements.txt
```
(저사양 PC는 `requirements-cpu.txt`)

### 3. 실행
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```
브라우저에서 `http://localhost:8501`.

### 4. VLM 설정 (OpenAI / Google Gemini) — API 키 필수, 없으면 분석 시작 안 함.

## 현재 한계

- 체크리스트는 고정 4개 샘플. 배경 영역은 키프레임 사이 보간, 순간 이동 항목은 어긋날 수
  있음. 체크리스트 분석 실패 시 로컬 검출 대신 안내. CPU 환경은 느리다(30~60초 영상 권장).
## 기술 스택

- Streamlit / YOLO11-seg / ByteTrack / VLM(OpenAI·Gemini) / FFmpeg 단일 패스 H.264
- Python 3.12

## 검사

```powershell
.\.venv\Scripts\python.exe scripts\smoke.py
.\.venv\Scripts\python.exe scripts\robustness.py
.\.venv\Scripts\python.exe scripts\checklist.py
.\.venv\Scripts\python.exe scripts\freeform.py
.\.venv\Scripts\python.exe scripts\regression.py
.\.venv\Scripts\python.exe scripts\apptest.py
```

2026-11-05 기준 6개 검사 모두 PASS. `apptest.py`는 스텁 VLM으로 업로드 → 자동 분석 →
미리보기/다운로드 흐름과 프롬프트 변경 시 체크 해제까지 검증한다.
