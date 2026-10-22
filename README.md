# 영상 속 개인정보 보호 모자이크 시스템

영상을 업로드하고 프롬프트(예: "얼굴과 차량번호판을 모자이크해줘")를 입력하면,
VLM이 영상 프레임을 보고 **프롬프트가 지정한 항목**과 **추가 개인정보 후보**를 찾는다.
앱은 결과를 체크리스트로 보여주고, 사용자가 체크한 항목만 모자이크한다.

## 동작 방식

1. **사람 검출**: YOLO11-seg + ByteTrack으로 사람과 마스크를 추적한다.
2. **VLM 분석**: 앞/중간/끝 3개 샘플 프레임을 VLM에 보내 카테고리별 항목을 받는다.
   샘플 분석은 모든 카테고리를 후보로 찾고(`include_all`), 처리 단계는 체크한
   카테고리만 판정한다.
   - 프롬프트가 지정한 항목은 `프롬프트에서 찾은 항목`으로, 그 외 개인정보는
     `추가 개인정보 후보`로 묶는다.
   - 두 그룹 모두 기본으로 체크된다. 사용자는 가리지 않을 항목을 해제할 수 있다.
   - API 키가 없거나 VLM 호출이 실패하면 로컬 YOLO-World로 폴백한다.
   - 같은 업로드/프롬프트/설정이면 VLM을 다시 호출하지 않고 결과를 재사용한다.
3. **2패스 렌더**: 1패스에서 `(사람 트랙, 카테고리)`별 판정을 모아 확정하고,
   2패스에서 선택된 카테고리만 렌더한다.
   - 1패스와 2패스는 **모든 프레임**을 같은 설정으로 추적해 트랙 ID를 일치시킨다.
     VLM/World 탐지는 1패스의 키프레임에서만 한다. (프레임을 건너뛰며 추적하면
     패스마다 ID가 달라져 다른 사람을 가리는 문제가 있었다.)
   - 사람 카테고리(얼굴, 동승자, 명찰, 문신): 사람 마스크/얼굴 영역을 가린다.
   - 배경 카테고리(번호판, 주소·간판, 신분증, 휴대폰 화면): 사람과 무관한
     사각형 영역을 가린다. YOLO-World를 키프레임마다 돌려 박스를 갱신하고,
     VLM이 찾은 박스는 샘플 구간 동안 유지한다.

## 카테고리

| 키 | 표시 | 처리 |
|---|---|---|
| `face` | 얼굴 | 사람 |
| `passenger` | 차량동승자 | 사람 |
| `name_tag` | 명찰 | 사람 |
| `tattoo` | 문신 | 사람 |
| `license_plate` | 차량번호판 | 배경 영역 |
| `address` | 주소·간판 | 배경 영역 |
| `id_card` | 신분증·문서 | 배경 영역 |
| `phone_screen` | 휴대폰 화면 | 배경 영역 |

프롬프트에 없는 자유 형식 항목(예: "빨간 옷 입은 사람")은 VLM이 `matches_prompt`로
표시하며, 사람 박스에 붙으면 사람 항목으로 처리된다.

## 구조

```
app.py                  메인 앱: 업로드, 프롬프트/프리셋, 체크리스트, 모자이크, 다운로드
src/                    라이브러리
  config.py             카테고리/별칭/타입, 프리셋, 출력 경로
  detect.py             검출 후처리 (작은 박스 제거, 겹친 박스 필터링)
  preview.py            영상 샘플 분석: 사람 탐지 + 카테고리 목록(체크리스트)
  select.py             VLMSelector: VLM 탐지, 사람 박스 매칭, 얼굴 폴백
  vlm_package.py        프레임 base64 전송, JSON 파싱, YOLO-World 폴백
  world.py              YOLO-World 개방형 검출, 카테고리 매핑
  render.py             사람/배경 모자이크, ffmpeg 오디오 믹싱, 구간 자르기
  models.py             YOLO 모델 로딩 캐시
  pipeline.py           2패스 파이프라인: 카테고리별 투표, 선택 렌더
  history.py            작업 히스토리 (outputs/jobs.json)
  retention.py          24시간 보관 정책, 다운로드 시 삭제 옵션
configs/                추적기 설정 (bytetrack_stable.yaml)
weights/                모델 가중치 (yolo11n/m/x-seg.pt, yolov8s-world.pt)
outputs/                결과물 + 작업 기록 (jobs.json)
scripts/
  smoke.py              설치 확인 (필수 라이브러리 import)
  robustness.py         강건성 검사 (예외 입력)
  regression.py         회귀 테스트 (face.mp4 고정 결과)
  evaluate.py           첫 프레임 탐지 리포트
  checklist.py          체크리스트/배경 영역 모의 VLM 검사
  freeform.py           자유 형식 person 카테고리 검사 (모의 VLM)
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

## 개인정보 처리

- 업로드한 영상 파일은 서버(실행 PC)의 임시 폴더에서만 처리된다.
  서버 시작 시 24시간이 지난 업로드 임시 폴더를 자동으로 지운다.
- OpenAI API 키를 입력한 경우에만 VLM 판정에 쓰는 **영상 프레임(샘플 3장 + 처리 키프레임)**
  이 외부로 전송된다. 영상 파일 자체는 전송되지 않는다.
- 결과물은 `outputs/`에 저장되며 24시간이 지나면 자동 삭제된다.
  "받으면 결과 자동 삭제"를 켜면 다운로드 직후 삭제된다.

## 현재 한계

- 체크리스트는 앞/중간/끝 3개 샘플만 보고 만든다. 아주 짧게 등장하는 항목은
  목록에서 빠질 수 있다. 처리 단계에서는 키프레임마다 다시 탐지한다.
- 배경 영역은 키프레임 사이에서 박스를 유지하므로, 빠르게 움직이는 번호판은
  순간적으로 어긋날 수 있다.
- CPU 환경에서는 긴 영상을 처리하면 느리다. 데모용으로는 짧은 영상(30~60초)을 권장한다.
- `가리는 범위`(전체/상반신/얼굴만)는 사람 항목 전체에 하나의 값으로 적용된다.
  예를 들어 `얼굴만`을 고르면 명찰·문신이 선택되어 있어도 그 부위가 가려지지 않을 수 있다.
  해당 항목까지 확실히 가리려면 `전체`를 사용한다. 배경 항목은 이 설정과 무관하게 박스 영역만 가린다.
- 외부 URL 배포(HTTPS)는 아직 하지 않았다. 현재는 로컬 Streamlit 실행 단계다.

## 기술 스택

- **Frontend**: Streamlit
- **Object Detection / Segmentation**: YOLO11-seg
- **Open-Vocabulary Detection**: YOLO-World (VLM 없을 때 폴백 + 배경 영역 보조)
- **VLM**: OpenAI GPT-4o-mini (`src/config.py`의 `VLM_MODEL`에서 변경)
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
```
