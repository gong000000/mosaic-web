# 영상 속 개인정보 보호 모자이크 시스템

영상을 업로드하면 YOLO11-seg + YOLO-World(VLM) 기반으로 자동으로 개인정보(얼굴, 차량번호판, 주소, 차량동승자 등)를 탐지하고,
사용자가 선택한 항목에만 모자이크를 적용하는 웹 애플리케이션

## 구조

```
app.py                  메인 앱: 영상 업로드, 판정, 모자이크 처리 UI
src/                    라이브러리
  config.py             전역 설정 (개인정보 항목 CANDS, 한글 문구 KO, 프리셋 PRESETS, 출력 디렉터리 OUTPUT_DIR)
  prompts.py            한국어 지시문 파싱 (쉼표/와/과/랑/하고/및 분리, 유사어 매칭)
  detect.py             검출 후처리 (작은 박스 제거, 겹친 박스 필터링)
  preview.py            첫 화면 분석: 사람 탐지 + VLM/월드 판정, 미리보기 이미지 생성
  render.py             모자이크 렌더링 (모자이크/흐리게/검게), ffmpeg 오디오 믹싱, 구간 자르기
  models.py             YOLO 모델 로딩 캐시 (가중치 우선, 동일 모델 한 번만 로드)
  pipeline.py           2패스 파이프라인: 1패스 키프레임 투표, 2패스 전프레임 렌더
  select.py             판정 인터페이스: VLMSelector (API 키 필요), WorldSelector (로컬, 무료), PromptQuery
  world.py              YOLO-World 개방형 검출: 얼굴/제복/안전모/로고/옷 등 개체 검출, 사람 박스와 물건 박스 매칭
  vlm_package.py        VLM 기반 개인정보 탐지: OpenAI GPT-4o-mini 호출, base64 인코딩된 첫 프레임 전송, JSON 파싱
  history.py            작업 히스토리: outputs/jobs.json, JobHistory 클래스 (add/all_rows/delete)
  retention.py          보관 정책: 24시간 지난 결과 자동 삭제, 다운로드 시 삭제 옵션
configs/                추적기 설정 (bytetrack_stable.yaml)
weights/                모델 가중치 (yolo11n/m/x-seg.pt, yolov8s-world.pt)
outputs/                결과물 + 작업 기록 (jobs.json)
scripts/
  regression.py         회귀 테스트 (face.mp4 고정 타겟)
  evaluate.py           첫 프레임 점수표 (사람 수, 타겟별 점수)
  robustness.py         강건성 검사 (예외 입력)
  bench.py              속도 벤치마크 (모델·해상도별 FPS)
  smoke.py              설치 확인 (필수 라이브러리 import 테스트)
docs/                   (비어 있음)
```

### 주요 모듈 역할

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit UI: 영상 업로드, 지시문 입력, 대상 확인, 모자이크 처리, 결과 비교, 다운로드 |
| `src/config.py` | CANDS(개인정보 항목 목록), KO(한글 문구), PRESETS(예시), OUTPUT_DIR |
| `src/prompts.py` | 한국어 지시문 파싱: 쉼표/와/과/랑/하고/및 분리, 유사어 매칭 |
| `src/detect.py` | 검출 후처리: 작은 박스 제거, 겹친 박스 필터링 |
| `src/preview.py` | 첫 화면 분석: 사람 탐지 + VLM/월드 판정, 미리보기 이미지 생성 |
| `src/render.py` | 모자이크 렌더링: 모자이크/흐리게/검게, ffmpeg 오디오 믹싱, 구간 자르기 |
| `src/models.py` | YOLO 모델 로딩 캐시: weights/ 우선, 동일 모델 한 번만 로드 |
| `src/pipeline.py` | 2패스 파이프라인: 1패스 키프레임 투표, 2패스 전프레임 렌더 |
| `src/select.py` | 판정 인터페이스: VLMSelector (API 키 필요), WorldSelector (로컬, 무료), PromptQuery |
| `src/world.py` | YOLO-World 개방형 검출: 얼굴/제복/안전모/로고/옷 등 개체 검출, 사람 박스와 물건 박스 매칭 |
| `src/vlm_package.py` | VLM 기반 개인정보 탐지: OpenAI GPT-4o-mini 호출, base64 인코딩된 첫 프레임 전송, JSON 파싱 |
| `src/history.py` | 작업 히스토리: outputs/jobs.json, JobHistory 클래스 (add/all_rows/delete) |
| `src/retention.py` | 보관 정책: 24시간 지난 결과 자동 삭제, 다운로드 시 삭제 옵션 |

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
py -m streamlit run app.py --server.port 8501
```

브라우저에서 `http://localhost:8501`로 접속합니다.


## 기술 스택

- **Frontend**: Streamlit
- **Object Detection**: YOLO11-seg (사람 마스크)
- **Open-Vocabulary Detection**: YOLO-World (사람 이름, 옷, 액세서리 등)
- **VLM**: OpenAI GPT-4o-mini (선택사항, API 키 필요)
- **Multi-Object Tracking**: ByteTrack
- **Video Encoding**: H.264 via ffmpeg (imageio-ffmpeg)
- **언어**: Python 3.12

