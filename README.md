# 프롬프트 기반 선택적 모자이크 웹

## 구조
```
app.py                  Streamlit 진입점 (루트 유지가 관례)
src/                    라이브러리
  config.py             옷목록/프리셋/모드
  prompts.py            한국어 프롬프트 파싱
  clothing.py           HSV 옷 판정(야간 자동 보정)
  detect.py             최소 크기·겹침 박스 제거
  preview.py            첫 화면 분석(미리보기 공용)
  render.py             모자이크/ffmpeg/엄격 렌더
  models.py             YOLO 캐시 (weights/ 우선)
  pipeline.py           2패스 추적+렌더(MosaicJob)
  select.py             판정 인터페이스(휴리스틱+월드 하이브리드)
  world.py              YOLO-World 개방형 검출(안경·우산·모자·안전모, 로컬)
  vlm_package.py        첫 프레임 포장(전송 없음, API형 VLM 자리)
  history.py            outputs/ 작업 기록
configs/                추적기 설정
weights/                모델(.pt, 깃 제외. yolo11n/m/x-seg + yolov8s-world)
outputs/                결과물+작업기록
scripts/
  regression.py         face.mp4 4타겟 고정 회귀
  evaluate.py           첫 프레임 점수표 (밤 영상 보기용)
docs/
  night-street-table.md 밤 영상 HSV 쏠림 기록
```

## 실행 (Python 3.12 고정. 시스템 3.13 `py`로는 안 된다)
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -m streamlit run app.py
```
