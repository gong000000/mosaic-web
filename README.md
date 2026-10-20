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

## 실행 (팀원 공유 절차)

1. 이 저장소를 clone 또는 다운로드 받는다.
2. Python 3.12 가상환경을 만든다. (팀원 각자 로컬에서 생성, git에 올리지 않음)
   ```
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. 필요한 라이브러리를 설치한다.
   ```
   pip install -r requirements.txt
   ```
   (GPU가 없는 저사양 PC는 `requirements-cpu.txt`를 대신 사용)
4. 실행한다.
   ```
   py -m streamlit run app.py
   ```

중요:
- `.venv/`는 이미 `.gitignore`에 있어 올리지 않는다. 팀원 각자 2번에서 생성한다.
- `src/retention.py` 같은 `src/` 안 파일은 직접 실행하지 않는다. `app.py`를 통해 streamlit으로 실행한다.
- 첫 실행에 YOLO-World 가중치(약 370MB)를 자동으로 받아온다. 로컬에서만 처리되며 영상은 외부로 전송되지 않는다.
