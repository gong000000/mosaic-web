"""Streamlit AppTest E2E 검사.

실제 VLM 대신 스텁을 주입해
업로드 → 분석 목록 → '모자이크 시작' → 모자이크 영상 미리보기/다운로드
흐름을 검증한다. 실제 API 키/네트워크는 사용하지 않는다.
실행: .\\.venv\\Scripts\\python.exe scripts\\apptest.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 실제 .env 키를 읽지 않고 가짜 키로 앱의 자동 키 경로만 검사한다.
# (config._load_dotenv는 MOSAIC_SKIP_DOTENV=1이면 .env를 읽지 않는다.
#  VLM 호출은 모두 스텁으로 바뀌므로 네트워크를 타지 않는다.)
os.environ["MOSAIC_SKIP_DOTENV"] = "1"
for _name in ("GOOGLE_API_KEY", "VLM_API_KEY"):
    os.environ[_name] = ""
os.environ["GEMINI_API_KEY"] = "fake-key"

from streamlit.testing.v1 import AppTest

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
STREET = r"C:\Users\gong0000\Downloads\street.mp4"


_SEEN_QUERIES = []


def _face_stub(frame, phrases, api_key, **kwargs):
    _SEEN_QUERIES.append(str(kwargs.get("query_raw") or ""))
    h, w = frame.shape[:2]
    return [{
        "category": "face", "label_ko": "얼굴",
        "xyxy": [int(w * 0.12), int(h * 0.12),
                 int(w * 0.32), int(h * 0.35)],
        "conf": 0.9, "matches_prompt": True,
        "region_type": "person", "source": "vlm",
    }]


def _plate_stub(frame, phrases, api_key, **kwargs):
    _SEEN_QUERIES.append(str(kwargs.get("query_raw") or ""))
    h, w = frame.shape[:2]
    return [{
        "category": "license_plate", "label_ko": "차량번호판",
        "xyxy": [int(w * 0.42), int(h * 0.62),
                 int(w * 0.55), int(h * 0.68)],
        "conf": 0.9, "matches_prompt": True,
        "region_type": "region", "source": "vlm",
    }]


def _face_plate_stub(frame, phrases, api_key, **kwargs):
    _SEEN_QUERIES.append(str(kwargs.get("query_raw") or ""))
    h, w = frame.shape[:2]
    return [{
        "category": "face", "label_ko": "얼굴",
        "xyxy": [int(w * 0.12), int(h * 0.12),
                 int(w * 0.32), int(h * 0.35)],
        "conf": 0.9, "matches_prompt": True,
        "region_type": "person", "source": "vlm",
    }, {
        "category": "license_plate", "label_ko": "차량번호판",
        "xyxy": [int(w * 0.42), int(h * 0.62),
                 int(w * 0.55), int(h * 0.68)],
        "conf": 0.9, "matches_prompt": True,
        "region_type": "region", "source": "vlm",
    }]


def _patch(fn):
    import src.preview as pv
    import src.vlm_package as vp
    vp.detect_with_vlm = fn
    pv.detect_with_vlm = fn


def _open_app():
    return AppTest.from_file(
        os.path.join(os.path.dirname(__file__), "..", "app.py"),
        default_timeout=900)


def _warnings(at):
    return " ".join(str(w.value) for w in at.warning)


def _checkboxes(at):
    return " | ".join(str(c.label) for c in at.checkbox)


def _keys(at):
    elements = list(at.button) + list(at.download_button)
    return [str(getattr(el, "key", "")) for el in elements]


def _cleanup(at):
    out = at.session_state.get("render_out") or ""
    if not out:
        return
    from src.history import JobHistory
    JobHistory().delete(os.path.basename(out))


def _run_until_analysis(at, timeout=900):
    at.run(timeout=timeout)
    analysis = at.session_state.get("analysis")
    assert isinstance(analysis, dict), "분석 상태 없음"
    assert analysis.get("ok"), analysis.get("err")


def _click_start(at):
    """'모자이크 시작'을 누르고 실제 렌더가 끝날 때까지 실행한다."""
    buttons = [b for b in at.button if b.key == "start_mosaic"]
    assert buttons, "모자이크 시작 버튼 없음: " + str(_keys(at))
    buttons[0].click()
    at.run(timeout=1800)
    out = at.session_state.get("render_out") or ""
    assert out and os.path.exists(out), "렌더 결과 파일 없음"
    assert os.path.getsize(out) > 1024, "렌더 결과가 비어 있음"
    return out


def _upload_and_analyze(at, video, prompt, stub):
    _patch(stub)
    at.run()
    # 프롬프트를 먼저 정하고 업로드해야 분석 호출에 프롬프트가 들어간다.
    at.sidebar.text_area(key="prompt_txt").input(prompt)
    at.run()
    at.sidebar.file_uploader[0].upload(
        os.path.basename(video), open(video, "rb").read(), "video/mp4")
    _run_until_analysis(at)

    # 분석만으로는 렌더하지 않는다(자동 모자이크 금지).
    assert not at.session_state.get("render_out"), "분석만으로 자동 렌더됨"
    assert "download_mosaic" not in _keys(at), "다운로드 버튼이 미리 노출됨"


def run_face():
    _SEEN_QUERIES.clear()
    at = _open_app()
    _upload_and_analyze(at, FACE, "얼굴만 모자이크해줘", _face_stub)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["analysis"]["items"]
    cats = {it["category"] for it in items}
    assert "face" in cats, cats
    assert "얼굴" in _checkboxes(at), _checkboxes(at)
    assert "얼굴만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    out = _click_start(at)
    assert at.session_state.get("render_sig"), "렌더 서명 없음"
    assert "download_mosaic" in _keys(at), _keys(at)
    assert not [e for e in at.error], [e.value for e in at.error]
    print(f"OK face-flow ({os.path.getsize(out) // 1024} KB)")
    _cleanup(at)


def run_plate():
    _SEEN_QUERIES.clear()
    at = _open_app()
    _upload_and_analyze(at, STREET, "차량번호판만 모자이크해줘", _plate_stub)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["analysis"]["items"]
    cats = {it["category"] for it in items}
    assert "license_plate" in cats, cats
    plate = next(it for it in items if it["category"] == "license_plate")
    assert plate["type"] == "region"
    assert "차량번호판" in _checkboxes(at), _checkboxes(at)
    assert "차량번호판만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    out = _click_start(at)
    assert "download_mosaic" in _keys(at), _keys(at)
    assert not [e for e in at.error], [e.value for e in at.error]
    print(f"OK plate-flow ({os.path.getsize(out) // 1024} KB)")
    _cleanup(at)


def run_missing_region():
    """샘플에서 못 찾은 배경 항목은 선택할 수 없고, 얼굴은 선택된다."""
    at = _open_app()
    _upload_and_analyze(at, FACE, "얼굴과 차량번호판 모자이크", _face_stub)

    plate_cb = next((c for c in at.checkbox
                     if "차량번호판" in str(c.label)), None)
    assert plate_cb is not None, "못 찾은 요청 항목이 목록에 없음"
    assert "못 찾음" in str(plate_cb.label), plate_cb.label
    face_cb = next(c for c in at.checkbox if "얼굴" in str(c.label))
    assert getattr(plate_cb.proto, "disabled", False), "못 찾은 배경 항목이 활성"
    assert not plate_cb.value, "못 찾은 배경 항목이 선택됨"
    assert face_cb.value, "얼굴이 기본 선택되지 않음"
    assert not [e for e in at.error], [e.value for e in at.error]
    _cleanup(at)
    print("OK missing-region")


def run_prompt_switch():
    """프롬프트 변경 시 VLM 재분석 없이 체크리스트가 즉시 갱신되는지."""
    _SEEN_QUERIES.clear()
    at = _open_app()
    _upload_and_analyze(at, FACE, "얼굴과 차량번호판 모자이크",
                        _face_plate_stub)

    items = at.session_state["analysis"]["items"]
    cats = {it["category"] for it in items}
    assert {"face", "license_plate"} <= cats, cats
    first = set()
    for c in at.checkbox:
        lab = str(c.label)
        if "얼굴" in lab and c.value:
            first.add("face")
        if "차량번호판" in lab and c.value:
            first.add("license_plate")
    assert first == {"face", "license_plate"}, first
    assert "차량번호판" in _checkboxes(at), _checkboxes(at)

    calls_before = len(_SEEN_QUERIES)
    # 사이드바 프롬프트를 '얼굴만'으로 바꾼다. 체크박스 키에 프롬프트
    # 필터 ID가 포함되므로 번호판 체크는 자동으로 사라지고, VLM은 다시
    # 부르지 않는다(체크리스트만 갱신).
    at.sidebar.text_area(key="prompt_txt").input("얼굴만 모자이크")
    at.run()
    second = set()
    for c in at.checkbox:
        lab = str(c.label)
        if "얼굴" in lab and c.value:
            second.add("face")
        if "차량번호판" in lab and c.value:
            second.add("license_plate")
    assert second == {"face"}, second
    plate_boxes = [c for c in at.checkbox if "차량번호판" in str(c.label)]
    assert plate_boxes, _checkboxes(at)
    assert not any(c.value for c in plate_boxes), _checkboxes(at)
    face_boxes = [c for c in at.checkbox if "얼굴" in str(c.label)]
    assert face_boxes and face_boxes[0].value, _checkboxes(at)
    assert len(_SEEN_QUERIES) == calls_before, "프롬프트 변경에 VLM 재호출됨"

    assert not [e for e in at.error], [e.value for e in at.error]
    _cleanup(at)
    print("OK prompt-switch")


def main():
    missing = [p for p in (FACE, STREET) if not os.path.exists(p)]
    if missing:
        print("SKIP 파일 없음: " + ", ".join(missing))
        return 0
    run_face()
    run_plate()
    run_missing_region()
    run_prompt_switch()
    print("APPTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
