"""Streamlit AppTest E2E 검사.

실제 VLM 대신 스텁을 주입해 업로드 → 체크리스트 → 모자이크 → 결과 저장
흐름을 검증한다. 실제 API 키/네트워크는 사용하지 않는다.
실행: .\\.venv\\Scripts\\python.exe scripts\\apptest.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def _cleanup(at):
    res = at.session_state.get("result") or {}
    fname = res.get("out_name") or ""
    if not fname:
        return
    from src.history import JobHistory
    JobHistory().delete(fname)


def _run_until_result(at, timeout=900):
    """분석+미리보기가 끝날 때까지 앱을 실행한다."""
    at.run(timeout=timeout)
    assert "preview_items" in at.session_state, "체크리스트 항목 없음"


def run_face():
    _SEEN_QUERIES.clear()
    _patch(_face_stub)
    at = _open_app()
    at.run()
    at.sidebar.file_uploader[0].upload(
        os.path.basename(FACE), open(FACE, "rb").read(), "video/mp4")
    at.run()
    assert "VLM API 키" in _warnings(at), _warnings(at)

    at.sidebar.text_input(key="vlm_key_OpenAI").input("fake-key")
    at.sidebar.text_area(key="prompt_txt").input("얼굴만 모자이크해줘")
    _run_until_result(at)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["preview_items"]
    cats = {it["category"] for it in items}
    assert "face" in cats, cats
    assert "얼굴" in _checkboxes(at), _checkboxes(at)
    # 2번 프롬프트 입력값이 실제 VLM 호출까지 전달되는지 확인한다.
    assert "얼굴만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    assert not [e for e in at.error], [e.value for e in at.error]
    _cleanup(at)
    print("OK face-flow")


def run_plate():
    _SEEN_QUERIES.clear()
    _patch(_plate_stub)
    at = _open_app()
    at.run()
    at.sidebar.file_uploader[0].upload(
        os.path.basename(STREET), open(STREET, "rb").read(), "video/mp4")
    at.sidebar.text_input(key="vlm_key_OpenAI").input("fake-key")
    at.sidebar.text_area(key="prompt_txt").input("차량번호판만 모자이크해줘")
    _run_until_result(at)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["preview_items"]
    cats = {it["category"] for it in items}
    assert "license_plate" in cats, cats
    plate = next(it for it in items if it["category"] == "license_plate")
    assert plate["type"] == "region"
    assert "차량번호판" in _checkboxes(at), _checkboxes(at)
    assert "차량번호판만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    assert not [e for e in at.error], [e.value for e in at.error]
    _cleanup(at)
    print("OK plate-flow")


def run_prompt_switch():
    """프롬프트 변경 시 VLM 재분석 없이 체크리스트가 즉시 갱신되는지."""
    _SEEN_QUERIES.clear()
    _patch(_face_plate_stub)
    at = _open_app()
    at.run()
    at.sidebar.file_uploader[0].upload(
        os.path.basename(FACE), open(FACE, "rb").read(), "video/mp4")
    at.sidebar.text_input(key="vlm_key_OpenAI").input("fake-key")
    at.sidebar.text_area(key="prompt_txt").input("얼굴과 차량번호판 모자이크")
    _run_until_result(at)

    items = at.session_state["preview_items"]
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

    # 미리보기 화면의 프롬프트를 '얼굴만'으로 바꾼다. 체크박스 키에
    # 프롬프트 필터 ID가 포함되므로 번호판 체크는 자동으로 사라진다.
    at.text_area(key="prompt_edit").input("얼굴만 모자이크")
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
    run_prompt_switch()
    print("APPTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
