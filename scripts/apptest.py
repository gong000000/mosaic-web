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


def _click_process(at):
    buttons = [b for b in at.button if b.label == "모자이크 시작"]
    assert buttons, "모자이크 시작 버튼 없음"
    assert not buttons[0].disabled, "모자이크 시작 버튼이 비활성"
    buttons[0].click()
    at.run(timeout=900)


def _cleanup(at):
    res = at.session_state["result"]
    from src.history import JobHistory
    JobHistory().delete(res["out_name"])


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
    at.sidebar.radio[0].set_value("빠름")
    at.run(timeout=900)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["preview_items"]
    cats = {it["category"] for it in items}
    assert "face" in cats, cats
    assert "얼굴" in _checkboxes(at), _checkboxes(at)
    # 2번 프롬프트 입력값이 실제 VLM 호출까지 전달되는지 확인한다.
    assert "얼굴만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    _click_process(at)
    assert not [e for e in at.error], [e.value for e in at.error]
    res = at.session_state["result"]
    assert res["selected_cats"] == ["face"], res["selected_cats"]
    assert res["final_n"] >= 1, res
    assert os.path.exists(res["out_path"]) and os.path.getsize(res["out_path"]) > 1000
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
    at.sidebar.radio[0].set_value("빠름")
    at.run(timeout=900)

    assert "분석 실패" not in _warnings(at), _warnings(at)
    items = at.session_state["preview_items"]
    cats = {it["category"] for it in items}
    assert "license_plate" in cats, cats
    plate = next(it for it in items if it["category"] == "license_plate")
    assert plate["type"] == "region"
    assert "차량번호판" in _checkboxes(at), _checkboxes(at)
    assert "차량번호판만 모자이크해줘" in _SEEN_QUERIES, _SEEN_QUERIES

    _click_process(at)
    assert not [e for e in at.error], [e.value for e in at.error]
    res = at.session_state["result"]
    assert res["selected_cats"] == ["license_plate"], res["selected_cats"]
    assert res["region_n"] >= 1, res
    assert os.path.exists(res["out_path"]) and os.path.getsize(res["out_path"]) > 1000
    _cleanup(at)
    print("OK plate-flow")


def main():
    missing = [p for p in (FACE, STREET) if not os.path.exists(p)]
    if missing:
        print("SKIP 파일 없음: " + ", ".join(missing))
        return 0
    run_face()
    run_plate()
    print("APPTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
