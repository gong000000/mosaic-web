"""강건성 검사. 엣지 입력에도 죽지 않으면 통과. 전체 처리 없이 빠르게.
실행: .\\.venv\\Scripts\\python.exe scripts\\robustness.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
YTDOWN = r"C:\Users\gong0000\Downloads\YTDown.com_YouTube_Media__VuJA-VQRcY_Central-Cee-Doja-Official-Music-Video_002_720p.mp4"
TMP = os.path.join(os.environ.get("TEMP", "."), "mosaic_robust_black.mp4")


def check(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__} {e}")
        return 1
    print(f"OK {name}")
    return 0


def t_preview_no_person():
    import cv2
    from src.models import get_model
    from src.preview import analyze_first_frame_items
    if not os.path.exists(YTDOWN):
        return
    cap = cv2.VideoCapture(YTDOWN)
    ok, fr = cap.read()
    cap.release()
    assert ok
    anno, persons, items = analyze_first_frame_items(
        fr, get_model("yolo11n-seg.pt"), 0.20, 480,
        query_raw="초록", targets=["green shirt"])
    assert isinstance(anno, type(fr))
    assert isinstance(persons, list) and isinstance(items, list)


def t_preview_closeup():
    import cv2
    from src.models import get_model
    from src.preview import analyze_first_frame_items
    if not os.path.exists(FACE):
        return
    cap = cv2.VideoCapture(FACE)
    ok, fr = cap.read()
    cap.release()
    assert ok
    anno, persons, items = analyze_first_frame_items(
        fr, get_model("yolo11m-seg.pt"), 0.20, 640,
        query_raw="얼굴", targets=["face"])
    assert isinstance(anno, type(fr)) and isinstance(persons, list)
    assert len(persons) >= 1
    assert all("detail" in p for p in persons)
    assert any(it["category"] == "face" for it in items)


def t_black_video():
    import cv2
    import numpy as np
    from src.models import get_model
    from src.preview import analyze_video
    os.makedirs(os.path.dirname(TMP), exist_ok=True)
    vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100))
    for _ in range(5):
        vw.write(np.zeros((100, 100, 3), np.uint8))
    vw.release()
    anno, persons, items = analyze_video(
        TMP, get_model("yolo11n-seg.pt"), 0.20, 480,
        query_raw="아무것도", targets=["face"])
    assert anno is None or anno.shape[2] == 3
    assert persons == []
    try:
        os.remove(TMP)
    except OSError:
        pass


def t_world_empty_targets():
    import numpy as np
    from src.world import get_world, phrases_for_categories
    assert phrases_for_categories([]) == []
    frame = np.zeros((64, 64, 3), np.uint8)
    assert get_world().ground(frame, []) == []


def t_vlm_parse_and_normalize():
    from src.vlm_package import _normalize_item, _parse_json_array
    assert _parse_json_array("") == []
    assert _parse_json_array("```json\n[{\"a\": 1}]\n```") == [{"a": 1}]
    assert _parse_json_array("설명 [1, 2] 뒤") == [1, 2]
    item = {"category": "license_plate", "bbox": [0.1, 0.1, 0.5, 0.5],
            "matches_prompt": "false", "region_type": "REGION", "conf": "0.9"}
    norm = _normalize_item(item, 100, 200, 1.0, "번호판")
    assert norm is not None
    assert norm["category"] == "license_plate"
    assert norm["matches_prompt"] is False
    assert norm["region_type"] == "region"
    assert abs(norm["conf"] - 0.9) < 1e-6
    assert _normalize_item({"bbox": [1, 2, 3]}, 100, 100, 1.0, "") is None
    assert _normalize_item({"bbox": ["a", 1, 2, 3]}, 100, 100, 1.0, "") is None


def main():
    fails = 0
    fails += check("preview-0person", t_preview_no_person)
    fails += check("preview-closeup", t_preview_closeup)
    fails += check("black-video", t_black_video)
    fails += check("world-empty-targets", t_world_empty_targets)
    fails += check("vlm-parse-normalize", t_vlm_parse_and_normalize)
    print("ROBUSTNESS_PASS" if fails == 0 else f"ROBUSTNESS_FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
