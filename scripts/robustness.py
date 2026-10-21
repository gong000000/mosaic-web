"""강건성 검사. 엣지 입력에도 죽지 않으면 통과. 전체 처리 없이 빠르게.
실행: .\\.venv\\Scripts\\python.exe scripts\\robustness.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
YTDOWN = r"C:\Users\gong0000\Downloads\YTDown.com_YouTube_Media__VuJA-VQRcY_Central-Cee-Doja-Official-Music-Video_002_720p.mp4"


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
    from src.preview import analyze_first_frame
    if not os.path.exists(YTDOWN):
        return
    cap = cv2.VideoCapture(YTDOWN)
    ok, fr = cap.read()
    cap.release()
    assert ok
    anno, persons = analyze_first_frame(fr, get_model("yolo11n-seg.pt"),
                                        0.20, 480, ["green shirt"], 0.04,
                                        query_raw="초록")
    assert isinstance(anno, type(fr)) and isinstance(persons, list)


def t_preview_closeup():
    import cv2
    from src.models import get_model
    from src.preview import analyze_first_frame
    cap = cv2.VideoCapture(r"C:\Users\gong0000\Downloads\Video Project 6.mp4")
    ok, fr = cap.read()
    cap.release()
    assert ok
    anno, persons = analyze_first_frame(fr, get_model("yolo11m-seg.pt"),
                                        0.20, 640, ["white sleeveless vest"], 0.04,
                                        query_raw="흰 나시")
    assert len(persons) >= 1
    assert all("detail" in p for p in persons)


def t_black_video():
    import cv2
    import numpy as np
    from src.preview import analyze_first_frame
    from src.models import get_model
    path = r"C:\Users\gong0000\AppData\Local\Temp\opencode\black.mp4"
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(10):
        out.write(np.zeros((240, 320, 3), np.uint8))
    out.release()
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    assert ok
    anno, persons = analyze_first_frame(fr, get_model("yolo11n-seg.pt"),
                                        0.20, 480, ["green shirt"], 0.04)
    assert persons == []
    os.remove(path)


def t_world_only_empty_targets():
    from src.select import HybridSelector
    import numpy as np
    sel = HybridSelector([], query_raw="모자 쓴 사람", world_mode="auto")
    assert sel.use_world
    crop = np.zeros((100, 60, 3), np.uint8)
    hit, _ = sel.judge(crop, None)
    assert hit is False


def t_vlm_package_empty():
    import numpy as np
    from src.vlm_package import build_first_frame_package
    pkg = build_first_frame_package(np.zeros((100, 100, 3), np.uint8), [], query_raw="x")
    assert pkg["persons"] == [] and pkg["first_frame_only"]


def main():
    fails = 0
    fails += check("preview-0person", t_preview_no_person)
    fails += check("preview-closeup", t_preview_closeup)
    fails += check("black-video", t_black_video)
    fails += check("world-empty-targets", t_world_only_empty_targets)
    fails += check("vlm-package-empty", t_vlm_package_empty)
    print("ROBUSTNESS_PASS" if fails == 0 else f"ROBUSTNESS_FAIL {fails}건")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
