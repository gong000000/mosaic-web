"""체크리스트/카테고리 파이프라인 검사.
VLM 호출을 모의 응답으로 바꿔 프롬프트 매칭, 배경 영역 모자이크를 확인한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\checklist.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

import src.vlm_package as vp
from src.config import CANDS, canon_category
from src.models import get_model
from src.preview import analyze_video
from src.pipeline import mosaic_video
from src.vlm_package import prompt_categories


def fake_vlm(frame, phrases, api_key, model="gpt-4o-mini", timeout=30,
             query_raw="", include_all=True, **_kwargs):
    h, w = frame.shape[:2]
    return [{
        "category": "license_plate",
        "label_ko": "차량번호판",
        "xyxy": [int(100 * w / 320), int(120 * h / 240),
                 int(220 * w / 320), int(180 * h / 240)],
        "conf": 0.9,
        "matches_prompt": "번호판" in str(query_raw),
        "region_type": "region",
        "source": "vlm",
    }]


def main():
    assert prompt_categories("얼굴과 차량번호판을 가려줘") == {"face", "license_plate"}
    assert canon_category("name tag") == "name_tag"
    assert canon_category("주소·간판") == "address"
    # 영문 별칭은 단어 경계로만 잡는다. ('surface'를 face로 보면 안 됨)
    assert prompt_categories("surface") == set()
    assert canon_category("surface") == "surface"
    assert canon_category("human face") == "face"

    # 샘플 이전 프레임에는 미래 탐지 박스가 새면 안 된다.
    from src.pipeline import _sample_boxes
    boxes = [{"xyxy": [1, 1, 2, 2]}]
    assert _sample_boxes({2: boxes}, 0) == []
    assert _sample_boxes({2: boxes}, 1) == []
    assert _sample_boxes({2: boxes}, 2) == boxes
    assert _sample_boxes({2: boxes}, 5) == boxes
    assert _sample_boxes({}, 0) == []

    tmp = tempfile.mkdtemp(prefix="mosaic_check_")
    src = os.path.join(tmp, "in.mp4")
    out = os.path.join(tmp, "out.mp4")
    w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(12):
        w.write(np.full((240, 320, 3), 180, np.uint8))
    w.release()

    vp.detect_with_vlm = fake_vlm
    model = get_model("yolo11n-seg.pt")

    anno, persons, items = analyze_video(
        src, model, 0.20, 320, query_raw="차량번호판을 가려줘",
        targets=CANDS, api_key="fake-key", samples=2)
    assert anno is not None
    cats = {it["category"]: it for it in items}
    assert "license_plate" in cats, cats
    assert cats["license_plate"]["type"] == "region"
    assert cats["license_plate"]["prompt_match"] is True

    stats = mosaic_video(
        src, out, targets=["license_plate"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="차량번호판을 가려줘", api_key="fake-key")
    assert stats.get("_regions", 0) >= 1, stats
    assert os.path.exists(out) and os.path.getsize(out) > 1000

    cap = cv2.VideoCapture(out)
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame[150, 160].tolist() != [180, 180, 180], "배경 영역이 가려지지 않음"

    # 체크 해제한 카테고리는 프롬프트에 적혀 있어도 렌더하지 않아야 한다.
    cap = cv2.VideoCapture(src)
    ok, src_frame = cap.read()
    cap.release()
    assert ok
    base = [int(v) for v in src_frame[150, 160]]
    out2 = os.path.join(tmp, "out2.mp4")
    stats2 = mosaic_video(
        src, out2, targets=["face"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="차량번호판을 가려줘", api_key="fake-key")
    assert stats2.get("_regions", 0) == 0, stats2
    cap = cv2.VideoCapture(out2)
    ok, frame2 = cap.read()
    cap.release()
    assert ok
    assert all(abs(int(v) - b) <= 8 for v, b in zip(frame2[150, 160], base)), \
        (base, frame2[150, 160].tolist())

    for p in (src, out, out2):
        try:
            os.remove(p)
        except Exception:
            pass
    try:
        os.rmdir(tmp)
    except Exception:
        pass
    print("CHECKLIST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
