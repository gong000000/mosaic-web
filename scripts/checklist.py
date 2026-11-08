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

import src.preview as pv
import src.vlm_package as vp
from src.config import CANDS, canon_category
from src.models import get_model
from src.preview import analyze_video
from src.pipeline import mosaic_video
from src.vlm_package import prompt_categories


def fake_vlm(frame, phrases, api_key, model="gemini-3-flash-preview", timeout=30,
             query_raw="", include_all=True, focus=None, base_url=None,
             **_kwargs):
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


def empty_vlm(frame, phrases, api_key, model="gemini-3-flash-preview", timeout=30,
              query_raw="", include_all=True, focus=None, base_url=None,
              **_kwargs):
    return []


def _read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"프레임을 읽지 못함: {path} ({idx})")
    return frame


def _box_diff(a, b, box):
    x1, y1, x2, y2 = box
    ca = a[y1:y2, x1:x2].astype(np.int16)
    cb = b[y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(ca - cb).mean())


def name_tag_vlm(frame, phrases, api_key, model="gemini-3-flash-preview", timeout=30,
                 query_raw="", include_all=True, focus=None, base_url=None,
                 **_kwargs):
    """사람이 없는 영상에서도 명찰 박스를 돌려주는 스텁.

    YOLO가 사람을 찾지 못해 트랙에 매칭되지 않는 경우(tid=-1)에도
    절대 좌표 fallback으로 가려지는지 확인하는 데 쓴다.
    """
    h, w = frame.shape[:2]
    return [{
        "category": "name_tag",
        "label_ko": "명찰",
        "xyxy": [int(120 * w / 320), int(90 * h / 240),
                 int(200 * w / 320), int(150 * h / 240)],
        "conf": 0.9,
        "matches_prompt": True,
        "region_type": "person",
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
    # due 이전 프레임에는 체크리스트 박스가 새지 않는다.
    due_box = [{"xyxy": [1, 1, 8, 8], "due": 6}]
    assert _sample_boxes({3: due_box}, 3, frame_idx=0) == []
    assert _sample_boxes({3: due_box}, 3, frame_idx=6) == due_box
    # 보간은 앞/뒤 키프레임의 같은 객체를 따라간다.
    a = [{"xyxy": [0, 0, 10, 10]}]
    b = [{"xyxy": [10, 0, 20, 10]}]
    mid = _sample_boxes({0: a, 1: b}, 0, frame_idx=1,
                        stride=2, interpolate=True)
    assert mid and mid[0]["xyxy"] == [5, 0, 15, 10], mid

    tmp = tempfile.mkdtemp(prefix="mosaic_check_")
    src = os.path.join(tmp, "in.mp4")
    out = os.path.join(tmp, "out.mp4")
    # 단색 화면은 모자이크를 해도 같은 색이라 검증이 안 된다. 저대비
    # 고주파 텍스처를 써서 실제 픽셀이 바뀌는지 확인한다.
    rng = np.random.default_rng(7)
    tex = (150 + rng.integers(0, 60, size=(240, 320, 3))).astype(np.uint8)
    w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(12):
        w.write(tex)
    w.release()

    vp.detect_with_vlm = fake_vlm
    pv.detect_with_vlm = fake_vlm
    model = get_model("yolo11n-seg.pt")

    anno, _persons, items = analyze_video(
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

    frame = _read_frame(out, 0)
    src0 = _read_frame(src, 0)
    assert _box_diff(frame, src0, [100, 120, 220, 180]) > 8, \
        "배경 영역이 가려지지 않음"

    # 체크리스트에서 VLM이 찾은 박스를 priors로 넘기면, 처리 단계 VLM이
    # 같은 프레임에서 놓쳐도 체크한 항목은 계속 가려져야 한다.
    priors = {"license_plate": cats["license_plate"]["samples"]}
    vp.detect_with_vlm = empty_vlm
    pv.detect_with_vlm = empty_vlm
    out3 = os.path.join(tmp, "out3.mp4")
    stats3 = mosaic_video(
        src, out3, targets=["license_plate"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="차량번호판을 가려줘", api_key="fake-key",
        region_priors=priors)
    assert stats3.get("_regions", 0) >= 1, stats3
    frame3 = _read_frame(out3, 11)
    src11 = _read_frame(src, 11)
    assert _box_diff(frame3, src11, [100, 120, 220, 180]) > 8, \
        "체크리스트 priors가 렌더되지 않음"
    vp.detect_with_vlm = fake_vlm
    pv.detect_with_vlm = fake_vlm

    # 체크 해제한 카테고리는 프롬프트에 적혀 있어도 렌더하지 않아야 한다.
    out2 = os.path.join(tmp, "out2.mp4")
    stats2 = mosaic_video(
        src, out2, targets=["face"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="차량번호판을 가려줘", api_key="fake-key")
    assert stats2.get("_regions", 0) == 0, stats2
    frame2 = _read_frame(out2, 0)
    assert _box_diff(frame2, src0, [100, 120, 220, 180]) < 4, \
        "체크 해제한 번호판이 렌더됨"

    # 사람이 없는 영상에서도 명찰 박스는 tid=-1 절대 좌표 fallback으로 가려져야 한다.
    vp.detect_with_vlm = name_tag_vlm
    out4 = os.path.join(tmp, "out4.mp4")
    mosaic_video(
        src, out4, targets=["name_tag"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="명찰", api_key="fake-key")
    frame4 = _read_frame(out4, 0)
    assert _box_diff(frame4, src0, [120, 90, 200, 150]) > 8, \
        "사람 트랙에 매칭되지 않은 명찰이 가려지지 않음"

    # 체크리스트의 사람 부착(명찰) 박스도 처리 단계 초기값으로 쓸 수 있고,
    # due 프레임 이전에는 그리지 않아야 한다. 처리 단계 VLM은 비워 둔다.
    vp.detect_with_vlm = empty_vlm
    pv.detect_with_vlm = empty_vlm
    person_priors = {"name_tag": {6: [[120, 90, 200, 150]]}}
    out5 = os.path.join(tmp, "out5.mp4")
    stats5 = mosaic_video(
        src, out5, targets=["name_tag"], model_name="yolo11n-seg.pt",
        imgsz=320, ultra=False, keyframe_stride=2,
        query_raw="명찰", api_key="fake-key",
        person_priors=person_priors)
    assert "name_tag" in stats5.get("_cats", []), stats5
    early = _read_frame(out5, 0)
    late = _read_frame(out5, 8)
    assert _box_diff(early, _read_frame(src, 0),
                     [120, 90, 200, 150]) < 4, \
        "person prior가 due 이전 프레임에 그려짐"
    assert _box_diff(late, _read_frame(src, 8),
                     [120, 90, 200, 150]) > 8, \
        "체크리스트 person prior가 렌더되지 않음"

    for p in (src, out, out2, out3, out4, out5):
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
