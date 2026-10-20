"""첫 화면 분석. 개인정보 보호용.
YOLO-seg로 사람을 찾고, VLM/월드로 개인정보 항목을 탐지한다."""
import cv2
import numpy as np
from .config import CANDS
from .detect import filter_contained, big_enough
from .select import VLMSelector, WorldSelector


def torso_roi(crop_bgr, mask_crop=None):
    """사람 마스크 영역 중 몸통 부분 추출."""
    h, w = crop_bgr.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.70)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    roi = crop_bgr[y1:y2, x1:x2].copy()
    m = None
    if mask_crop is not None:
        try:
            m = mask_crop[y1:y2, x1:x2]
            m = (m > 127).astype(np.uint8)
        except Exception:
            m = None
    if m is not None and m.sum() > 100:
        return roi, m
    h2, w2 = roi.shape[:2]
    return roi, np.ones((h2, w2), dtype=np.uint8)


def analyze_first_frame(frame, model, conf, imgsz,
                        query_raw="", targets=None,
                        api_key=None,
                        vlm_model="gpt-4o-mini", use_vlm=False,
                        world_mode="auto", frame_stride=1):
    """returns (anno_bgr, persons). persons: dict 리스트.
    
    VLM 또는 월드로 개인정보 항목을 첫 화면 1회에만 탐지한다.
    """
    pr = list(model.predict(source=frame, conf=conf, classes=[0], verbose=False, imgsz=imgsz))[0]
    if pr.boxes is None or len(pr.boxes) == 0:
        return frame.copy(), []

    boxes = pr.boxes.xyxy.cpu().numpy().astype(int)
    masks = None
    try:
        if pr.masks is not None:
            masks = pr.masks.data.cpu().numpy()
    except Exception:
        masks = None

    keep = filter_contained(boxes)
    anno = frame.copy()
    persons = []
    cand = []

    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        if not keep[bi]:
            continue
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        if not big_enough(x1, y1, x2, y2):
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mc = None
        if masks is not None and bi < len(masks):
            try:
                m = cv2.resize(masks[bi], (frame.shape[1], frame.shape[0]))
                m = (m > 0.5).astype("uint8") * 255
                mc = m[y1:y2, x1:x2]
            except Exception:
                mc = None
        cand.append({"key": len(cand), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "xyxy": [x1, y1, x2, y2], "crop": crop, "mc": mc})

    if use_vlm and api_key:
        selector = VLMSelector(query_raw=query_raw, targets=targets,
                               api_key=api_key, vlm_model=vlm_model,
                               frame_stride=frame_stride)
    else:
        from .select import WorldSelector
        selector = WorldSelector(targets=targets, query_raw=query_raw,
                                 conf=conf, imgsz=imgsz)

    judged = selector.select_first_frame(frame, cand)

    for c in cand:
        is_t, detail = judged.get(c["key"], (False, {"selector": "world"}))
        hits = detail.get("hits", [])
        wm = detail.get("world_match")
        if wm is not None:
            top1 = wm.get("phrase", "-")
        else:
            top1 = hits[0] if hits else "-"
        x1, y1, x2, y2, crop, mc = c["x1"], c["y1"], c["x2"], c["y2"], c["crop"], c["mc"]
        roi, m2 = torso_roi(crop, mc)
        persons.append({
            "bi": len(persons), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "crop": crop, "is_t": bool(is_t), "top1": top1, "hits": hits,
            "detail": detail, "mc": mc, "roi": roi, "mask": m2
        })
        color = (0, 255, 0) if is_t else (160, 160, 160)
        cv2.rectangle(anno, (x1, y1), (x2, y2), color, 3)
        cv2.putText(anno, str(len(persons)), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return anno, persons
