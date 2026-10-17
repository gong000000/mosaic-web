"""첫 화면 분석. 미리보기와 파이프라인이 같은 판정 코드를 쓰도록 분리."""
import cv2
from .clothing import torso_roi, hsv_hist
from .detect import filter_contained, big_enough
from .select import HybridSelector


def analyze_first_frame(frame, model, conf, imgsz, targets, margin,
                        query_raw="", world_mode="auto"):
    """returns (anno_bgr, persons). persons: dict 리스트.
    열린 말(안경·우산 등)이면 월드를 첫 화면 1회에만 돌린다."""
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
    selector = HybridSelector(targets, query_raw=query_raw, margin=margin,
                              world_mode=world_mode, world_stride=1)
    judged = selector.judge_frame(frame, cand)
    for c in cand:
        is_t, detail = judged.get(c["key"], (False, {"selector": "heuristic"}))
        hits = detail.get("hits", [])
        wm = detail.get("world_match")
        if wm is not None:
            top1 = wm.get("phrase", "-")
        else:
            try:
                top1 = hits[0] if hits else detail["top"][0][0]
            except Exception:
                top1 = "-"
        x1, y1, x2, y2, crop, mc = c["x1"], c["y1"], c["x2"], c["y2"], c["crop"], c["mc"]
        roi, m2 = torso_roi(crop, mc)
        hh = hsv_hist(roi, m2)
        persons.append({"bi": len(persons), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "crop": crop, "is_t": bool(is_t), "top1": top1, "hits": hits, "hh": hh,
                        "detail": detail})
        cv2.rectangle(anno, (x1, y1), (x2, y2), (0, 255, 0) if is_t else (160, 160, 160), 3)
        cv2.putText(anno, str(len(persons)), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if is_t else (160, 160, 160), 3)
    return anno, persons
