"""영상 분석. 개인정보 보호용.
YOLO-seg로 사람을 찾고, VLM/월드로 개인정보 항목을 탐지한다.
분석 결과는 카테고리별 목록(체크리스트)으로 묶어서 돌려준다.
"""
import cv2
import numpy as np

from .config import CANDS, VLM_MODEL, canon_category, category_type, label_ko
from .detect import filter_contained, big_enough
from .select import VLMSelector
from .vlm_package import prompt_categories
from .world import match_box


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    ba = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / max(1.0, float(aa + ba - inter))


def _dedupe(boxes, thresh=0.55):
    out = []
    for b in boxes:
        if all(_iou(b, x) < thresh for x in out):
            out.append(list(b))
    return out


def _match_person(box_xyxy, cand):
    """VLM/월드 박스가 사람 박스 안에 있으면 그 사람을 돌려준다."""
    best, best_iou = None, 0.0
    for p in cand or []:
        hit, _matched, iou = match_box(p["xyxy"], [{
            "xyxy": list(box_xyxy), "conf": 1.0, "phrase": "", "category": "",
        }])
        if hit and iou >= best_iou:
            best, best_iou = p, iou
    return best


def _item_shell(cat, box):
    rtype = box.get("region_type")
    if rtype not in ("person", "region"):
        rtype = category_type(cat)
    return {
        "category": cat,
        "label_ko": box.get("label_ko") or label_ko(cat),
        "type": rtype,
        "prompt_match": bool(box.get("matches_prompt")),
        "boxes": [],
        "person_keys": set(),
        "source": box.get("source", "world"),
    }


def _add_box(items, cat, box, person=None):
    if cat not in items:
        items[cat] = _item_shell(cat, box)
    it = items[cat]
    if box.get("matches_prompt"):
        it["prompt_match"] = True
    if person is not None:
        it["person_keys"].add(person["key"])
        it["type"] = "person"
    it["boxes"].append(list(box["xyxy"]))


def _finalize(items, frame):
    out = []
    for cat, it in items.items():
        boxes = _dedupe(it["boxes"], 0.45)
        it["boxes"] = boxes
        if it["type"] == "person" and it["person_keys"]:
            it["count"] = len(it["person_keys"])
        else:
            it["count"] = len(boxes)
        it.pop("person_keys", None)
        it["thumb"] = None
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            try:
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    w = 120
                    h = max(1, int(w * crop.shape[0] / max(1, crop.shape[1])))
                    it["thumb"] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        out.append(it)
    # 프롬프트 항목 먼저, 그다음 추가 후보. 각각 개수 많은 순.
    out.sort(key=lambda x: (not x["prompt_match"], -x["count"], x["label_ko"]))
    return out


def analyze_first_frame_items(frame, model, conf, imgsz,
                              query_raw="", targets=None,
                              api_key=None,
                              vlm_model=VLM_MODEL):
    """returns (anno_bgr, persons, items).
    items: 카테고리별 체크리스트 항목. persons: 가벼운 요약 목록.
    """
    targets = ([targets] if isinstance(targets, str)
               else (list(targets) if targets else list(CANDS)))
    if "face" not in targets:
        targets = list(targets) + ["face"]

    pr = list(model.predict(source=frame, conf=conf, classes=[0],
                            verbose=False, imgsz=imgsz))[0]
    boxes = np.zeros((0, 4), dtype=int)
    if pr.boxes is not None and len(pr.boxes) > 0:
        boxes = pr.boxes.xyxy.cpu().numpy().astype(int)

    keep = filter_contained(boxes) if len(boxes) else []
    anno = frame.copy()
    cand = []
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        if not keep[bi]:
            continue
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        if not big_enough(x1, y1, x2, y2):
            continue
        cand.append({"key": len(cand), "xyxy": [x1, y1, x2, y2]})

    selector = VLMSelector(query_raw=query_raw, targets=targets,
                           api_key=api_key, vlm_model=vlm_model)
    world_boxes = selector.detect_boxes(frame)
    judged = selector.match_items(cand, world_boxes)

    # 사람별 판정 + 화면 표시
    persons = []
    for c in cand:
        is_t, detail = judged.get(c["key"], (False, {"selector": "world"}))
        hits = detail.get("hits", [])
        wm = detail.get("world_match")
        top1 = wm.get("phrase", "-") if wm else (label_ko(hits[0]) if hits else "-")
        x1, y1, x2, y2 = c["xyxy"]
        persons.append({
            "bi": len(persons), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "is_t": bool(is_t), "top1": top1, "hits": hits, "detail": detail,
        })
        color = (0, 255, 0) if is_t else (160, 160, 160)
        cv2.rectangle(anno, (x1, y1), (x2, y2), color, 3)
        cv2.putText(anno, str(len(persons)), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 카테고리 항목 만들기
    items = {}
    prompt_set = prompt_categories(query_raw)
    face_people = set()

    for box in world_boxes:
        cat = canon_category(box.get("category") or box.get("phrase"))
        rtype = box.get("region_type")
        if rtype not in ("person", "region"):
            rtype = category_type(cat)
        if cat in prompt_set:
            box = dict(box)
            box["matches_prompt"] = True
        person = None
        if rtype == "person" or category_type(cat) == "person":
            person = _match_person(box["xyxy"], cand)
        _add_box(items, cat, box, person)
        if cat == "face" and person is not None:
            face_people.add(person["key"])

    # 얼굴은 사람마다 항상 후보로 올린다. (선택 해제하면 렌더 안 함)
    if "face" in targets:
        for c in cand:
            if c["key"] in face_people:
                continue
            x1, y1, x2, y2 = c["xyxy"]
            fy2 = y1 + int((y2 - y1) * 0.35)
            box = {
                "category": "face", "label_ko": label_ko("face"),
                "xyxy": [x1, y1, x2, fy2], "conf": 1.0,
                "matches_prompt": "face" in prompt_set,
                "region_type": "person", "source": "fallback",
            }
            _add_box(items, "face", box, c)

    # 각 사람의 hits에 얼굴 보정 (프리뷰 배지용)
    for p in persons:
        if "face" in targets and "face" not in p["hits"]:
            p["hits"] = list(p["hits"]) + ["face"]

    final = _finalize(items, frame)
    # 배경 개인정보 박스도 미리보기 화면에 표시한다.
    for it in final:
        if it.get("type") != "region":
            continue
        for x1, y1, x2, y2 in it.get("boxes", []):
            cv2.rectangle(anno, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(anno, it["label_ko"], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    return anno, persons, final


def _merge_items(all_items):
    merged = {}
    for items in all_items:
        for it in items:
            cat = it["category"]
            if cat not in merged:
                m = dict(it)
                m["boxes"] = list(it.get("boxes", []))
                m["person_keys"] = set()
                merged[cat] = m
            m = merged[cat]
            m["count"] = max(m.get("count", 0), it.get("count", 0))
            m["prompt_match"] = m.get("prompt_match") or it.get("prompt_match", False)
            m["boxes"] = _dedupe(list(m.get("boxes", [])) + list(it.get("boxes", [])), 0.45)
            if m.get("thumb") is None:
                m["thumb"] = it.get("thumb")
    out = []
    for m in merged.values():
        if m.get("type") != "person" or not m.get("boxes"):
            m["count"] = len(m.get("boxes", [])) if m.get("type") == "region" else m.get("count", 0)
        m.pop("person_keys", None)
        out.append(m)
    out.sort(key=lambda x: (not x["prompt_match"], -x.get("count", 0), x["label_ko"]))
    return out


def analyze_video(video_path, model, conf, imgsz,
                  query_raw="", targets=None, api_key=None,
                  vlm_model=VLM_MODEL, samples=3):
    """영상 앞/중간/끝을 샘플링해 체크리스트 항목을 만든다.
    returns (anno_bgr, persons, items). 첫 샘플 화면을 미리보기로 쓴다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, [], []
        return analyze_first_frame_items(frame, model, conf, imgsz,
                                         query_raw=query_raw, targets=targets,
                                         api_key=api_key, vlm_model=vlm_model)
    n = max(1, int(samples))
    idxs = sorted(set(int(round(v)) for v in np.linspace(0, max(0, total - 1), n)))
    anno0, persons0 = None, []
    all_items = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        anno, persons, items = analyze_first_frame_items(
            frame, model, conf, imgsz, query_raw=query_raw, targets=targets,
            api_key=api_key, vlm_model=vlm_model)
        if anno0 is None:
            anno0, persons0 = anno, persons
        all_items.append(items)
    cap.release()
    if not all_items:
        return None, [], []
    return anno0, persons0, _merge_items(all_items)
