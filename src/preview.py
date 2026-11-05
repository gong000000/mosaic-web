"""영상 분석. 개인정보 보호용.
YOLO11-seg는 사람 박스/마스크와 차량 크롭 후보만 만들고, 개인정보 항목은
VLM이 찾는다. VLM이 찾은 항목을 카테고리별 체크리스트로 묶어 돌려준다.
"""
import cv2
import numpy as np

from .config import (CANDS, CATEGORY_TYPE, VLM_MODEL, canon_category,
                     category_type, label_ko)
from .detect import filter_contained, big_enough, iou, match_box
from .select import VLMSelector
from .vlm_budget import VLMBudget
from .vlm_package import (VLMError, detect_with_vlm, prompt_categories)

# COCO 차량 클래스: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def _rtype(cat, box):
    """표준 카테고리는 설정을 우선하고, 자유 형식은 VLM 표기를 따른다."""
    if cat in CATEGORY_TYPE:
        return CATEGORY_TYPE[cat]
    rtype = box.get("region_type")
    return rtype if rtype in ("person", "region") else category_type(cat)


def _dedupe(boxes, thresh=0.55):
    out = []
    for b in boxes:
        if all(iou(b, x) < thresh for x in out):
            out.append(list(b))
    return out


def _match_person(box_xyxy, cand):
    """VLM 박스가 사람 박스 안에 있으면 그 사람을 돌려준다."""
    best, best_iou = None, 0.0
    for p in cand or []:
        hit, _matched, score = match_box(p["xyxy"], [{
            "xyxy": list(box_xyxy), "conf": 1.0, "phrase": "", "category": "",
        }])
        if hit and score >= best_iou:
            best, best_iou = p, score
    return best


def _item_shell(cat, box):
    rtype = _rtype(cat, box)
    return {
        "category": cat,
        "label_ko": box.get("label_ko") or label_ko(cat),
        "type": rtype,
        "prompt_match": bool(box.get("matches_prompt")),
        "boxes": [],
        "person_keys": set(),
        "samples": {},
        "source": "vlm",
    }


def _add_box(items, cat, box, person=None, frame_idx=0):
    if cat not in items:
        items[cat] = _item_shell(cat, box)
    it = items[cat]
    if box.get("matches_prompt"):
        it["prompt_match"] = True
    if person is not None:
        it["person_keys"].add(person["key"])
        it["type"] = "person"
    xyxy = list(box["xyxy"])
    it["boxes"].append(xyxy)
    it["samples"].setdefault(int(frame_idx), []).append(xyxy)


def _finalize(items, frame):
    out = []
    for it in items.values():
        boxes = _dedupe(it["boxes"], 0.45)
        it["boxes"] = boxes
        it["samples"] = {int(k): _dedupe(v, 0.45)
                         for k, v in it.get("samples", {}).items()}
        if it["type"] == "person" and it["person_keys"]:
            it["count"] = len(it["person_keys"])
        else:
            it["count"] = len(boxes)
        it.pop("person_keys", None)
        it["thumb"] = None
        it["box_count"] = len(boxes)
        if boxes:
            # 대표 썸네일: 가장 큰 박스 선택 (가장 선명하게 찍힐 확률이 높음)
            best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = best
            try:
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    w = 120
                    h = max(1, int(w * crop.shape[0] / max(1, crop.shape[1])))
                    it["thumb"] = cv2.resize(crop, (w, h),
                                             interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        out.append(it)
    # 요청 항목 먼저, 그다음 추가 후보. 각각 개수 많은 순.
    out.sort(key=lambda x: (not x["prompt_match"], -x["count"], x["label_ko"]))
    return out


def _detect_vehicles(model, frame, imgsz):
    """번호판 확대 탐지를 위한 차량 크롭 후보. VLM이 아니라 YOLO11-seg가
    '차량 영역'만 제안하고, 번호판 판정은 VLM이 한다."""
    try:
        pr = list(model.predict(source=frame, conf=0.18,
                                classes=VEHICLE_CLASSES, verbose=False,
                                imgsz=max(640, int(imgsz)), max_det=20))[0]
        if pr.boxes is None or len(pr.boxes) == 0:
            return []
        xy = pr.boxes.xyxy.cpu().numpy().astype(int)
        cf = pr.boxes.conf.cpu().numpy()
    except Exception:
        return []
    out = []
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2), conf in zip(xy, cf):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        # 아주 작은 차량도 번호판 확대 후보로 남긴다. 최종 판정은 VLM이 한다.
        if x2 - x1 < 24 or y2 - y1 < 18:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return out


def _refine_plates(frame, vlm_boxes, model, imgsz, api_key, vlm_model,
                   vlm_base_url, max_calls=2, max_vehicles=4,
                   budget: VLMBudget | None = None):
    """VLM이 전체 프레임에서 번호판을 놓쳤을 때, 차량 크롭을 확대해
    번호판 전용 VLM 호출을 한다. 찾은 박스는 원본 좌표로 되돌린다.
    번호판 판정은 여전히 VLM이 하며, YOLO는 차량 크롭만 제안한다.

    max_calls는 이 프레임에서 추가로 허용할 VLM 호출 수다.
    budget이 주어지면 영상 전체 예산을 확인하고 소진 시 호출하지 않는다.
    """
    h0, w0 = frame.shape[:2]
    plates = [b for b in (vlm_boxes or [])
              if canon_category(b.get("category") or b.get("phrase"))
              == "license_plate"]
    vehicles = _detect_vehicles(model, frame, imgsz)

    # 예산 확인: 영상 전체 예산이 소진되면 보조 호출을 하지 않는다.
    if budget is not None:
        if not budget.can_call():
            return vlm_boxes
        max_calls = min(max_calls, budget.remaining())

    if not vehicles or max_calls <= 0:
        return vlm_boxes

    added = []
    calls = 0
    for vx1, vy1, vx2, vy2, _conf in vehicles[:max_vehicles]:
        if calls >= max_calls:
            break
        # 예산 재확인 (중간 호출에서 소진될 수 있음)
        if budget is not None and not budget.can_call():
            break
        # 이미 번호판을 찾은 차량은 건너뛴다.
        already = False
        for p in plates + added:
            x1, y1, x2, y2 = p["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                already = True
                break
        if already:
            continue

        vw, vh = vx2 - vx1, vy2 - vy1
        pad_x, pad_y = int(vw * 0.12), int(vh * 0.12)
        cx1, cy1 = max(0, vx1 - pad_x), max(0, vy1 - pad_y)
        cx2, cy2 = min(w0, vx2 + pad_x), min(h0, vy2 + pad_y)
        if cx2 - cx1 < 80 or cy2 - cy1 < 60:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        cw, ch = crop.shape[1], crop.shape[0]
        target_w = min(1280, max(640, cw))
        if target_w > cw:
            target_h = max(1, int(ch * target_w / cw))
            crop = cv2.resize(crop, (target_w, target_h),
                              interpolation=cv2.INTER_CUBIC)
        ch2, cw2 = crop.shape[:2]
        calls += 1
        if budget is not None:
            budget.record(1)
        try:
            found = detect_with_vlm(
                crop, ["license_plate"], api_key, model=vlm_model,
                query_raw="", include_all=False, base_url=vlm_base_url,
                focus="license_plate",
                max_retries=2)
        except VLMError as e:
            print(f"번호판 확대 탐지 실패(무시): {e}")
            continue

        ix, iy = cw / float(cw2), ch / float(ch2)
        for b in found:
            if canon_category(b.get("category") or b.get("phrase")) \
                    != "license_plate":
                continue
            bx1, by1, bx2, by2 = b["xyxy"]
            fx1 = max(0, min(w0, cx1 + int(round(bx1 * ix))))
            fy1 = max(0, min(h0, cy1 + int(round(by1 * iy))))
            fx2 = max(0, min(w0, cx1 + int(round(bx2 * ix))))
            fy2 = max(0, min(h0, cy1 + int(round(by2 * iy))))
            if fx2 - fx1 < 8 or fy2 - fy1 < 4:
                continue
            full = dict(b)
            full["xyxy"] = [fx1, fy1, fx2, fy2]
            full["source"] = "vlm"
            if all(iou(full["xyxy"], p["xyxy"]) < 0.3
                   for p in plates + added):
                added.append(full)
    return list(vlm_boxes or []) + added


def analyze_first_frame_items(frame, model, conf, imgsz,
                              query_raw="", targets=None,
                              api_key=None,
                              vlm_model=VLM_MODEL,
                              vlm_base_url=None,
                              frame_idx=0,
                              refine_plates=True,
                              plate_zoom_calls=2,
                              plate_budget: VLMBudget | None = None):
    """returns (anno_bgr, persons, items).
    items: VLM이 찾은 카테고리별 체크리스트 항목. persons: 사람 요약 목록.
    """
    targets = ([targets] if isinstance(targets, str)
               else (list(targets) if targets else list(CANDS)))

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
                           api_key=api_key, vlm_model=vlm_model,
                           vlm_base_url=vlm_base_url)
    vlm_boxes = selector.detect_boxes(frame)
    if refine_plates and "license_plate" in targets and plate_zoom_calls > 0:
        vlm_boxes = _refine_plates(
            frame, vlm_boxes, model, imgsz, api_key, vlm_model, vlm_base_url,
            max_calls=min(2, plate_zoom_calls), budget=plate_budget)
    judged = selector.match_items(cand, vlm_boxes)

    persons = []
    for c in cand:
        is_t, detail = judged.get(
            c["key"], (False, {"selector": "vlm", "hits": []}))
        hits = detail.get("hits", [])
        vm = detail.get("vlm_match")
        top1 = (vm.get("phrase", "-") if vm
                else (label_ko(hits[0]) if hits else "-"))
        x1, y1, x2, y2 = c["xyxy"]
        persons.append({
            "bi": len(persons), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "is_t": bool(is_t), "top1": top1, "hits": hits, "detail": detail,
        })
        color = (0, 255, 0) if is_t else (160, 160, 160)
        cv2.rectangle(anno, (x1, y1), (x2, y2), color, 3)
        cv2.putText(anno, str(len(persons)), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    items = {}
    prompt_set = prompt_categories(query_raw)
    for box in vlm_boxes:
        raw_cat = box.get("category") or box.get("phrase")
        if not raw_cat:
            continue
        cat = canon_category(raw_cat)
        if cat in ("tattoo", "문신"):
            continue
        rtype = _rtype(cat, box)
        if cat in prompt_set:
            box = dict(box)
            box["matches_prompt"] = True
        person = None
        if rtype == "person" or category_type(cat) == "person":
            person = _match_person(box["xyxy"], cand)
        _add_box(items, cat, box, person, frame_idx)

    final = _finalize(items, frame)
    # 배경 개인정보 박스(번호판·간판 등)를 미리보기 화면에 표시한다.
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
                m["samples"] = {}
                m["person_keys"] = set()
                merged[cat] = m
            m = merged[cat]
            m["count"] = max(m.get("count", 0), it.get("count", 0))
            m["box_count"] = m.get("box_count", 0) + it.get("box_count", 0)
            m["prompt_match"] = (m.get("prompt_match")
                                 or it.get("prompt_match", False))
            m["boxes"] = _dedupe(list(m.get("boxes", []))
                                 + list(it.get("boxes", [])), 0.45)
            for fidx, boxes in (it.get("samples") or {}).items():
                m["samples"].setdefault(int(fidx), []).extend(boxes)
            if m.get("thumb") is None:
                m["thumb"] = it.get("thumb")
    out = []
    for m in merged.values():
        m["samples"] = {int(k): _dedupe(v, 0.45)
                        for k, v in m.get("samples", {}).items()}
        # count는 영상 전체 누적 박스 수가 아니라 한 프레임에서 동시에
        # 보인 최대 개수다. 움직이는 번호판을 여러 '곳'으로 세지 않는다.
        m.pop("person_keys", None)
        out.append(m)
    out.sort(key=lambda x: (not x["prompt_match"], -x.get("count", 0),
                            x["label_ko"]))
    return out


def analyze_video(video_path, model, conf, imgsz,
                  query_raw="", targets=None, api_key=None,
                  vlm_model=VLM_MODEL, samples=12, vlm_base_url=None,
                  plate_zoom_budget=6):
    """영상 앞/중간/끝을 고르게 샘플링해 체크리스트 항목을 만든다.
    기본 12개 샘플로 짧게 등장하는 번호판도 최대한 포함한다.
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
        return analyze_first_frame_items(
            frame, model, conf, imgsz, query_raw=query_raw, targets=targets,
            api_key=api_key, vlm_model=vlm_model, vlm_base_url=vlm_base_url,
            frame_idx=0)
    n = max(1, int(samples))
    idxs = sorted(set(int(round(v))
                      for v in np.linspace(0, max(0, total - 1), n)))
    anno0, persons0 = None, []
    all_items = []
    plate_budget = VLMBudget(
        per_video_limit=max(0, int(plate_zoom_budget)))
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # 일부 코덱은 요청한 프레임과 실제로 읽은 프레임이 다르다.
        # priors를 프레임에 정확히 맞추기 위해 실제 위치를 쓴다.
        try:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            actual_idx = max(0, pos - 1)
        except Exception:
            actual_idx = idx
        anno, persons, items = analyze_first_frame_items(
            frame, model, conf, imgsz, query_raw=query_raw, targets=targets,
            api_key=api_key, vlm_model=vlm_model, vlm_base_url=vlm_base_url,
            frame_idx=actual_idx, plate_budget=plate_budget)
        if anno0 is None:
            anno0, persons0 = anno, persons
        all_items.append(items)
    cap.release()
    if not all_items:
        return None, [], []
    return anno0, persons0, _merge_items(all_items)
