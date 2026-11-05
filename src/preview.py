"""영상 분석. 개인정보 보호용.
YOLO11-seg는 사람 박스/마스크와 차량 크롭 후보만 만들고, 개인정보 항목은
VLM이 찾는다. VLM이 찾은 항목을 카테고리별 체크리스트로 묶어 돌려준다.
"""
import os
import cv2
import math
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (CANDS, CATEGORY_TYPE, VLM_MODEL, canon_category,
                     category_type, label_ko)
from .detect import filter_contained, big_enough, iou, match_box
from .render import render_person_strict, render_region
from .select import VLMSelector
from .models import MODEL_LOCK
from .vlm_budget import VLMBudget, classify_vlm_error
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
        # 여러 샘플 스레드가 같은 YOLO 모델을 동시에 predict하면 내부
        # predictor 상태가 깨질 수 있다. 추론만 직렬화하고 VLM 대기는
        # 각 스레드에서 병렬로 유지한다.
        with MODEL_LOCK:
            pr = list(model.predict(source=frame, conf=0.18,
                                    classes=VEHICLE_CLASSES, verbose=False,
                                    imgsz=max(640, int(imgsz)),
                                    max_det=20))[0]
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
                   budget: VLMBudget | None = None, cancel_event=None):
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
        if cancel_event is not None and cancel_event.is_set():
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
        # 병렬 샘플 스레드에서도 한도를 넘지 않도록 원자적으로 예약한다.
        if budget is not None and not budget.reserve():
            break
        calls += 1
        try:
            found = detect_with_vlm(
                crop, ["license_plate"], api_key, model=vlm_model,
                query_raw="", include_all=False, base_url=vlm_base_url,
                focus="license_plate",
                max_retries=2)
        except VLMError as e:
            # 한도/인증/모델 오류는 더 호출해도 실패하므로 상위로 올려
            # 남은 샘플 호출을 중단시킨다. 서버/파싱 오류만 무시한다.
            if classify_vlm_error(e) in ("quota_exceeded", "auth_error",
                                         "model_not_found"):
                raise
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
                              plate_budget: VLMBudget | None = None,
                              cancel_event=None):
    """returns (anno_bgr, persons, items).
    items: VLM이 찾은 카테고리별 체크리스트 항목. persons: 사람 요약 목록.
    """
    targets = ([targets] if isinstance(targets, str)
               else (list(targets) if targets else list(CANDS)))

    with MODEL_LOCK:
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
    if (refine_plates and "license_plate" in targets and plate_zoom_calls > 0
            and not (cancel_event is not None and cancel_event.is_set())):
        vlm_boxes = _refine_plates(
            frame, vlm_boxes, model, imgsz, api_key, vlm_model, vlm_base_url,
            max_calls=min(2, plate_zoom_calls), budget=plate_budget,
            cancel_event=cancel_event)
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
                  plate_zoom_budget=6, parallel_workers=3):
    """영상 앞/중간/끝을 고르게 샘플링해 체크리스트 항목을 만든다.
    샘플 프레임들을 ThreadPoolExecutor로 병렬 처리한다. 각 워커는
    해당 프레임만 읽고 VLM을 호출해 독립적으로 결과를 낸다.
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

    # 병렬 처리를 위해 프레임 비트맵을 사전에 읽어 둔다.
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            actual_idx = max(0, pos - 1)
        except Exception:
            actual_idx = idx
        frames.append((actual_idx, frame))
    cap.release()
    if not frames:
        return None, [], []

    plate_budget = VLMBudget(
        per_video_limit=max(0, int(plate_zoom_budget)))
    results = [None] * len(frames)
    errors = []
    fatal_errors = []
    fatal = threading.Event()

    def _process(i, actual_idx, frame):
        # 한도/인증 오류가 먼저 난 경우 대기 중이던 샘플은 호출하지 않는다.
        if fatal.is_set():
            results[i] = (None, [], [])
            return
        try:
            anno, persons, items = analyze_first_frame_items(
                frame, model, conf, imgsz, query_raw=query_raw,
                targets=targets, api_key=api_key, vlm_model=vlm_model,
                vlm_base_url=vlm_base_url, frame_idx=actual_idx,
                plate_budget=plate_budget, cancel_event=fatal)
            results[i] = (anno, persons, items)
        except Exception as e:
            errors.append(e)
            if classify_vlm_error(e) in ("quota_exceeded", "auth_error",
                                         "model_not_found"):
                fatal_errors.append(e)
                fatal.set()
            results[i] = (None, [], [])

    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        futs = []
        for i, (actual_idx, frame) in enumerate(frames):
            futs.append(ex.submit(_process, i, actual_idx, frame))
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errors.append(e)

    # 한도/인증 오류는 빈 결과로 숨기지 않고 그대로 알린다.
    if fatal_errors:
        raise fatal_errors[0]

    collected = [r for r in results
                 if r is not None and r[0] is not None]
    if not collected:
        if errors:
            raise errors[0]
        return None, [], []
    anno0, persons0 = collected[0][0], collected[0][1]
    all_items = [r[2] for r in collected]
    return anno0, persons0, _merge_items(all_items)


def _pair_boxes(prev_boxes, next_boxes):
    """앞/뒤 샘플 박스를 중심 거리로 짝짓는다. 미리보기 보간용."""
    used = set()
    pairs = []
    for p in prev_boxes:
        pxy = [float(v) for v in p]
        pw = max(1.0, pxy[2] - pxy[0])
        ph = max(1.0, pxy[3] - pxy[1])
        pcx, pcy = (pxy[0] + pxy[2]) / 2.0, (pxy[1] + pxy[3]) / 2.0
        best, best_score = None, None
        for j, n in enumerate(next_boxes):
            if j in used:
                continue
            nxy = [float(v) for v in n]
            nw = max(1.0, nxy[2] - nxy[0])
            nh = max(1.0, nxy[3] - nxy[1])
            ratio = (nw * nh) / max(1.0, pw * ph)
            if ratio < 0.25 or ratio > 4.0:
                continue
            ncx, ncy = (nxy[0] + nxy[2]) / 2.0, (nxy[1] + nxy[3]) / 2.0
            dist = math.hypot(ncx - pcx, ncy - pcy) / max(pw, ph, nw, nh)
            if dist > 1.5:
                continue
            if best_score is None or dist < best_score:
                best, best_score = j, dist
        if best is None:
            pairs.append((pxy, pxy))
        else:
            used.add(best)
            pairs.append((pxy, [float(v) for v in next_boxes[best]]))
    return pairs


def _boxes_at_frame(samples, frame_idx):
    """현재 프레임에 가장 가까운 박스 묶음을 돌려준다(가능하면 보간)."""
    if not samples:
        return []
    keys = sorted(int(k) for k in samples)
    if not keys:
        return []
    prev = [k for k in keys if k <= frame_idx]
    if not prev:
        return [list(map(int, b)) for b in samples[keys[0]]]
    prev_k = prev[-1]
    nxt = [k for k in keys if k > frame_idx]
    prev_boxes = [list(map(int, b)) for b in samples[prev_k]]
    if not nxt:
        return prev_boxes
    next_k = nxt[0]
    gap = next_k - prev_k
    if gap <= 0 or gap > 120:
        return prev_boxes
    t = max(0.0, min(1.0, (frame_idx - prev_k) / float(gap)))
    next_boxes = [list(map(int, b)) for b in samples[next_k]]
    out = []
    for a, b in _pair_boxes(prev_boxes, next_boxes):
        out.append([int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4)])
    return out


def _fmt_time(seconds):
    """재생 시간 표시용 mm:ss."""
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_filmstrip(frames, width=1600, thumb_h=72, max_thumbs=48,
                    progress=0.0):
    """타임라인 아래 필름스트립을 한 장의 이미지로 만든다.

    samples의 대표 프레임들을 시간 순으로 균등하게 나열한다. blur.me처럼
    '어디를 봐도 어떤 장면인지' 바로 보이게 하고, 현재 재생 위치에 흰색
    커서를 표시한다. 썸네일은 한 번만 만들어 streamlit 캐시에 둔다.
    """
    if not frames:
        return None
    h0, w0 = frames[0].shape[:2]
    aspect = w0 / float(max(1, h0))
    thumb_w = max(32, int(round(thumb_h * aspect)))
    n = min(len(frames), int(max_thumbs))
    if n <= 0:
        return None
    if len(frames) > n:
        idxs = [int(round(i * (len(frames) - 1) / float(max(1, n - 1))))
                for i in range(n)]
    else:
        idxs = list(range(len(frames)))
    thumbs = []
    for i in idxs:
        fr = frames[i]
        thumbs.append(cv2.resize(fr, (thumb_w, thumb_h),
                                 interpolation=cv2.INTER_AREA))
    total_w = max(1, sum(t.shape[1] for t in thumbs))
    strip = np.hstack(thumbs)
    if strip.shape[1] != total_w:
        strip = cv2.resize(strip, (total_w, thumb_h))
    else:
        strip = strip.copy()
    # 경계선 + 현재 위치 커서
    for x in range(thumb_w, total_w, thumb_w):
        cv2.line(strip, (x, 0), (x, thumb_h), (0, 0, 0), 1)
    cv2.rectangle(strip, (0, 0), (total_w - 1, thumb_h - 1), (30, 30, 30), 1)
    cx = int(round(max(0.0, min(1.0, float(progress))) * (total_w - 1)))
    cv2.line(strip, (cx, 0), (cx, thumb_h), (255, 255, 255), 3)
    return strip


def build_preview_video(
        proc_in, items, selected_cats, style="pixel", blocks=30,
        blur=61, area_mode="face_only", max_frames=90, tmp_dir=None,
        on_progress=None, preset="ultrafast", crf=28):
    """타임라인을 움직이지 않아도 움직임을 볼 수 있는 미리보기 영상.

    선택 항목이 없으면 원본 H.264 사본만 만든다. 항목이 있으면 샘플
    박스를 전 프레임에 보간해 render_preview_frame으로 그린다. 전체
    렌더가 아니라 재생 미리보기용이라 프레임 수와 길이를 제한한다.
    returns: 파일 경로 또는 None.
    """
    if not proc_in:
        return None
    cap = cv2.VideoCapture(proc_in)
    if not cap.isOpened():
        return None
    error = None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_n = min(max(1, total or max_frames), int(max_frames))
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            out_path = os.path.join(tmp_dir, "preview_video.mp4")
        else:
            import tempfile
            out_path = tempfile.mktemp(suffix=".mp4", prefix="preview_")
        writer = None
        writer_is_ffmpeg = False
        try:
            from .render import FFmpegFrameWriter
            writer = FFmpegFrameWriter(out_path, 0, 0, fps,
                                       audio_src=proc_in,
                                       preset=preset, crf=crf)
            writer_is_ffmpeg = True
        except Exception:
            writer = None
        try:
            if writer is not None:
                width = writer.width
                height = writer.height
            else:
                width = height = 0
            idxs = []
            if total > 1:
                idxs = [int(round(i * (total - 1) / float(max(1, frame_n - 1))))
                        for i in range(frame_n)]
            else:
                idxs = [0]
            for i, frame_idx in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if selected_cats and items:
                    frame = render_preview_frame(
                        frame, items, selected_cats, style=style,
                        blocks=blocks, blur=blur, area_mode=area_mode,
                        frame_idx=int(frame_idx))
                # writer.write()가 인코더 크기(0이면 첫 프레임에서 결정)를
                # 처리하므로 여기서 미리 resize하지 않는다.
                if writer_is_ffmpeg:
                    writer.write(frame)
                elif writer is not None:
                    writer.write(frame)
                else:
                    if writer is None and width == 0:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(out_path, fourcc, fps,
                                                 (width, height))
                        if not writer.isOpened():
                            return None
                    writer.write(frame)
                if on_progress is not None:
                    try:
                        on_progress((i + 1) / float(max(1, len(idxs))))
                    except Exception:
                        pass
            if writer is not None:
                if writer_is_ffmpeg:
                    writer.close()
                else:
                    writer.release()
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            return None
        except Exception as e:
            error = e
            try:
                if writer is not None and hasattr(writer, "abort"):
                    writer.abort()
                elif writer is not None:
                    writer.release()
            except Exception:
                pass
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return None
    finally:
        cap.release()
        # 실패한 경우에만 임시 프리뷰 파일을 정리한다(성공은 return로 빠져나간다).
        if error is not None and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


def render_preview_frame(frame, items, selected_cats, style="pixel",
                         blocks=30, blur=61, area_mode="face_only",
                         frame_idx=0):
    """한 프레임에 선택 항목 모자이크를 미리 적용한다.

    전체 영상 렌더 전에 타임라인/효과를 즉시 확인하기 위한 근사 렌더다.
    사람 항목은 '가리는 범위'를 적용하고, 번호판·명찰은 탐지 박스를
    그대로 가린다.
    """
    out = frame.copy()
    selected = set(selected_cats or [])
    for it in items or []:
        cat = it.get("category")
        if cat not in selected:
            continue
        boxes = _boxes_at_frame(it.get("samples") or {}, int(frame_idx))
        if not boxes:
            continue
        for xyxy in boxes:
            if len(xyxy) != 4:
                continue
            if it.get("type") == "region" or cat == "name_tag":
                render_region(out, *xyxy, style=style, blocks=blocks,
                              blur=blur, pad=4)
            else:
                try:
                    render_person_strict(
                        out, *xyxy, None, style=style, blocks=blocks,
                        blur=blur, area_mode=area_mode)
                except Exception:
                    render_region(out, *xyxy, style=style, blocks=blocks,
                                  blur=blur, pad=4)
    return out
