"""2패스 파이프라인 (개인정보 보호용).
1패스는 모든 프레임을 추적해 트랙 ID를 고정하고, 키프레임에서만 VLM
개인정보 탐지를 수행해 (사람 트랙, 카테고리)별로 수집한다.
2패스는 같은 추적 시퀀스로 선택된 카테고리만 렌더링한다.
사람 항목은 사람 마스크/얼굴 영역, 배경 항목(번호판·간판 등)은 VLM이 찾은
<<<<<<< HEAD
사각형 영역으로 가린다. 명찰처럼 사람에 붙은 항목은 VLM 박스를 사람 기준
상대 좌표로 저장해 '가리는 범위'와 무관하게 직접 가린다.
체크리스트에서 VLM이 찾은 배경 박스(region_priors)를 처리 단계의 초기값으로
넘겨, 처리 중 VLM이 같은 프레임에서 놓쳐도 체크리스트와 결과가 어긋나지
않게 한다.
=======
사각형 영역으로 가린다. 체크리스트에서 VLM이 찾은 배경 박스(region_priors)를
처리 단계의 초기값으로 넘겨, 처리 중 VLM이 같은 프레임에서 놓쳐도
체크리스트와 결과가 어긋나지 않게 한다.
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
"""
import cv2
import math
import numpy as np
import os
import shutil
import time
from collections import defaultdict

from .config import (CATEGORY_TYPE, ROOT_DIR, VLM_MODEL, canon_category,
                     category_type)
from .models import get_model
from .detect import filter_contained, big_enough, iou
from .render import (render_person_strict, render_region, mux_audio_h264,
                     apply_area_limit, cleanup_tmp)
from .select import VLMSelector

_TRACKER_CFG = os.path.join(ROOT_DIR, "configs", "bytetrack_stable.yaml")


def _reset_tracker(model):
    try:
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            delattr(pred, "trackers")
    except Exception:
        pass


def _cat_type(cat, rtype=None):
    """표준 카테고리는 설정을 따르고, 자유 형식은 VLM이 준 region_type을 쓴다."""
    if cat in CATEGORY_TYPE:
        return CATEGORY_TYPE[cat]
    if rtype in ("person", "region"):
        return rtype
    return category_type(cat)


def _dedupe_boxes(boxes, thresh=0.55):
    out = []
    for b in boxes:
        xy = b["xyxy"]
        dup = False
        for o in out:
            ox = o["xyxy"]
            ix1, iy1 = max(xy[0], ox[0]), max(xy[1], ox[1])
            ix2, iy2 = min(xy[2], ox[2]), min(xy[3], ox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            aa = max(1, xy[2] - xy[0]) * max(1, xy[3] - xy[1])
            ba = max(1, ox[2] - ox[0]) * max(1, ox[3] - ox[1])
            if inter / max(1.0, aa + ba - inter) >= thresh:
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def _sample_boxes(samples, sample_idx):
    """샘플 인덱스 이하에서 가장 가까운 VLM 박스 묶음을 돌려준다.
    첫 샘플보다 이전 프레임에는 아직 탐지가 없었으므로 빈 목록을 준다."""
    if not samples:
        return []
    pick = None
    for k in sorted(samples):
        if k <= sample_idx:
            pick = k
        else:
            break
    return samples[pick] if pick is not None else []


def _relative_box(inner, outer):
    """사람 박스 기준 상대 좌표(0~1)로 바꾼다. 사람이 움직여도 같이 따라간다."""
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    ow, oh = max(1.0, ox2 - ox1), max(1.0, oy2 - oy1)
    x1, y1, x2, y2 = [float(v) for v in inner]

    def clamp(v):
        return max(0.0, min(1.0, v))

    return [clamp((x1 - ox1) / ow), clamp((y1 - oy1) / oh),
            clamp((x2 - ox1) / ow), clamp((y2 - oy1) / oh)]


def _absolute_box(rel, outer, width, height):
    ox1, oy1, ox2, oy2 = outer
    ow, oh = max(1, ox2 - ox1), max(1, oy2 - oy1)
    x1 = int(round(ox1 + rel[0] * ow))
    y1 = int(round(oy1 + rel[1] * oh))
    x2 = int(round(ox1 + rel[2] * ow))
    y2 = int(round(oy1 + rel[3] * oh))
    return [max(0, x1), max(0, y1),
            min(width, x2), min(height, y2)]


def _store_person_box(samples, cat, sample_idx, box, persons):
    """사람에 붙은 VLM 박스를 트랙 ID와 상대 좌표로 저장한다.
    사람이 움직여도 명찰/얼굴 박스가 같이 따라가도록 하기 위함이다."""
    xyxy = [int(round(float(v))) for v in box.get("xyxy", [])]
    if len(xyxy) != 4 or xyxy[2] - xyxy[0] < 4 or xyxy[3] - xyxy[1] < 4:
        return
    cx, cy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
    best = None
    best_area = None
    for p in persons:
        px1, py1, px2, py2 = p["xyxy"]
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        if (px1 - pw * 0.03 <= cx <= px2 + pw * 0.03
                and py1 - ph * 0.03 <= cy <= py2 + ph * 0.03):
            area = pw * ph
            if best is None or area < best_area:
                best, best_area = p, area
    if best is None:
        # 중심이 사람 박스 밖이면 겹침이 충분한 사람을 쓴다.
        best_iou = 0.12
        for p in persons:
            overlap = iou(p["xyxy"], xyxy)
            if overlap >= best_iou:
                best, best_iou = p, overlap
    tid = int(best["tid"]) if best is not None else -1
    rel = _relative_box(xyxy, best["xyxy"]) if best is not None else None
    entries = samples[cat][sample_idx]
    for e in entries:
        if e["tid"] == tid and iou(e["abs"], xyxy) >= 0.55:
            return
    entries.append({"tid": tid, "rel": rel, "abs": xyxy})


def _nearest_rel_boxes(samples, sample_idx, tid):
    """해당 트랙에서 가장 최근에 찾은 사람 부착 박스의 상대 좌표."""
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        rels = [e["rel"] for e in samples[k]
                if e["tid"] == tid and e.get("rel")]
        if rels:
            return rels
    return []


def _nearest_abs_boxes(samples, sample_idx, tid=-1):
    """사람 트랙에 매칭되지 않은 박스의 절대 좌표."""
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        boxes = [e["abs"] for e in samples[k]
                 if e["tid"] == tid and e.get("abs")]
        if boxes:
            return boxes
    return []


def _render_rel_boxes(frame, raw_box, rels, style, blocks, blur):
    """사람 박스 기준 상대 좌표를 현재 프레임 좌표로 바꿔 가린다."""
    if not rels:
        return False
    h, w = frame.shape[:2]
    rendered = False
    for rel in rels:
        bx = _absolute_box(rel, raw_box, w, h)
        if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
            continue
        if render_region(frame, *bx, style=style, blocks=blocks,
                         blur=blur, pad=4):
            rendered = True
    return rendered


class MosaicJob:
    """개인정보 보호 모자이크 작업 1건."""

    def __init__(self, input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, query_raw="",
                 api_key=None, vlm_model=VLM_MODEL, vlm_base_url=None,
                 region_priors=None):
        self.input_path = input_path
        self.output_path = output_path
        self.targets = ([targets] if isinstance(targets, str)
                        else (list(targets) if targets else ["face"]))
<<<<<<< HEAD
=======
        if not self.targets:
            self.targets = ["face"]
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
        self.invert = invert
        self.conf = conf
        self.model_name = model_name
        self.imgsz = imgsz
        self.blocks = blocks
        self.style = style
        self.blur = blur
        self.ultra = ultra
        self.area_mode = area_mode
        self.encode_preset = encode_preset
        self.keyframe_stride = max(1, int(keyframe_stride))
        self.query_raw = query_raw or ""
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.vlm_base_url = vlm_base_url
        self.region_priors = region_priors or {}

    def run(self, progress_fn=None):
        input_path, output_path = self.input_path, self.output_path
        targets = self.targets
        invert = self.invert
        conf = self.conf
        model_name = self.model_name
        imgsz = self.imgsz
        blocks = self.blocks
        style = self.style
        blur = self.blur
        ultra = self.ultra
        area_mode = self.area_mode
        encode_preset = self.encode_preset
        keyframe_stride = self.keyframe_stride
        query_raw = self.query_raw
        api_key = self.api_key
        vlm_model = self.vlm_model
        vlm_base_url = self.vlm_base_url
        region_priors = self.region_priors

        model = get_model(model_name)
        _reset_tracker(model)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, first = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
        H, W = first.shape[:2]
        del first

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        if ultra:
            imgsz = max(imgsz, 960)
            conf = min(conf, 0.18)

        selector = VLMSelector(query_raw=query_raw, targets=targets,
                               api_key=api_key, vlm_model=vlm_model,
                               include_all=False,
                               vlm_base_url=vlm_base_url)

        # ---- 체크리스트 VLM 박스를 배경 영역 초기값으로 넣는다 ----
        region_samples = defaultdict(lambda: defaultdict(list))
<<<<<<< HEAD
        person_box_samples = defaultdict(lambda: defaultdict(list))
=======
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
        max_sidx = max(0, (total - 1) // keyframe_stride) if total > 0 else 0
        for cat, by_frame in (region_priors or {}).items():
            if cat not in targets or _cat_type(cat) != "region":
                continue
            for fidx, boxes in (by_frame or {}).items():
                try:
                    fidx = int(fidx)
                except (TypeError, ValueError):
                    continue
                # 1패스의 sample_idx = frame_idx // keyframe_stride 와 같게
                # 맞춘다. 반올림하면 짧게 등장하는 박스가 몇 프레임 늦게
                # 적용되어 놓칠 수 있다.
                sidx = min(max_sidx, max(0, fidx // keyframe_stride))
                for xyxy in boxes or []:
                    if len(xyxy) != 4:
                        continue
                    try:
                        coords = [int(float(v)) for v in xyxy]
                    except (TypeError, ValueError):
                        continue
                    if coords[2] - coords[0] < 4 or coords[3] - coords[1] < 4:
                        continue
                    region_samples[cat][sidx].append({
                        "xyxy": coords, "conf": 1.0, "category": cat,
                        "region_type": "region", "source": "vlm",
                        "matches_prompt": True,
                    })

        # ---- 1패스: 모든 프레임 추적 + 키프레임 VLM 탐지 ----
        # 1패스와 2패스가 같은 프레임 시퀀스를 추적해야 트랙 ID가 일치한다.
        votes = defaultdict(lambda: defaultdict(list))
        first_thumb = {}
        observed_types = {}
        person_cats_seen = {c for c in targets
                            if CATEGORY_TYPE.get(c) == "person"}
        all_detail = defaultdict(dict)

        t0 = time.time()
        results1 = model.track(source=input_path, stream=True, persist=True,
                               tracker=_TRACKER_CFG,
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30)

        frame_idx = -1
        for r in results1:
            frame_idx += 1
            if progress_fn and total > 0 and frame_idx % 5 == 0:
                try:
                    progress_fn(0.5 * min(1.0, (frame_idx + 1) / total),
                                frame_idx + 1, total, time.time() - t0)
                except Exception:
                    pass

            if frame_idx % keyframe_stride != 0:
                continue
            sample_idx = frame_idx // keyframe_stride

            frame = r.orig_img
            boxes = np.zeros((0, 4), dtype=int)
            ids = np.zeros((0,), dtype=int)
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = (r.boxes.id.cpu().numpy().astype(int)
                       if r.boxes.id is not None else [-1] * len(boxes))

            keep = filter_contained(boxes) if len(boxes) else []
            cand = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if not keep[i]:
                    continue
                tid = int(ids[i])
                x1c, y1c = max(0, int(x1)), max(0, int(y1))
                x2c, y2c = (min(frame.shape[1], int(x2)),
                            min(frame.shape[0], int(y2)))
                if not big_enough(x1c, y1c, x2c, y2c):
                    continue
                cand.append({"key": (tid, len(cand)), "tid": tid,
                             "xyxy": [x1c, y1c, x2c, y2c]})
                if tid not in first_thumb:
                    try:
                        crop = frame[y1c:y2c, x1c:x2c]
                        th = cv2.resize(
                            crop,
                            (96, max(1, int(96 * crop.shape[0]
                                           / max(1, crop.shape[1])))),
                            interpolation=cv2.INTER_AREA)
                        first_thumb[tid] = th
                    except Exception:
                        first_thumb[tid] = None

            # 사람이 없어도 배경 개인정보를 찾아야 하므로 항상 호출한다.
            vlm_boxes = selector.detect_boxes(frame)
            judged = selector.match_items(cand, vlm_boxes)
            hits_by_key = {}
            detail_by_key = {}
            for it in cand:
                _hit, _detail = judged.get(
                    it["key"], (False, {"selector": "vlm", "hits": []}))
                hits_by_key[it["key"]] = set(_detail.get("hits", []))
                detail_by_key[it["key"]] = _detail

            # 사람별 카테고리 투표. 선택된 카테고리만 반영한다.
            for it in cand:
                tid = it["tid"]
                hits = {c for c in hits_by_key.get(it["key"], set())
                        if c in targets}
                # 같은 키프레임에서 처음 발견된 자유 형식 카테고리도 즉시 투표한다.
                person_cats_seen.update(hits)
                cats_here = set(person_cats_seen)
                for cat in cats_here:
                    votes[tid][cat].append(cat in hits)
                for cat in hits:
                    all_detail[tid].setdefault(
                        cat, detail_by_key.get(it["key"], {}))

            # 배경 영역 후보 저장 (VLM이 찾은 번호판·간판 등)
            for b in vlm_boxes:
                cat = canon_category(b.get("category") or b.get("phrase"))
                if cat not in targets:
                    continue
                rtype = _cat_type(cat, b.get("region_type"))
                observed_types.setdefault(cat, rtype)
                if rtype == "region":
                    region_samples[cat][sample_idx].append(b)
                else:
                    # 명찰·얼굴처럼 사람에 붙은 항목도 별도 박스로 저장한다.
                    _store_person_box(person_box_samples, cat, sample_idx,
                                      b, cand)

        _reset_tracker(model)

        # ---- 전역 확정 ----
        # 사람 카테고리 / 배경 카테고리 분리. 선택 해제한 항목은 targets에 없다.
        person_cats = {c for c in targets
                       if c in person_cats_seen
                       or _cat_type(c, observed_types.get(c)) == "person"}
        region_cats = {c for c in targets if c not in person_cats}
        # 분석/체크리스트에서 한 번이라도 보인 배경 카테고리는 2패스에서 렌더한다.
        region_cats |= {c for c in region_samples
                        if c in targets and c not in person_cats}
        # 사람에 붙은 항목(명찰·얼굴 등)은 '가리는 범위'와 별개로
        # VLM 박스 위치를 직접 가린다.
        direct_cats = {c for c in person_cats if person_box_samples.get(c)}

        cat_on = {}
        global_on = {}
        global_hit_targets = {}
        global_detail = {}
        for tid, by_cat in votes.items():
            cat_on[tid] = {}
            for cat, lst in by_cat.items():
                if not lst:
                    continue
                frac = sum(1 for v in lst if v) / len(lst)
                cat_on[tid][cat] = frac >= 0.2
            on_cats = [c for c in targets
                       if cat_on[tid].get(c) and c in person_cats]
            global_on[tid] = bool(on_cats)
            global_hit_targets[tid] = on_cats
            fracs = [sum(1 for v in lst if v) / len(lst)
                     for lst in by_cat.values() if lst]
            frac_any = max(fracs) if fracs else 0.0
            samp = {}
            for c in on_cats or list(cat_on[tid].keys()):
                if tid in all_detail and c in all_detail[tid]:
                    samp = all_detail[tid][c]
                    break
            global_detail[tid] = {"frac": round(float(frac_any), 3),
                                  "cat_on": dict(cat_on[tid]),
                                  "sample": samp}

        # ---- 2패스: 선택된 카테고리만 렌더 ----
        tmp_noaudio = output_path + ".noaudio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
        if not out.isOpened():
            cleanup_tmp(tmp_noaudio)
            raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")

        smooth = {}
        hist_mask = {}
        hist_miss = defaultdict(int)
        rendered_tids = set()
        ALPHA = 0.55
        MAX_MISS = 12

        rendered_region_cats = set()

        results2 = model.track(source=input_path, stream=True, persist=True,
                               tracker=_TRACKER_CFG,
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30)
        done = 0
        for r in results2:
            frame = r.orig_img.copy()
<<<<<<< HEAD
            sidx = done // keyframe_stride
=======
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72

            seen = set()
            keep_union = None
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = (r.boxes.id.cpu().numpy().astype(int)
                       if r.boxes.id is not None else [-1] * len(boxes))
                masks_full = None
                if r.masks is not None:
                    try:
                        masks_full = r.masks.data.cpu().numpy()
                    except Exception:
                        masks_full = None
                keep = filter_contained(boxes)
                for _d, _k in zip(ids, keep):
                    if not _k:
                        seen.add(int(_d))
                boxes = boxes[keep]
                ids = ids[keep]
                if masks_full is not None:
                    masks_full = masks_full[np.array(keep)]
                items = []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    tid = int(ids[i])
                    seen.add(tid)
                    if tid in smooth:
                        px1, py1, px2, py2 = smooth[tid]
                        x1 = int(ALPHA * x1 + (1 - ALPHA) * px1)
                        y1 = int(ALPHA * y1 + (1 - ALPHA) * py1)
                        x2 = int(ALPHA * x2 + (1 - ALPHA) * px2)
                        y2 = int(ALPHA * y2 + (1 - ALPHA) * py2)
                    smooth[tid] = (x1, y1, x2, y2)
                    hist_miss[tid] = 0
<<<<<<< HEAD
                    raw_box = (int(x1), int(y1), int(x2), int(y2))
                    x1, y1, x2, y2 = apply_area_limit(
                        *raw_box, area_mode)
=======
                    x1, y1, x2, y2 = apply_area_limit(
                        int(x1), int(y1), int(x2), int(y2), area_mode)
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
                    now_on = bool(global_on.get(tid, False))
                    should = (not now_on) if invert else now_on
                    mf = None
                    if masks_full is not None and i < len(masks_full):
                        try:
                            mfull = cv2.resize(
                                masks_full[i],
                                (frame.shape[1], frame.shape[0]))
                            mfull = (mfull > 0.5).astype(np.uint8) * 255
                            if area_mode != "full":
                                tmp = np.zeros_like(mfull)
                                tmp[max(0, y1):y2, max(0, x1):x2] = 255
                                mfull = cv2.bitwise_and(
                                    mfull, mfull, mask=tmp)
                            hist_mask[tid] = mfull
                            mf = mfull
                        except Exception:
                            mf = None
                    else:
                        mf = hist_mask.get(tid)
                    items.append((tid, x1, y1, x2, y2, should, mf,
                                  raw_box))

                keep_union = None
                for (_tid, _x1, _y1, _x2, _y2, _should, _mf,
                     _raw) in items:
                    if _should or _mf is None:
                        continue
                    try:
                        b = (_mf > 127).astype(np.uint8) * 255
                        b = cv2.dilate(b, np.ones((5, 5), np.uint8),
                                       iterations=1)
                        keep_union = (b if keep_union is None
                                      else cv2.bitwise_or(keep_union, b))
                    except Exception:
                        pass

                for (tid, x1, y1, x2, y2, should, mf, raw_box) in items:
                    # 명찰·얼굴 등 선택된 사람 부착 항목은 '가리는 범위'와
                    # 무관하게 VLM이 찾은 위치를 직접 가린다.
                    if direct_cats and not invert:
                        for cat in direct_cats:
                            rels = _nearest_rel_boxes(
                                person_box_samples.get(cat), sidx, tid)
                            if _render_rel_boxes(frame, raw_box, rels,
                                                 style, blocks, blur):
                                rendered_tids.add(tid)
                    if not should:
                        continue
                    if mf is not None and keep_union is not None:
                        try:
                            mf = cv2.bitwise_and(
                                mf, mf, mask=cv2.bitwise_not(keep_union))
                        except Exception:
                            pass
                    if render_person_strict(
                            frame, x1, y1, x2, y2, mf,
                            style=style, blocks=blocks, blur=blur):
                        rendered_tids.add(tid)

            # 트랙이 잠깐 사라진 프레임도 같은 카테고리면 계속 가린다.
            for tid in list(smooth.keys()):
                if tid in seen:
                    continue
                hist_miss[tid] += 1
                if hist_miss[tid] > MAX_MISS:
                    smooth.pop(tid, None)
                    hist_mask.pop(tid, None)
                    hist_miss.pop(tid, None)
                    continue
                x1, y1, x2, y2 = smooth[tid]
                raw_box = (int(x1), int(y1), int(x2), int(y2))
                if direct_cats and not invert:
                    for cat in direct_cats:
                        rels = _nearest_rel_boxes(
                            person_box_samples.get(cat), sidx, tid)
                        if _render_rel_boxes(frame, raw_box, rels,
                                             style, blocks, blur):
                            rendered_tids.add(tid)
                if not invert and not global_on.get(tid, False):
                    continue
                if invert and global_on.get(tid, False):
                    continue
<<<<<<< HEAD
                x1, y1, x2, y2 = apply_area_limit(
                    *raw_box, area_mode)
=======
                x1, y1, x2, y2 = smooth[tid]
                x1, y1, x2, y2 = apply_area_limit(
                    int(x1), int(y1), int(x2), int(y2), area_mode)
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
                _mf = hist_mask.get(tid)
                if _mf is not None and keep_union is not None:
                    try:
                        _mf = cv2.bitwise_and(
                            _mf, _mf, mask=cv2.bitwise_not(keep_union))
                    except Exception:
                        pass
                if render_person_strict(
                        frame, x1, y1, x2, y2, _mf,
                        style=style, blocks=blocks, blur=blur):
                    rendered_tids.add(tid)

<<<<<<< HEAD
            # 사람 트랙에 매칭되지 않은 사람 부착 박스도 절대 좌표로 가린다.
            if direct_cats and not invert:
                for cat in direct_cats:
                    boxes = _nearest_abs_boxes(
                        person_box_samples.get(cat), sidx)
                    for bx in _dedupe_boxes(
                            [{"xyxy": b} for b in boxes]):
                        render_region(frame, *bx["xyxy"], style=style,
                                      blocks=blocks, blur=blur, pad=4)

            # 배경 영역 렌더 (체크리스트 VLM 박스 + 처리 중 VLM 박스)
=======
            # 배경 영역 렌더 (체크리스트 VLM 박스 + 처리 중 VLM 박스)
            sidx = done // keyframe_stride
>>>>>>> a541d4dd35d9df26870696446f01c6e8cfc21b72
            for cat in region_cats:
                boxes = _sample_boxes(region_samples.get(cat), sidx)
                boxes = _dedupe_boxes(boxes)
                for b in boxes:
                    x1, y1, x2, y2 = b["xyxy"]
                    if render_region(frame, x1, y1, x2, y2,
                                     style=style, blocks=blocks, blur=blur,
                                     pad=8):
                        rendered_region_cats.add(cat)

            out.write(frame)
            done += 1
            if progress_fn and total > 0:
                try:
                    progress_fn(0.5 + 0.5 * min(1.0, done / max(1, total)),
                                done, total, time.time() - t0)
                except Exception:
                    pass

        out.release()
        if done == 0:
            cleanup_tmp(tmp_noaudio)
            raise RuntimeError("처리된 프레임이 없습니다.")
        try:
            mux_audio_h264(tmp_noaudio, input_path, output_path,
                           preset=encode_preset)
        except Exception:
            cleanup_tmp(output_path)
            shutil.copy(tmp_noaudio, output_path)
        finally:
            cleanup_tmp(tmp_noaudio)

        stats = {}
        for tid in votes:
            _by_cat = votes[tid]
            stats[tid] = {
                "mosaic": tid in rendered_tids,
                "frames": (sum(len(v) for v in _by_cat.values())
                           if isinstance(_by_cat, dict) else 0),
                "thumb": first_thumb.get(tid),
                "selector": "vlm",
                "query_raw": query_raw,
                "targets": list(targets),
                "hit_targets": global_hit_targets.get(tid, []),
                "cat_on": {k: v for k, v in cat_on.get(tid, {}).items()
                           if v},
                "frac": global_detail.get(tid, {}).get("frac", 0.0),
                "sample": global_detail.get(tid, {}).get("sample", {}),
            }
        stats["_regions"] = len(rendered_region_cats)
        return stats


def mosaic_video(input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, progress_fn=None,
                 query_raw="", api_key=None, vlm_model=VLM_MODEL,
                 vlm_base_url=None, region_priors=None):
    """호환 래퍼."""
    job = MosaicJob(input_path, output_path, targets=targets, invert=invert,
                    conf=conf, model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur, ultra=ultra,
                    area_mode=area_mode, encode_preset=encode_preset,
                    keyframe_stride=keyframe_stride, query_raw=query_raw,
                    api_key=api_key, vlm_model=vlm_model,
                    vlm_base_url=vlm_base_url,
                    region_priors=region_priors)
    return job.run(progress_fn)
