"""2패스 파이프라인 (개인정보 보호용).
1패스는 모든 프레임을 추적해 트랙 ID를 고정하고, 키프레임에서만 VLM
개인정보 탐지를 수행해 (사람 트랙, 카테고리)별로 수집한다.
2패스는 같은 추적 시퀀스로 선택된 카테고리만 렌더링한다.
사람 항목은 사람 마스크/얼굴 영역, 배경 항목(번호판·간판 등)은 VLM이 찾은
사각형 영역으로 가린다. 명찰처럼 사람에 붙은 항목은 VLM 박스를 사람 기준
상대 좌표로 저장해 '가리는 범위'와 무관하게 직접 가린다.
체크리스트에서 VLM이 찾은 배경 박스(region_priors)를 처리 단계의 초기값으로
넘겨, 처리 중 VLM이 같은 프레임에서 놓쳐도 체크리스트와 결과가 어긋나지
않게 한다.
"""
import cv2
import json
import logging
import math
import numpy as np
import os
import pickle
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from .config import (CATEGORY_TYPE, ROOT_DIR, VLM_MODEL, canon_category,
                     category_type, plausible_box)
from .models import MODEL_LOCK, get_model, get_thread_model
from .detect import filter_contained, big_enough, iou, region_has_detail
from .faces import (FaceBox, FaceTracker, detect_faces,
                    detect_faces_rotated,
                    detect_faces_yolo,
                    detect_faces_in_person_crops, expand_face_box,
                    filter_faces_by_person, persons_without_faces,
                    scene_signature, is_scene_cut, _blend_angle)
from .render import (_face_ratio, render_person_strict, render_person_roi,
                     render_region, render_face_ellipse, render_face_polygon,
                     mux_audio_h264, apply_area_limit, cleanup_tmp,
                     region_pad_for, FFmpegFrameWriter, mosaic_block_size)
from .flow import RegionFlow, shift_box
from .face_mesh import FaceMeshRefiner, FacePolyCache
from .offline import (backfill_and_smooth, bridge_gaps, find_gap_ranges,
                      fill_same_tid_gaps, missing_track_boxes,
                      nearest_track_values, temporal_smooth)
from .manual import apply_manual, normalize_regions
from .errors import RenderCancelled
from .select import VLMSelector
from .vlm_budget import VLMBudget, classify_vlm_error

log = logging.getLogger("mosaic.pipeline")

# MOSAIC_DEBUG=1 이면 결과 영상에 박스/ID/출처를 색으로 그린다.
# 초록=YuNet 얼굴 트랙, 자홍=머리 보완, 노랑=YOLO 사람 트랙, 청록=VLM 박스
DEBUG_OVERLAY = os.environ.get("MOSAIC_DEBUG", "") == "1"
# 머리 보완 마스크는 기본 OFF. YOLO 얼굴 검출기가 마스크/누운 얼굴을
# 잡으므로, 포즈 머리로 목/어깨를 덮는 과마스크를 막는다.
DRAW_HEAD_FALLBACK = os.environ.get("MOSAIC_DRAW_HEAD_FALLBACK", "0") == "1"


def _dbg_box(frame, box, color, label="", force=False):
    if not force and not DEBUG_OVERLAY:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, str(label), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


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


def _prior_coords(boxes):
    """체크리스트의 [x1,y1,x2,y2] 목록을 안전한 정수 박스로 바꾼다."""
    out = []
    for xyxy in boxes or []:
        if len(xyxy) != 4:
            continue
        try:
            coords = [int(float(v)) for v in xyxy]
        except (TypeError, ValueError):
            continue
        if coords[2] - coords[0] < 4 or coords[3] - coords[1] < 4:
            continue
        out.append(coords)
    return out


def _sample_boxes(samples, sample_idx, frame_idx=None, stride=1,
                  interpolate=False):
    """렌더 프레임에 쓸 VLM 박스 묶음을 돌려준다.

    체크리스트 priors는 실제 프레임(``due``)에 흩어져 있고 샘플 버킷
    인덱스와 정확히 맞지 않는다. 버킷 인덱스로만 고르면 다음 priors보다
    이른 프레임이 빈 목록이 되어 번호판이 깜빡인다. 그래서 이 함수는
    ``due`` 기준으로 이전 앵커와 다음 앵커를 고르고, 그 사이를 보간한다.
    due가 없는 1패스 검출은 키프레임 인덱스(stride 배수)라 앵커 프레임을
    버킷 위치로 간주한다.
    """
    if not samples:
        return []
    keys = sorted(samples)
    if frame_idx is None:
        pick = None
        for k in keys:
            if k <= sample_idx:
                pick = k
            else:
                break
        if pick is None:
            return []
        return list(samples[pick])

    stride = max(1, int(stride))
    anchors = []
    for k in keys:
        fallback = int(k) * stride
        for entry in samples[k]:
            due = entry.get("due", fallback)
            try:
                due = int(due)
            except (TypeError, ValueError):
                due = fallback
            anchors.append((due, int(k), entry))
    if not anchors:
        return []

    due_keys = sorted({due for due, _k, _e in anchors})
    prev_due = None
    nxt_due = None
    for due in due_keys:
        if due <= int(frame_idx):
            prev_due = due
        else:
            nxt_due = due
            break
    if prev_due is None:
        # 첫 앵커 이전 프레임은 다음 앵커가 가까우면 미리 가린다.
        # 너무 이전(45프레임 초과)은 다른 장면일 가능성이 커서 가리지 않는다.
        if nxt_due is not None:
            pre_hold = min(45, min(180, max(60, 12 * stride)))
            if nxt_due - int(frame_idx) <= pre_hold:
                return [entry for due, _k, entry in anchors
                        if due == nxt_due]
        return []
    # 마지막 앵커 이후 오래 지나면 대상을 지나간 것으로 본다. 번호판/간판이
    # 화면을 떠난 뒤 같은 자리에 고스트 모자이크가 남는 것을 막는다.
    # 프라이버시 우선이라 번호판은 더 길게 유지한다.
    max_hold = min(180, max(60, 12 * stride))
    if nxt_due is None and int(frame_idx) - prev_due > max_hold:
        return []
    prev = [entry for due, _k, entry in anchors if due == prev_due]
    if not interpolate or nxt_due is None:
        return prev
    gap = nxt_due - prev_due
    max_gap = min(180, max(60, 12 * stride))
    if gap <= 0 or gap > max_gap:
        # 앵커 사이가 너무 멀면 다른 장면의 다른 물체일 가능성이 크다.
        # 이전 앵커를 고집하지 말고, 가까운 쪽만 쓴다.
        if int(frame_idx) - prev_due > max_hold:
            if nxt_due - int(frame_idx) <= max_hold:
                return [entry for due, _k, entry in anchors
                        if due == nxt_due]
            return []
        return prev
    t = (int(frame_idx) - prev_due) / float(gap)
    if t <= 0.0:
        return prev
    nxt = [entry for due, _k, entry in anchors if due == nxt_due]
    max_dist = min(3.0, 1.5 + (gap / float(max(1, stride))) * 0.15)
    return _interpolate_boxes(prev, nxt, min(1.0, t), max_dist=max_dist)


def _flow_region_boxes(flow, cat, samples, sidx, done, stride):
    """앵커 박스를 광학 흐름으로 따라가게 하고, 보간 박스와 합친다."""
    anchors = _sample_boxes(samples, sidx, frame_idx=done, stride=stride,
                            interpolate=False)
    tracked = flow.track(cat, anchors)
    interp = _sample_boxes(samples, sidx, frame_idx=done, stride=stride,
                           interpolate=True)
    return RegionFlow.merge(tracked, interp)


def _interpolate_boxes(prev_boxes, next_boxes, t, max_dist=2.5):
    """앞뒤 키프레임 박스를 크기/거리로 짝지어 선형 보간한다.

    짝을 못 찾은 이전 박스는 그대로 유지하고, 다음 키프레임에만 있는
    새 박스는 아직 이 프레임에 없을 수 있으므로 추가하지 않는다.
    두 키프레임 사이에서 물체가 움직인 거리에 비례해 박스를 조금 넓힌다.
    """
    if not prev_boxes or not next_boxes:
        return prev_boxes
    max_dist = max(0.5, float(max_dist))
    used = set()
    out = []
    for p in prev_boxes:
        pxy = p["xyxy"]
        pw = max(1.0, pxy[2] - pxy[0])
        ph = max(1.0, pxy[3] - pxy[1])
        pcx, pcy = (pxy[0] + pxy[2]) / 2.0, (pxy[1] + pxy[3]) / 2.0
        best = None
        best_score = None
        for j, n in enumerate(next_boxes):
            if j in used:
                continue
            nxy = n["xyxy"]
            nw = max(1.0, nxy[2] - nxy[0])
            nh = max(1.0, nxy[3] - nxy[1])
            ratio = (nw * nh) / max(1.0, pw * ph)
            if ratio < 0.25 or ratio > 4.0:
                continue
            ncx, ncy = (nxy[0] + nxy[2]) / 2.0, (nxy[1] + nxy[3]) / 2.0
            scale = max(pw, ph, nw, nh)
            dist = math.hypot(ncx - pcx, ncy - pcy) / max(1.0, scale)
            if dist > max_dist:
                continue
            if best_score is None or dist < best_score:
                best, best_score = j, dist
        if best is None:
            out.append(p)
            continue
        used.add(best)
        nxy = next_boxes[best]["xyxy"]
        xy = [int(round(pxy[i] + (nxy[i] - pxy[i]) * t))
              for i in range(4)]
        motion = float(best_score or 0.0)
        if motion > 0.02 and xy[2] > xy[0] and xy[3] > xy[1]:
            pad = min(0.30, 0.05 + 0.25 * motion)
            padx = int(round((xy[2] - xy[0]) * pad))
            pady = int(round((xy[3] - xy[1]) * pad))
            xy = [xy[0] - padx, xy[1] - pady,
                  xy[2] + padx, xy[3] + pady]
        merged = dict(p)
        merged["xyxy"] = xy
        out.append(merged)
    return out


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


def _store_person_box(samples, cat, sample_idx, box, persons, due=0):
    """사람에 붙은 VLM 박스를 트랙 ID와 상대 좌표로 저장한다."""
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
        best_iou = 0.12
        for p in persons:
            overlap = iou(p["xyxy"], xyxy)
            if overlap >= best_iou:
                best, best_iou = p, overlap
    tid = int(best["tid"]) if best is not None else -1
    if best is None and cat == "face":
        return
    if best is not None and cat in ("face", "name_tag"):
        px1, py1, px2, py2 = best["xyxy"]
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        iw, ih = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        if iw > pw * 1.15 or ih > ph * 0.95:
            return
        max_frac = 0.55 if cat == "face" else 0.35
        if (iw * ih) > (pw * ph) * max_frac:
            return
    rel = _relative_box(xyxy, best["xyxy"]) if best is not None else None
    entries = samples[cat][sample_idx]
    for e in entries:
        if e["tid"] == tid and iou(e["abs"], xyxy) >= 0.55:
            return
    entries.append({"tid": tid, "rel": rel, "abs": xyxy, "due": int(due)})


def _nearest_rel_boxes(samples, sample_idx, tid, frame_idx=None):
    """해당 트랙에서 가장 최근에 찾은 사람 부착 박스의 상대 좌표."""
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        rels = [e["rel"] for e in samples[k]
                if e["tid"] == tid and e.get("rel")
                and (frame_idx is None
                     or int(e.get("due", 0) or 0) <= int(frame_idx))]
        if rels:
            return rels
    return []


def _missing_abs_boxes(samples, sample_idx, active_tids, max_age=2,
                       frame_idx=None):
    """현재 2패스에서 보이지 않는 트랙(tid=-1 포함)의 절대 박스."""
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        if sample_idx - k > max_age:
            return []
        out = []
        for e in samples[k]:
            tid = int(e.get("tid", -1))
            if frame_idx is not None and int(e.get("due", 0) or 0) > frame_idx:
                continue
            if tid < 0 or tid not in active_tids:
                if e.get("abs"):
                    out.append((tid, e["abs"]))
        return out
    return []


def _render_rel_boxes(frame, raw_box, rels, style, blocks, blur,
                      area_mode="full", clip_area=True,
                      clip_to_person=True, cat=None):
    """사람 박스 기준 상대 좌표를 현재 프레임 좌표로 바꿔 가린다.

    얼굴은 VLM 좌표를 믿지 않는다. VLM face 박스는 몸통·손·배경을 크게
    감싸는 경우가 많아, 얼굴은 로컬 YuNet/포즈 경로에서만 그린다.
    동승자 박스는 사람 마스크가 이미 그려진 경우 호출부에서 건너뛴다.
    """
    if not rels or cat == "face":
        return False
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = [int(v) for v in raw_box]
    ax1, ay1, ax2, ay2 = apply_area_limit(*raw_box, area_mode)
    rendered = False
    for rel in rels:
        bx = _absolute_box(rel, raw_box, w, h)
        if clip_to_person:
            bx = [max(bx[0], rx1), max(bx[1], ry1),
                  min(bx[2], rx2), min(bx[3], ry2)]
        if clip_area and area_mode != "full":
            bx = [max(bx[0], ax1), max(bx[1], ay1),
                  min(bx[2], ax2), min(bx[3], ay2)]
        if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
            continue
        ok = render_region(frame, *bx, style=style, blocks=blocks,
                           blur=blur, pad=4)
        if ok:
            rendered = True
    return rendered


def _poly_roi_mask(poly, x1, y1, x2, y2, grow=1):
    """원본 좌표 폴리곤(Results.masks.xy)을 ROI 크기 마스크로 그린다.

    r.masks.data 는 레터박스된 추론 해상도라 프레임과 1:1이 아니다.
    masks.xy 는 이미 원본 좌표이므로 보정이 필요 없고, 서브픽셀 AA 로
    계단 현상이 없다. grow 기본값은 1(팽창 없음)로 두어 사람 바깥 배경으로
    모자이크가 번지지 않게 한다.
    """
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    if w < 1 or h < 1 or poly is None or len(poly) < 3:
        return None
    pts = np.round((np.asarray(poly, np.float32)
                    - np.array([x1, y1], np.float32)) * 16.0).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA, shift=4)
    m = (m > 96).astype(np.uint8) * 255
    if grow and grow > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (grow, grow)))
    return m


def _detect_heads_pose(model, frame, conf=0.18, imgsz=640, kconf=0.35,
                       return_persons=False):
    """yolo-pose 키포인트(코/눈/귀)로 '얼굴 크기'의 박스를 만든다.

    사람 박스 상단을 머리로 가정하던 예전 방식과 달리, 실제로 보이는
    눈·코·귀가 있을 때만 만든다. 뒤통수/손/어깨처럼 얼굴이 없는 영역에는
    박스를 만들지 않아 엉뚱한 모자이크를 막는다.

    return_persons=True면 (heads, person_boxes)를 돌려준다. person_boxes는
    개/고양이 얼굴 오검출을 거르는 데 쓴다.
    """
    if model is None or frame is None:
        return ([], []) if return_persons else []
    try:
        with MODEL_LOCK:
            pr = list(model.predict(source=frame, conf=conf, verbose=False,
                                    imgsz=int(imgsz), max_det=30))[0]
        if pr.boxes is None or len(pr.boxes) == 0:
            return ([], []) if return_persons else []
        pboxes = pr.boxes.xyxy.cpu().numpy()
        kxy = kcf = None
        if getattr(pr, "keypoints", None) is not None:
            kxy = pr.keypoints.xy.cpu().numpy()
            kc = getattr(pr.keypoints, "conf", None)
            kcf = kc.cpu().numpy() if kc is not None else None
    except Exception:
        log.exception("pose 머리 검출 실패")
        return ([], []) if return_persons else []
    if kxy is None:
        persons = [[float(v) for v in pb[:4]] for pb in pboxes]
        return ([], persons) if return_persons else []
    h, w = frame.shape[:2]
    out = []
    for i, pb in enumerate(pboxes):
        if i >= len(kxy):
            continue
        pts = {}
        for k in range(min(5, kxy.shape[1])):
            c = float(kcf[i][k]) if kcf is not None else 1.0
            x, y = float(kxy[i][k][0]), float(kxy[i][k][1])
            if c >= kconf and x > 0 and y > 0:
                pts[k] = (x, y, c)
        nose = pts.get(0)
        eye_r, eye_l = pts.get(1), pts.get(2)
        ear_r, ear_l = pts.get(3), pts.get(4)

        d = cx = cy = None
        if eye_r and eye_l:
            d = math.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1])
            cx = (eye_r[0] + eye_l[0]) / 2.0
            cy = (eye_r[1] + eye_l[1]) / 2.0
        elif ear_r and ear_l and nose:
            d = math.hypot(ear_l[0] - ear_r[0],
                           ear_l[1] - ear_r[1]) / 1.70
            cx = (ear_r[0] + ear_l[0]) / 2.0
            cy = (ear_r[1] + ear_l[1] + nose[1]) / 3.0
        elif nose and (eye_r or eye_l):
            eye = eye_r or eye_l
            ear = (ear_r if eye_r else ear_l)
            if ear:
                d = math.hypot(ear[0] - eye[0], ear[1] - eye[1]) / 0.95
                cx = (eye[0] + ear[0]) / 2.0
                cy = (eye[1] + ear[1] + nose[1]) / 3.0
            else:
                # 한쪽 눈만 보이면 코까지의 거리로 눈 간격을 추정한다.
                d = 1.7 * math.hypot(nose[0] - eye[0], nose[1] - eye[1])
                cx = eye[0]
                cy = eye[1]
        if d is None or d < 4.0:
            continue

        fw = 2.05 * d
        head = (cx - fw / 2.0, cy - 1.00 * d,
                cx + fw / 2.0, cy + 2.10 * d)
        # 세로로 지나치게 긴 박스는 목/가슴까지 덮는 오검출로 본다.
        if (head[3] - head[1]) > 1.9 * (head[2] - head[0]):
            continue
        # 사람 박스보다 지나치게 큰 '머리'는 키포인트 오검출로 본다.
        pbw = float(pb[2] - pb[0])
        pbh = float(pb[3] - pb[1])
        if (head[2] - head[0]) > 0.95 * pbw + 8:
            continue
        if (head[3] - head[1]) > 0.85 * pbh + 8:
            continue
        x1, y1 = max(0, int(head[0])), max(0, int(head[1]))
        x2, y2 = min(w, int(head[2])), min(h, int(head[3]))
        if x2 - x1 >= 12 and y2 - y1 >= 12:
            out.append((x1, y1, x2, y2))
    if return_persons:
        persons = [[float(v) for v in pb[:4]] for pb in pboxes]
        return out, persons
    return out


def _faces_in_person(faces, box, margin=0.12):
    """사람 박스(약간 확장) 안에 중심이 있는 얼굴 박스 목록."""
    x1, y1, x2, y2 = [float(v) for v in box]
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    out = []
    for f in faces or []:
        cx, cy = (f[0] + f[2]) / 2.0, (f[1] + f[3]) / 2.0
        if x1 - mw <= cx <= x2 + mw and y1 - mh <= cy <= y2 + mh:
            out.append(f)
    return out


def _small_head_fallback_box(raw_box, W, H):
    """포즈 모델이 없을 때만 쓰는 보수적인 머리 위치 추정.

    몸을 숙인 사람/바닥 반사/다리 조각처럼 사람 박스 상단이 머리가 아닌
    경우를 걸러내고, 사람 박스 높이의 13~20%만 머리로 본다.
    """
    try:
        px1, py1, px2, py2 = [int(v) for v in raw_box[:4]]
    except (TypeError, ValueError):
        return None
    pw, ph = px2 - px1, py2 - py1
    if pw < 18 or ph < 30:
        return None
    ar = ph / float(max(1, pw))
    # 사람 박스 상단이 화면 아래쪽이면 바닥 반사/하체 조각일 가능성이 크다.
    if py1 > 0.58 * float(H):
        return None
    if ar < 1.25:
        return None
    hh = int(round(ph * (0.13 if ar >= 1.8 else 0.20)))
    hw = int(round(min(pw * 0.32, hh * 0.80)))
    hh = max(20, hh)
    hw = max(16, hw)
    cx = (px1 + px2) // 2
    y1 = max(0, py1 + int(ph * 0.02))
    box = [max(0, cx - hw // 2), y1,
           min(int(W), cx + hw // 2), min(int(H), y1 + hh)]
    if box[2] - box[0] < 16 or box[3] - box[1] < 20:
        return None
    return box


def _head_fallback_box(raw_box, mf=None, band=None):
    """얼굴 검출이 없을 때 사람 박스에서 '얼굴 크기'의 박스를 추정한다."""
    x1, y1, x2, y2 = [int(v) for v in raw_box]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    hh = int(h * _face_ratio(w, h) * 0.75)
    hw = int(min(w * 0.5, hh * 0.42))
    cx = (x1 + x2) // 2
    if mf is not None and band is not None:
        try:
            rows = max(1, int(mf.shape[0] * 0.5))
            ys, xs = np.nonzero(mf[:rows] > 127)
            if len(xs) >= 20:
                cx = int(band[0] + float(np.mean(xs)))
        except Exception:
            pass
    box = [cx - hw, y1 - int(0.02 * h),
           cx + hw, y1 + int(hh * 1.02)]
    return box


def _blend_mask(cur, prev):
    """직전 프레임 마스크와 섞어 경계 떨림을 줄이되, 바깥 배경으로 번지지 않게 한다."""
    if cur is None or prev is None:
        return cur
    if prev.shape != cur.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    m = cv2.addWeighted(cur, 0.75, prev, 0.25, 0)
    return (m > 115).astype(np.uint8) * 255


def _clip_head(box, W, H):
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(int(W), int(box[2])), min(int(H), int(box[3]))
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def _face_rotate_enabled():
    return os.environ.get("MOSAIC_FACE_ROTATE", "1") != "0"


def _face_rotate_every():
    try:
        return max(1, int(os.environ.get("MOSAIC_FACE_ROTATE_EVERY", "3")))
    except (TypeError, ValueError):
        return 3


def _padded_head_box(frame_bgr, box, side=0.20, top=0.45, bottom=0.20):
    return expand_face_box(box, frame_bgr.shape[1], frame_bgr.shape[0],
                           side=side, top=top, bottom=bottom)


def _poly_covers_box(poly, box, min_ratio=0.80):
    """mesh 폴리곤이 얼굴/머리 박스를 충분히 덮는지 확인한다.

    옆얼굴/누운 얼굴에서 랜드마크 hull이 얼굴 한쪽만 덮으면, 그대로
    모자이크를 그렸을 때 반대쪽 볼·눈이 노출된다. 이 경우 박스형
    안전 마스크로 대체한다.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 - x1 < 4 or y2 - y1 < 4:
            return False
        box_poly = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        inter, _ = cv2.intersectConvexConvex(
            np.asarray(poly, np.float32).reshape(-1, 1, 2),
            box_poly.reshape(-1, 1, 2))
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return float(inter) / area >= float(min_ratio)
    except Exception:
        return False


def _poly_size_ok(poly, box, max_scale=1.6, max_area=2.2):
    """mesh 폴리곤이 얼굴 박스보다 지나치게 크게 번지지 않았는지 확인한다.

    뒤통수/머리카락에서 랜드마크가 크게 튀면 폴리곤이 몸 전체를 덮는
    거대한 모자이크가 될 수 있다. 얼굴 박스의 2배를 넘으면 버린다.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        px1, py1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        px2, py2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        if (px2 - px1) > max_scale * bw:
            return False
        if (py2 - py1) > max_scale * bh:
            return False
        area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
        return area <= max_area * bw * bh
    except Exception:
        return False


def _face_upper_ok(box, persons, max_frac=0.42, frame_h=None,
                   heads=None):
    """사람 박스 안의 얼굴이 상단에 있는지 확인한다.

    바닥 반사/신발/팔/다리 조각을 얼굴로 잡으면 사람 박스 안이더라도
    하단에 위치한다. 실제 얼굴은 (몸을 숙여도) 사람 박스 위쪽에 있거나,
    아래쪽이면 포즈 머리와 겹친다. 팔/손 오검출은 포즈 머리와 겹치지
    않아 여기서 걸러진다.
    """
    try:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
    except (TypeError, ValueError, IndexError):
        return True
    heads = list(heads or [])
    inside_any = False
    for p in persons or []:
        try:
            px1, py1, px2, py2 = [float(v) for v in p[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            inside_any = True
            if (py2 - py1) <= 1.05 * max(1.0, px2 - px1):
                rel = (cy - py1) / max(1.0, py2 - py1)
                if rel <= 0.38:
                    continue
            else:
                rel = (cy - py1) / max(1.0, py2 - py1)
                if rel <= max_frac:
                    continue
            if any(_overlap_ratio(box, h) > 0.50 for h in heads):
                continue
            return False
    if not inside_any and frame_h:
        return cy <= 0.75 * float(frame_h)
    return True


def _in_wide_person(box, persons):
    if not persons:
        return False
    try:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
    except (TypeError, ValueError, IndexError):
        return False
    for p in persons:
        try:
            px1, py1, px2, py2 = [float(v) for v in p[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        if (px1 <= cx <= px2 and py1 <= cy <= py2
                and (px2 - px1) > 1.15 * max(1.0, py2 - py1)):
            return True
    return False


def _force_face_ok(box, persons, frame_h, score=0.0):
    """YOLO 강제 마스크가 얼굴 위치인지 확인한다."""
    try:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
    except (TypeError, ValueError, IndexError):
        return False
    if not persons:
        return cy <= 0.75 * float(frame_h)
    for p in persons:
        try:
            px1, py1, px2, py2 = [float(v) for v in p[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
        rel = (cy - py1) / max(1.0, py2 - py1)
        # 고신뢰 YOLO 얼굴은 누운 사람/숙인 사람처럼 사람 박스 안에서
        # 아래쪽에 있어도 그린다. 팔/가슴 오검출은 점수가 낮아 여기까지
        # 오지 않는다(호출 전 0.55 미만 제외).
        if score >= 0.65:
            return rel <= 0.60
        wide = (px2 - px1) > 1.15 * max(1.0, py2 - py1)
        return rel <= (0.32 if wide else 0.22)
    return cy <= 0.75 * float(frame_h)


def _face_mask_margins(box):
    """얼굴 크기에 따른 마스크 여백.

    원거리 얼굴은 검출/추적 박스가 몇 px만 어긋나도 눈·이마가 노출된다.
    작을수록 여백을 키워 얼굴 전체를 확실히 덮고, 근접 얼굴은 옷/배경으로
    번지지 않게 기존의 작은 여백을 유지한다.
    """
    w = max(1.0, abs(float(box[2]) - float(box[0])))
    h = max(1.0, abs(float(box[3]) - float(box[1])))
    m = min(w, h)
    if m < 40:
        return 0.38, 0.75, 0.35
    if m < 72:
        return 0.20, 0.35, 0.20
    if m < 120:
        return 0.16, 0.28, 0.16
    return 0.14, 0.25, 0.15


def _render_face_box(frame, box, style, blocks, blur, angle=0.0,
                     mesh=None, motion=0.0, block_size=None):
    """랜드마크 외곽이 있으면 사람마다 다른 모양으로, 없으면 고정 얼굴형으로.

    mesh는 FaceMeshRefiner.shared() 또는 None. None이면 고정 얼굴형을 쓴다.
    motion(px)은 빠르게 움직일 때 여백으로 더한다.
    """
    if mesh is not None:
        try:
            poly = mesh.polygon_for_box(frame, box)
        except Exception:
            poly = None
        if poly is not None:
            if not _poly_covers_box(
                    poly, _padded_head_box(frame, box), 0.80):
                poly = None
        if poly is not None and not _poly_size_ok(poly, box):
            poly = None
        if poly is not None:
            if motion > 0:
                try:
                    m = min(float(motion), 12.0)
                    c = poly.mean(axis=0, keepdims=True)
                    d = poly - c
                    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
                    poly = (poly + d / n * m).astype(np.int32)
                except Exception:
                    pass
            if render_face_polygon(
                    frame, poly, style=style, blocks=blocks, blur=blur,
                    block_size=block_size):
                return True
    side, top, bottom = _face_mask_margins(box)
    bx = expand_face_box(box, frame.shape[1], frame.shape[0], angle=angle,
                        motion=motion, side=side, top=top, bottom=bottom)
    if not bx:
        return False
    return bool(render_face_ellipse(
        frame, *bx, style=style, blocks=blocks, blur=blur, angle=angle,
        block_size=block_size))


def _head_area(px1, py1, px2, py2, ratio=0.38, inset=0.04):
    """사람 박스에서 머리(상단) 영역을 대략 잡는다."""
    hh = max(1, int(py2) - int(py1))
    hw = max(1, int(px2) - int(px1))
    hx1 = int(px1) + int(hw * inset)
    hx2 = int(px2) - int(hw * inset)
    hy1 = int(py1)
    hy2 = int(py1) + int(hh * ratio)
    if hx2 - hx1 < 12 or hy2 - hy1 < 12:
        return None
    return [hx1, hy1, hx2, hy2]


def _overlap_ratio(a, b):
    """두 박스의 겹침 비율(작은 쪽 넓이 기준)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
    ba = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    return inter / float(max(1, min(aa, ba)))


def _region_area_ok(box, w, h, max_ratio=0.08):
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area <= max(0.001, float(max_ratio)) * float(w * h)
    except (TypeError, ValueError):
        return False


def _region_on_person(box, persons, side=0.25, top=0.10, bottom=0.60):
    """영역 박스 중심이 사람(옷/가방/다리 포함) 근처인지."""
    if not persons:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    except (TypeError, ValueError):
        return False
    for p in persons:
        px1, py1, px2, py2 = [float(v) for v in p[:4]]
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 16:
            continue
        if (px1 - pw * side <= cx <= px2 + pw * side
                and py1 - ph * top <= cy <= py2 + ph * bottom):
            return True
    return False


def _face_person_size_ok(box, persons, max_w=0.75, max_h=0.55):
    """얼굴 박스가 사람 박스 대비 지나치게 크면 오검출로 본다.

    검은 패딩/옷 전체를 얼굴로 잡으면 모자이크가 상체 전체를 덮는다.
    사람 안에 중심이 있는 얼굴은 사람 폭/높이의 일정 비율 이하여야 한다.
    """
    if not persons:
        return True
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        fw, fh = x2 - x1, y2 - y1
    except (TypeError, ValueError):
        return False
    for p in persons:
        px1, py1, px2, py2 = [float(v) for v in p[:4]]
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 16:
            continue
        if (px1 <= cx <= px2 and py1 <= cy <= py2):
            return fw <= pw * float(max_w) and fh <= ph * float(max_h)
    return True


def _detect_person_boxes(model, frame, conf=0.20, imgsz=640):
    """사람 박스만 찾는다. 머리 영역 fallback 에만 쓴다."""
    if model is None or frame is None:
        return []
    try:
        with MODEL_LOCK:
            pr = list(model.predict(source=frame, conf=conf, classes=[0],
                                    verbose=False, imgsz=int(imgsz),
                                    max_det=30))[0]
        if pr.boxes is None or len(pr.boxes) == 0:
            return []
        xy = pr.boxes.xyxy.cpu().numpy().astype(int)
    except Exception:
        return []
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in xy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 16 or y2 - y1 < 24:
            continue
        out.append((x1, y1, x2, y2))
    return out


# ---------------------------------------------------------------------
# 1패스 검출 기록/재생 (상반신/전체 모드의 진짜 2패스)
# ---------------------------------------------------------------------
class _FakeTensor:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeBoxes:
    def __init__(self, xyxy, ids):
        self.xyxy = _FakeTensor(np.asarray(xyxy, np.float32).reshape(-1, 4))
        self.id = _FakeTensor(np.asarray(ids, np.int64))
        self._n = len(self.xyxy.a)

    def __len__(self):
        return self._n


def _simplify_poly(poly):
    try:
        p = np.asarray(poly, np.float32)
        if len(p) < 3:
            return p
        a = cv2.approxPolyDP(p.reshape(-1, 1, 2), 0.8, True)
        return a.reshape(-1, 2) if len(a) >= 3 else p
    except Exception:
        return np.asarray(poly, np.float32)


class _DetRecorder:
    """1패스 추적 결과(박스/ID/폴리곤)를 프레임별로 저장한다.

    2패스가 같은 결과를 재생하므로 추론을 한 번만 하고, 트랙 ID가
    1·2패스에서 달라지는 문제가 없으며, 저장된 박스를 앞뒤 프레임으로
    후처리(역방향 채움, ID 전환 구간 보간)할 수 있다.
    """

    def __init__(self):
        self.frames = []

    def add(self, r):
        boxes = ids = polys = None
        try:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                ids = (r.boxes.id.cpu().numpy().astype(np.int64)
                       if r.boxes.id is not None
                       else np.full((len(boxes),), -1, np.int64))
                if r.masks is not None:
                    xy = list(r.masks.xy)
                    if len(xy) == len(boxes):
                        polys = [_simplify_poly(q) for q in xy]
        except Exception:
            log.exception("검출 기록 실패 - 이 프레임은 빈 검출로 저장")
            boxes = ids = polys = None
        self.frames.append({"b": boxes, "i": ids, "p": polys})

    def finalize(self):
        """비어 있던 프레임/ID 전환 구간에 합성 검출을 추가한다."""
        per = []
        for f in self.frames:
            d = {}
            if f["b"] is not None:
                for b, t in zip(f["b"], f["i"]):
                    if int(t) >= 0:
                        d[int(t)] = [float(v) for v in b]
            per.append(d)
        try:
            extra = missing_track_boxes(per, back=2, max_gap=8)
        except Exception:
            log.exception("검출 후처리 실패 - 원본 검출만 사용")
            extra = {}
        added = 0
        for i, add in extra.items():
            f = self.frames[i]
            nb = np.array(list(add.values()), np.float32).reshape(-1, 4)
            ni = np.array(list(add.keys()), np.int64)
            n_old = 0 if f["b"] is None else len(f["b"])
            f["b"] = nb if f["b"] is None else np.vstack([f["b"], nb])
            f["i"] = ni if f["i"] is None else np.concatenate([f["i"], ni])
            old_p = f["p"] if f["p"] is not None else [None] * n_old
            f["p"] = list(old_p) + [None] * len(ni)
            added += len(ni)
        return added

    def replay(self, input_path):
        cap = cv2.VideoCapture(input_path)
        try:
            for f in self.frames:
                ok, frm = cap.read()
                if not ok:
                    break
                boxes = (_FakeBoxes(f["b"], f["i"])
                         if f["b"] is not None and len(f["b"]) else None)
                masks = (SimpleNamespace(xy=f["p"])
                         if boxes is not None and f["p"] is not None
                         else None)
                yield SimpleNamespace(orig_img=frm, boxes=boxes, masks=masks)
        finally:
            cap.release()


def _serialize_cache_json(cache):
    """pickle 대신 JSON으로도 저장할 수 있도록 키와 집합을 변환한다."""
    def _conv_defaultdict(dd):
        return {str(k): {str(sk): sv for sk, sv in v.items()}
                for k, v in dd.items()}

    return {
        "version": cache["version"],
        "meta": cache["meta"],
        "votes": _conv_defaultdict(cache["votes"]),
        "person_box_samples": _conv_defaultdict(cache["person_box_samples"]),
        "region_samples": _conv_defaultdict(cache["region_samples"]),
        "person_cats_seen": sorted(cache.get("person_cats_seen") or []),
        "all_detail": {str(k): v for k, v in (cache.get("all_detail") or {}).items()},
        "observed_types": dict(cache.get("observed_types") or {}),
    }


def _deserialize_cache_json(raw):
    if raw.get("version") != 2:
        raise ValueError("cache version mismatch")
    votes = defaultdict(lambda: defaultdict(list))
    for tid, by_cat in (raw.get("votes") or {}).items():
        for cat, lst in (by_cat or {}).items():
            votes[int(tid)][cat] = list(lst)
    p_samples = defaultdict(lambda: defaultdict(list))
    for cat, by_sidx in (raw.get("person_box_samples") or {}).items():
        for sidx, entries in (by_sidx or {}).items():
            p_samples[cat][int(sidx)] = list(entries)
    r_samples = defaultdict(lambda: defaultdict(list))
    for cat, by_sidx in (raw.get("region_samples") or {}).items():
        for sidx, entries in (by_sidx or {}).items():
            r_samples[cat][int(sidx)] = list(entries)
    all_detail = {int(k): v for k, v in (raw.get("all_detail") or {}).items()}
    return {
        "version": 2,
        "meta": raw["meta"],
        "votes": votes,
        "person_box_samples": p_samples,
        "region_samples": r_samples,
        "person_cats_seen": set(raw.get("person_cats_seen") or []),
        "all_detail": all_detail,
        "observed_types": dict(raw.get("observed_types") or {}),
    }


@dataclass
class _Pass1State:
    """run() 의 준비~1패스(캐시 포함) 결과. 2패스가 읽는 영상 정보도 담는다."""
    model: Any = None
    conf: float = 0.0
    imgsz: int = 0
    keyframe_stride: int = 1
    fast_person_only: bool = False
    needs_person_pass1: bool = False
    cache_hit: bool = False
    votes: Any = None
    person_box_samples: Any = None
    region_samples: Any = None
    person_cats_seen: Any = None
    all_detail: Any = None
    observed_types: Any = None
    pending_person_priors: Any = None
    total: int = 0
    fps: float = 0.0
    H: int = 0
    W: int = 0
    max_sidx: int = 0
    selector: Any = None
    vlm_budget: Any = None
    vlm_targets: Any = None
    vlm_errors: int = 0
    vlm_disabled: bool = False
    t0: float = 0.0
    replay_on: bool = True
    recorder: Any = None


@dataclass
class _Plan:
    """전역 확정 결과(어떤 카테고리/트랙을 가릴지)."""
    person_cats: Any = None
    region_cats: Any = None
    direct_cats: Any = None
    mask_cats: Any = None
    blanket_cats: Any = None
    cat_on: Any = None
    global_on: Any = None
    mask_on: Any = None
    global_hit_targets: Any = None
    global_detail: Any = None
    needs_person: bool = False
    face_only_person: bool = False


@dataclass
class _RenderState:
    """2패스 프레임 루프가 프레임 사이에 유지하는 상태."""
    smooth: Any = field(default_factory=dict)
    hist_mask: Any = field(default_factory=dict)
    hist_miss: Any = field(default_factory=lambda: defaultdict(int))
    rendered_tids: Any = field(default_factory=set)
    rendered_direct_tids: Any = field(default_factory=set)
    rendered_cats: Any = field(default_factory=set)
    untracked_cats: Any = field(default_factory=set)
    rendered_region_cats: Any = field(default_factory=set)
    track_frames: Any = field(default_factory=lambda: defaultdict(int))
    ALPHA: float = 0.85  # 지연 최소화(프라이버시 우선)
    MAX_MISS: int = 12
    done: int = 0
    progress_base: float = 0.5
    cap2: Any = None
    region_flow: Any = None
    face_trk: Any = None
    prev_sig_run: Any = None
    prev_gray_run: Any = None
    debug_boxes: Any = field(default_factory=list)
    plate_dets: Any = field(default_factory=list)
    plate_updated: int = -10 ** 9
    plate_tracker: Any = None
    prev_gray_plate: Any = None


@dataclass
class _FrameCtx:
    """프레임 1장을 그리는 동안만 쓰는 값."""
    frame: Any = None
    r: Any = None
    sidx: int = 0
    faces_now: Any = field(default_factory=list)
    drawn_faces: Any = field(default_factory=set)
    seen: Any = field(default_factory=set)
    items: Any = field(default_factory=list)
    keep_union: Any = None
    debug_boxes: Any = field(default_factory=list)


class MosaicJob:
    """개인정보 보호 모자이크 작업 1건."""

    def __init__(self, input_path, output_path, targets=None, invert=False,
                 conf=0.20, model_name="yolo11n-seg.pt", imgsz=480,
                 blocks=30, style="pixel", blur=61, ultra=False,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=15, query_raw="",
                 api_key=None, vlm_model=VLM_MODEL, vlm_base_url=None,
                 region_priors=None, person_priors=None,
                 cache_path=None, use_cache=False,
                 vlm_call_limit=None, mask_all_persons=False,
                 manual_regions=None, debug_path=None, cancel_fn=None,
                 mux_audio=True):
        self.input_path = input_path
        self.output_path = output_path
        self.targets = ([targets] if isinstance(targets, str)
                        else (list(targets) if targets else ["face"]))
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
        self.person_priors = person_priors or {}
        self.cache_path = cache_path
        self.use_cache = bool(use_cache)
        self.vlm_call_limit = (None if vlm_call_limit is None
                               else max(0, int(vlm_call_limit)))
        self.mask_all_persons = bool(mask_all_persons)
        self.manual_regions = normalize_regions(manual_regions)
        self.debug_path = debug_path or None
        self.cancel_fn = cancel_fn
        self.mux_audio = bool(mux_audio)

    def _open_debug_writer(self, width, height, fps, audio_src=None):
        """디버그 영상은 최종 결과와 분리해 별도 파일로 저장한다."""
        if not self.debug_path:
            return None
        try:
            return FFmpegFrameWriter(
                self.debug_path, width, height, fps, audio_src=audio_src,
                preset=self.encode_preset)
        except Exception:
            log.exception("디버그 영상 작성기 열기 실패")
            return None

    @staticmethod
    def _draw_debug_marks(frame, marks):
        for box, color, label in marks or []:
            _dbg_box(frame, box, color, label, force=True)

    def _apply_manual(self, frame, original, idx, fps):
        if not self.manual_regions:
            return frame
        st_, bl_, bu_ = self.style, self.blocks, self.blur
        return apply_manual(
            frame, original, idx, fps, self.manual_regions,
            lambda f, x1, y1, x2, y2: render_region(
                f, x1, y1, x2, y2, style=st_, blocks=bl_, blur=bu_, pad=0),
            lambda f, x1, y1, x2, y2: render_face_ellipse(
                f, x1, y1, x2, y2, style=st_, blocks=bl_, blur=bu_))

    def _face_only_mode(self):
        """얼굴만/얼굴+배경 항목을 로컬 얼굴 추적만으로 렌더할 수 있는지."""
        if (self.invert or self.area_mode != "face_only"
                or not self.mask_all_persons or "face" not in self.targets):
            return False
        return all(c == "face" or _cat_type(c) != "person"
                   for c in self.targets)

    def _run_face_only(self, progress_fn=None):
        """YuNet 얼굴 검출 + 짧은 추적으로 '얼굴만' 영상을 렌더한다."""
        input_path, output_path = self.input_path, self.output_path
        targets = list(self.targets)
        style, blocks, blur = self.style, self.blocks, self.blur
        stride = max(1, int(self.keyframe_stride))
        preset = self.encode_preset
        rotate_faces = _face_rotate_enabled()
        rotate_every = _face_rotate_every()
        face_priors = self.person_priors.get("face") or {}
        try:
            region_max_area = float(
                os.environ.get("MOSAIC_REGION_MAX_AREA", "0.08"))
        except (TypeError, ValueError):
            region_max_area = 0.08

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
        H, W = first.shape[:2]
        del first
        cap.release()

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        region_samples = defaultdict(lambda: defaultdict(list))
        for cat, by_frame in (self.region_priors or {}).items():
            if cat not in targets or _cat_type(cat) != "region":
                continue
            for fidx, boxes in (by_frame or {}).items():
                try:
                    fidx = int(fidx)
                except (TypeError, ValueError):
                    continue
                sidx = max(0, fidx // stride)
                for coords in _prior_coords(boxes):
                    region_samples[cat][sidx].append({
                        "xyxy": coords, "conf": 1.0, "category": cat,
                        "region_type": "region", "source": "vlm",
                        "matches_prompt": True, "due": fidx,
                    })

        # VLM face 박스는 신뢰할 수 있는 위치가 아니다. 사람 몸통·손·배경을
        # 얼굴로 크게 잡은 박스를 추적 초기값으로 넣으면 거대한 모자이크가
        # 생기므로, 얼굴 위치는 로컬 YuNet/포즈 보완만 사용한다.
        tracker = FaceTracker(
            max_miss=max(10, int(round(fps * 6.5))),
            seed_miss=max(6, int(round(fps * 1.0))))
        pose_model = None
        head_model = None
        if os.environ.get("MOSAIC_HEAD_POSE", "1") != "0":
            try:
                pose_model = get_model(
                    os.environ.get("MOSAIC_POSE_MODEL", "yolo11n-pose.pt"))
            except Exception:
                log.exception("pose 모델 로드 실패 - 사람 박스 방식으로 대체")
                pose_model = None
        if pose_model is None:
            try:
                head_model = get_model(self.model_name)
                _reset_tracker(head_model)
            except Exception:
                head_model = None
        # 머리 보완은 0.5초까지 유지한다. 60fps에서 3프레임 간격이면
        # 보행자 머리가 50px 이상 움직이므로 짧게 유지하면 매번 새 트랙이
        # 생겨 모자이크가 제자리에 남거나 깜빡인다.
        head_tracker = FaceTracker(
            max_miss=max(10, int(round(fps * 0.80))),
            seed_miss=max(4, int(round(fps * 0.30))),
            iou_thresh=0.12)
        try:
            mesh_refiner = FaceMeshRefiner.shared()
            if not mesh_refiner.available():
                mesh_refiner = None
        except Exception:
            mesh_refiner = None
        rendered_tids = set()
        track_frames = defaultdict(int)
        rendered_cats = set()
        rendered_region_cats = set()
        max_active = 0
        fallback_heads = 0
        prev_sig = None
        done = 0
        t0 = time.time()

        tmp_noaudio = output_path + ".noaudio.mp4"
        writer = None
        debug_writer = None
        out = None
        try:
            # ---------- 1패스: 프레임별 검출/추적 결과만 모은다 ----------
            face_frames, raw_face_frames, head_frames, cuts = [], [], [], set()
            person_frames = {}
            face_angles = defaultdict(dict)
            last_fresh = {}
            prev_gray_f1 = None
            prev_face_active = []
            heads = []
            last_persons, last_persons_idx = [], -10 ** 9
            # YOLO 강제 박스는 프레임마다 새로 검출되지만, 1~2프레임 미스에도
            # 마스크가 깜빡이지 않게 슬롯으로 유지한다. 슬롯 ID가 안정적이라
            # 후처리 보간/평활도 같은 얼굴로 인식한다.
            force_slots = {}
            force_next = 0
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if self.cancel_fn is not None and self.cancel_fn():
                        raise RenderCancelled()
                    idx = len(face_frames)
                    if is_scene_cut(prev_sig, frame):
                        tracker.reset()
                        head_tracker.reset()
                        prev_face_active = []
                        last_persons = []
                        force_slots.clear()
                        cuts.add(idx)
                    prev_sig = scene_signature(frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _pg = prev_gray_f1
                    flow_fn = ((lambda b, _p=_pg, _c=gray:
                                shift_box(_p, _c, b))
                               if (_pg is not None and idx not in cuts)
                               else None)

                    # YuNet이 이미 얼굴을 잡은 프레임에서는 머리 보완 검출을
                    # 3프레임마다 수행하고 그 사이는 head_tracker로 유지해
                    # 렌더 시간을 줄인다. 얼굴이 하나도 안 잡힌 프레임은
                    # 즉시 보완 검출을 수행한다.
                    run_head_det = (not prev_face_active) or (idx % 3 == 0)
                    if run_head_det:
                        persons = []
                        if pose_model is not None:
                            heads, persons = _detect_heads_pose(
                                pose_model, frame, return_persons=True)
                        elif head_model is not None:
                            persons = _detect_person_boxes(head_model, frame)
                            heads = [h for h in (
                                _head_area(*pb) for pb in persons) if h]
                        else:
                            heads = []
                        if persons:
                            last_persons = persons
                            last_persons_idx = idx
                        # YuNet 얼굴과 겹치는 머리도 일단 트랙에 넣는다.
                        # 그리기 단계에서 실제 얼굴과 겹치면 건너뛰므로
                        # 중복은 없고, YuNet이 놓친 프레임을 머리 트랙이
                        # 이어서 가릴 수 있다.
                        dets = [(hd, 0.7) for hd in heads]
                        head_tracker.update(dets, flow_fn=flow_fn)
                    else:
                        head_tracker.update([], flow_fn=flow_fn)
                    person_frames[idx] = (
                        list(last_persons)
                        if last_persons and idx - last_persons_idx <= 8
                        else [])

                    raw_dets = detect_faces(frame)
                    if rotate_faces and (idx % rotate_every == 0
                                         or not raw_dets):
                        raw_dets = detect_faces_rotated(
                            frame, base=raw_dets)
                    # 개/고양이 얼굴 등 사람이 아닌 큰 검출을 거른다.
                    # 점수가 매우 높으면(사람 얼굴) 그대로 두고, 애매한
                    # 검출만 YOLO 사람 박스 안에 있는지 확인한다.
                    if last_persons and idx - last_persons_idx <= 8:
                        h_f, w_f = frame.shape[:2]
                        raw_dets = filter_faces_by_person(
                            raw_dets, last_persons, min_score=0.95,
                            frame_area=float(w_f * h_f))
                        raw_dets = [
                            (b, s) for b, s in raw_dets
                            if _face_person_size_ok(b, last_persons)]
                    # YOLO 얼굴 검출기: YuNet이 놓친 마스크/누운 얼굴 보완.
                    # 위쪽(이마)만 잡힌 얼굴은 마스크까지 아래로 확장한다.
                    # 얼굴 후보 검증에도 같은 결과를 재사용한다(중복 추론 방지).
                    yolo_raw = []
                    try:
                        yolo_raw = detect_faces_yolo(frame)
                    except Exception:
                        log.exception("YOLO 얼굴 검출 실패")
                    # VLM 얼굴 후보는 YuNet이 놓친 큰/측면 얼굴을 보완한다.
                    # 옷 무늬/바퀴 오검출을 막기 위해 크기·종횡비를 검증한다.
                    if face_priors:
                        prior_boxes = (face_priors.get(str(idx))
                                       or face_priors.get(idx) or [])
                        h_f, w_f = frame.shape[:2]
                        _support = list(raw_dets) + list(yolo_raw)
                        for coords in _prior_coords(prior_boxes):
                            bx = [int(round(float(v))) for v in coords[:4]]
                            bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
                            if bw_ < 20 or bh_ < 20:
                                continue
                            # 큰 얼굴 후보는 실제 검출기(YuNet/YOLO)가
                            # 받쳐줘야 한다. 머리카락/목/옷을 얼굴로 잡은
                            # 큰 후보가 통째로 모자이크되는 것을 막는다.
                            _supported = any(
                                _overlap_ratio(bx, sb) > 0.30
                                for sb, _ss in _support)
                            # 검출기가 이미 잡은 얼굴은 후보 박스(대개 더
                            # 크고 배경을 포함) 대신 검출기/트랙에 맡긴다.
                            if _supported:
                                continue
                            if min(bw_, bh_) >= 40:
                                continue
                            ar = (bw_ * bh_) / float(max(1, w_f * h_f))
                            if ar < 0.0005 or ar > 0.05:
                                continue
                            if not (0.50 <= bw_ / float(max(1, bh_)) <= 2.0):
                                continue
                            if last_persons and not _face_person_size_ok(
                                    bx, last_persons):
                                continue
                            # 큰 VLM 얼굴 후보는 포즈 머리와 겹쳐야 한다.
                            # 사슴/포스터/옷을 얼굴로 잡은 큰 박스를 거른다.
                            if (min(bw_, bh_) >= 0.08 * min(w_f, h_f)
                                    and not any(
                                        _overlap_ratio(bx, h) > 0.20
                                        for h in heads)):
                                continue
                            if any(_overlap_ratio(bx, b) > 0.35
                                   for b, _s in raw_dets):
                                continue
                            if (last_persons and ar < 0.02
                                    and not any(
                                        _faces_in_person(
                                            [FaceBox(bx)], p, margin=0.10)
                                        for p in last_persons)):
                                continue
                            raw_dets.append((FaceBox(bx), 0.82))
                        raw_dets.sort(
                            key=lambda item: item[1], reverse=True)
                    # 얼굴이 아직 없는 사람(원거리 군중)은 사람 상단 크롭을
                    # 확대해 찾는다. 전경 얼굴이 1개라도 있으면 예전에는
                    # 이 보완이 아예 안 돌아 뒤쪽 군중이 계속 빠졌다.
                    if last_persons and idx - last_persons_idx <= 4:
                        try:
                            missing = persons_without_faces(
                                last_persons, raw_dets)
                            if missing:
                                extra = detect_faces_in_person_crops(
                                    frame, missing, max_crops=10)
                                # 크롭 확대는 옷/머리카락/손을 얼굴로 오검출하기
                                # 쉽다. 확실한 검출만 받고, 이미 잡힌 얼굴과
                                # 겹치는 중복도 버린다. 큰 크롭 검출은 전체
                                # 프레임 YOLO 검출기가 받쳐줘야 한다.
                                extra = [
                                    (eb, es) for eb, es in extra
                                    if float(es) >= 0.85
                                    and not any(_overlap_ratio(eb, rb) > 0.35
                                                for rb, _rs in raw_dets)
                                    and (
                                        min(float(eb[2]) - float(eb[0]),
                                            float(eb[3]) - float(eb[1])) < 40
                                        or any(
                                            _overlap_ratio(eb, sb) > 0.30
                                            and float(ss) >= 0.55
                                            for sb, ss in yolo_raw))]
                                if extra:
                                    raw_dets = list(raw_dets) + list(extra)
                                    raw_dets.sort(
                                        key=lambda it: it[1], reverse=True)
                        except Exception:
                            log.exception("원거리 얼굴 크롭 보완 실패")
                    # 포즈 모델이 있으면 머리 키포인트가 없는 사람은
                    # 뒤통수/가려진 얼굴이므로 억지로 상단을 머리로 가정하지
                    # 않는다. (몸을 숙인 사람의 허리/발이 머리로 잡히는 것 방지)
                    # 포즈 모델이 없을 때만, 그것도 작은 원거리 사람에 한해
                    # 보수적으로 머리 위치를 보완한다.
                    allow_fallback = (
                        pose_model is None
                        and os.environ.get("MOSAIC_HEAD_FALLBACK", "1") != "0")
                    if (allow_fallback and last_persons
                            and idx - last_persons_idx <= 8):
                        active_heads = [t.box for t in head_tracker.active()]
                        h_f, w_f = frame.shape[:2]
                        for pb in last_persons:
                            if any(_faces_in_person(
                                    [FaceBox(b)], pb, margin=0.05)
                                   for b, _s in raw_dets):
                                continue
                            if any(_faces_in_person(
                                    [FaceBox(h)], pb, margin=0.05)
                                   for h in active_heads):
                                continue
                            bx = _small_head_fallback_box(pb, w_f, h_f)
                            if bx is None:
                                continue
                            if (bx[2] - bx[0]) / float(max(1, bx[3] - bx[1])) <= 2.0:
                                raw_dets.append((FaceBox(bx), 0.72))
                        raw_dets.sort(
                            key=lambda item: item[1], reverse=True)
                    yolo_force_boxes = []
                    yolo_force_scores = []
                    try:
                        for _b, _s in yolo_raw:
                            _wide_b = _in_wide_person(_b, last_persons)
                            if float(_s) < 0.55:
                                continue
                            _bw = max(1.0, float(_b[2]) - float(_b[0]))
                            _bh = max(1.0, float(_b[3]) - float(_b[1]))
                            _h, _w = frame.shape[:2]
                            _bcx = (float(_b[0]) + float(_b[2])) / 2.0
                            _bcy = (float(_b[1]) + float(_b[3])) / 2.0
                            _upper = False
                            for _p in last_persons or []:
                                try:
                                    _px1, _py1, _px2, _py2 = [
                                        float(v) for v in _p[:4]]
                                except (TypeError, ValueError, IndexError):
                                    continue
                                if (_px1 <= _bcx <= _px2
                                        and _py1 <= _bcy <= _py2
                                        and _bh <= 0.09 * max(
                                            1.0, _py2 - _py1)
                                        and (_bcy - _py1) <= 0.15 * max(
                                            1.0, _py2 - _py1)):
                                    _upper = True
                                break
                            if _upper:
                                _sx, _sy, _by = 0.30, 0.10, 1.80
                            else:
                                _sx, _sy, _by = 0.15, 0.05, 0.35
                            _exp = FaceBox([
                                max(0.0, float(_b[0]) - _sx * _bw),
                                max(0.0, float(_b[1]) - _sy * _bh),
                                min(float(_w), float(_b[2]) + _sx * _bw),
                                min(float(_h), float(_b[3]) + _by * _bh)],
                                angle=0.0)
                            # YOLO가 찾은 얼굴은 사람 박스 위치와 무관하게
                            # 강제로 그린다. 단, YuNet이 이미 정상 위치에서
                            # 잡은 얼굴은 중복 확장하지 않고, 점수가 낮은
                            # 팔/가슴 오검출도 제외한다.
                            if float(_s) >= 0.55:
                                _covered = False
                                for _rb, _rs in raw_dets:
                                    if (_overlap_ratio(_b, _rb) > 0.45
                                            and (not last_persons
                                                 or _face_upper_ok(
                                                     _rb, last_persons,
                                                     frame_h=frame.shape[0]))):
                                        _covered = True
                                        break
                                if not _covered:
                                    yolo_force_boxes.append(_exp)
                                    yolo_force_scores.append(float(_s))
                            if any(_overlap_ratio(_b, _rb) > 0.45
                                   for _rb, _rs in raw_dets):
                                continue
                            if last_persons and not _face_person_size_ok(
                                    _b, last_persons):
                                continue
                            if last_persons and not _face_upper_ok(
                                    _b, last_persons,
                                    frame_h=frame.shape[0]):
                                continue
                            raw_dets.append((_exp, _s))
                        raw_dets.sort(
                            key=lambda item: item[1], reverse=True)
                    except Exception:
                        log.exception("YOLO 얼굴 보완 실패")
                    tracker.update(raw_dets, flow_fn=flow_fn)
                    prev_gray_f1 = gray
                    active = tracker.active()
                    head_active = head_tracker.active()
                    prev_face_active = active
                    for t in active:
                        if t.misses == 0:
                            last_fresh[t.tid] = [idx, list(t.box)]
                    raw_face_frames.append(
                        {t.tid: list(t.box) for t in active if t.misses == 0})
                    # 짧게 놓친 얼굴은 유지하되, 그보다 오래된 트랙은 현재
                    # 포즈 머리와 겹칠 때만 그린다. 바닥 반사/허리/옷자락에
                    # 남은 고스트 마스크가 오래 깜빡이는 것을 막는다.
                    hold = max(8, int(round(fps * 6.0)))
                    fresh_heads = [t.box for t in head_active if t.misses == 0]
                    persons_now = person_frames.get(idx) or []
                    _fdict = {}
                    for t in active:
                        if not (t.misses <= hold
                                or any(_overlap_ratio(t.box, h) > 0.25
                                       for h in fresh_heads)):
                            continue
                        _ok = _face_upper_ok(
                            t.box, persons_now, max_frac=0.22,
                            frame_h=frame.shape[0],
                            heads=fresh_heads)
                        if not _ok:
                            _ok = _in_wide_person(t.box, persons_now)
                        if _ok:
                            _fdict[t.tid] = list(t.box)
                    # YOLO 강제 박스를 슬롯 ID로 유지한다. 검출기가 1~2프레임
                    # 놓쳐도 슬롯이 마지막 박스를 들고 있어 깜빡임(노출)이
                    # 생기지 않는다. 새로 들어온 박스는 위치 게이트를 통과해야
                    # 슬롯에 들어가고, 유지 중인 박스는 게이트 재검사 없이
                    # 홀드 기간만 유지한다(사람 박스 흔들림에 깜빡이지 않게).
                    _force_hold = max(12, int(round(fps * 0.66)))
                    _fresh_held = []
                    for _fb, _fs in zip(yolo_force_boxes, yolo_force_scores):
                        try:
                            _fss = float(_fs)
                        except (TypeError, ValueError):
                            continue
                        if not _force_face_ok(
                                _fb, persons_now, frame.shape[0],
                                score=_fss):
                            continue
                        _fresh_held.append((list(_fb), _fss))
                    _used_slots = set()
                    for _fb, _fs in _fresh_held:
                        try:
                            _cx = (float(_fb[0]) + float(_fb[2])) / 2.0
                            _cy = (float(_fb[1]) + float(_fb[3])) / 2.0
                            _sz = max(1.0, float(_fb[2]) - float(_fb[0]),
                                      float(_fb[3]) - float(_fb[1]))
                        except (TypeError, ValueError, IndexError):
                            continue
                        _best_id, _best_d = None, None
                        for _sid, _s in force_slots.items():
                            if _sid in _used_slots:
                                continue
                            if _s.get("misses", 0) > _force_hold:
                                continue
                            try:
                                _pb = _s["box"]
                                _px = (float(_pb[0]) + float(_pb[2])) / 2.0
                                _py = (float(_pb[1]) + float(_pb[3])) / 2.0
                                _ps = max(1.0, float(_pb[2]) - float(_pb[0]),
                                          float(_pb[3]) - float(_pb[1]))
                            except (TypeError, ValueError, IndexError):
                                continue
                            _d = (((_cx - _px) ** 2 + (_cy - _py) ** 2) ** 0.5
                                  / max(1.0, _sz, _ps))
                            if _d <= 0.8 or _overlap_ratio(_fb, _pb) > 0.15:
                                if _best_d is None or _d < _best_d:
                                    _best_d, _best_id = _d, _sid
                        if _best_id is None:
                            _best_id = force_next
                            force_next += 1
                        force_slots[_best_id] = {
                            "box": list(_fb), "score": float(_fs),
                            "misses": 0}
                        _used_slots.add(_best_id)
                    for _sid in list(force_slots):
                        if _sid not in _used_slots:
                            try:
                                force_slots[_sid]["misses"] = int(
                                    force_slots[_sid].get("misses", 0)) + 1
                            except (TypeError, ValueError):
                                force_slots[_sid]["misses"] = 1
                            if force_slots[_sid]["misses"] > _force_hold:
                                del force_slots[_sid]
                    for _sid, _s in force_slots.items():
                        try:
                            _fdict[10 ** 9 + int(_sid)] = list(_s["box"])
                        except (TypeError, ValueError, IndexError):
                            continue
                    face_frames.append(_fdict)
                    for t in active:
                        face_angles[idx][t.tid] = float(
                            getattr(t, "angle", 0.0))
                    head_frames.append({
                        t.tid: list(t.box) for t in head_active
                        if t.misses <= hold})
                    if progress_fn and total > 0 and idx % 5 == 0:
                        try:
                            progress_fn(min(0.5, 0.5 * idx / max(1, total)),
                                        idx, total, time.time() - t0)
                        except Exception:
                            pass
            finally:
                cap.release()
            if not face_frames:
                raise RuntimeError("처리된 프레임이 없습니다.")

            # ---------- 후처리: 역방향 채움 + 지연 없는 평활 ----------
            # 합집합 평활(win)은 박스 중심을 뒤처지게 하므로 쓰지 않는다.
            # 이동은 속도 예측·광학 흐름과 렌더 여백이 담당한다.
            face_frames = bridge_gaps(face_frames, cuts, max_gap=8,
                                      max_dist=0.6)
            face_s = backfill_and_smooth(face_frames, cuts, back=2, win=0)
            head_s = backfill_and_smooth(head_frames, cuts, back=1, win=0)
            # 검출 프레임 사이 1~2px 지터를 없애는 지연 없는 양방향 평활.
            # (0.2*prev + 0.6*cur + 0.2*next, 미래 프레임을 이미 알고 있음)
            face_s = temporal_smooth(face_s, cuts)
            head_s = temporal_smooth(head_s, cuts)
            # 같은 ID가 1~수 프레임 비었다가 돌아오면 선형 보간으로 메운다.
            # 검출기 단발 미스·게이트 흔들림이 그대로 노출(깜빡임)이 되지 않게
            # 한다. 멀리 움직인 경우는 잇지 않아 고스트를 만들지 않는다.
            face_s = fill_same_tid_gaps(face_s, cuts, max_gap=8)
            # 오래 못 보고 멀리 흘러간 박스는 2패스에서 mesh로 먼저 확인한다.
            # mesh가 얼굴을 찾으면 그 모양으로 그리고, 못 찾아도 마스크를
            # 빼지 않는다(노출보다 유지 우선). 갓 놓친 얼굴(6프레임 이내)은
            # mesh 확인 없이 그린다.
            n_frames = len(face_s)
            ghost_skip = [set() for _ in range(n_frames)]
            for i, boxes in enumerate(face_s):
                for tid, box in boxes.items():
                    lf = last_fresh.get(tid)
                    if lf is None:
                        continue
                    fidx, fbox = lf
                    if i - fidx <= 6:
                        continue
                    try:
                        fw = max(1.0, fbox[2] - fbox[0], fbox[3] - fbox[1])
                        dx = ((box[0] + box[2]) - (fbox[0] + fbox[2])) / 2.0
                        dy = ((box[1] + box[3]) - (fbox[1] + fbox[3])) / 2.0
                        if (dx * dx + dy * dy) ** 0.5 > 1.0 * fw:
                            ghost_skip[i].add(tid)
                    except (TypeError, ValueError, IndexError):
                        pass
            angle_frames = [face_angles.get(i, {})
                            for i in range(len(face_s))]
            angle_s = nearest_track_values(angle_frames, len(face_s), cuts)
            # 각도가 프레임마다 조금씩 튀면 얼굴형 마스크가 흔들린다.
            # 짧은 방향으로 부드럽게 이어 붙인다.
            _prev_ang = {}
            for _af in angle_s:
                for _tid, _a in list(_af.items()):
                    if _tid in _prev_ang:
                        _af[_tid] = _blend_angle(_prev_ang[_tid], _a, 0.25)
                    _prev_ang[_tid] = _af[_tid]
            face_gaps = find_gap_ranges(raw_face_frames, face_s, fps=fps)
            del face_frames, raw_face_frames, head_frames

            try:
                writer = FFmpegFrameWriter(
                    output_path, W, H, fps,
                    audio_src=input_path if self.mux_audio else None,
                    preset=preset)
            except Exception:
                writer = None
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
                if not out.isOpened():
                    cleanup_tmp(tmp_noaudio)
                    raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")
            debug_writer = self._open_debug_writer(W, H, fps)

            # ---------- 2패스: 그리기 ----------
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
            region_flow = RegionFlow()
            face_cache = FacePolyCache(mesh_refiner)
            head_cache = FacePolyCache(mesh_refiner)
            plate_dets: list = []
            plate_updated = -10 ** 9
            _detect_plates = None
            _filter_plates = None
            _detect_vehicles = None
            _plate_tracker = None
            plate_every = 2
            veh_model = None
            try:
                from .plates import _every as _plate_every
                from .plates import detect_plates as _detect_plates
                from .plates import (detect_vehicles as _detect_vehicles,
                                       filter_by_vehicles as _filter_plates,
                                       PlateTracker as _PlateTracker)
                plate_every = _plate_every()
                _plate_tracker = _PlateTracker(
                    max_miss=max(12, int(round(fps * 0.5))))
            except Exception:
                pass
            if _detect_plates is not None and "license_plate" in targets:
                try:
                    from .models import get_model as _get_model
                    veh_model = _get_model(self.model_name)
                    try:
                        _reset_tracker(veh_model)
                    except Exception:
                        pass
                except Exception:
                    veh_model = None
            prev_gray_plate = None
            try:
                while done < len(face_s):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if self.cancel_fn is not None and self.cancel_fn():
                        raise RenderCancelled()
                    gray_plate = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY)
                    if done in cuts:
                        region_flow.reset()
                        plate_dets = []
                        prev_gray_plate = None
                        if _plate_tracker is not None:
                            _plate_tracker.reset()
                    region_flow.begin(frame)
                    if _plate_tracker is not None \
                            and "license_plate" in targets:
                        dets = []
                        if (_detect_plates is not None
                                and done % max(1, plate_every) == 0):
                            try:
                                dets = _detect_plates(frame) or []
                                if _filter_plates is not None \
                                        and _detect_vehicles is not None \
                                        and veh_model is not None:
                                    try:
                                        vehs = _detect_vehicles(
                                            frame, veh_model)
                                    except Exception:
                                        vehs = []
                                    dets = _filter_plates(dets, vehs)
                            except Exception:
                                dets = []
                        try:
                            _plate_tracker.update(
                                dets, gray=gray_plate,
                                prev_gray=prev_gray_plate)
                            plate_dets = _plate_tracker.boxes()
                        except Exception:
                            log.exception("번호판 트랙 갱신 실패")
                        prev_gray_plate = gray_plate
                    elif (_detect_plates is not None
                            and "license_plate" in targets
                            and done % max(1, plate_every) == 0):
                        try:
                            cands = _detect_plates(frame) or []
                            if _filter_plates is not None \
                                    and _detect_vehicles is not None \
                                    and veh_model is not None:
                                try:
                                    vehs = _detect_vehicles(
                                        frame, veh_model)
                                except Exception:
                                    vehs = []
                                cands = _filter_plates(cands, vehs)
                            if cands:
                                plate_dets = cands
                                plate_updated = done
                        except Exception:
                            pass
                    if (_plate_tracker is None and plate_dets
                            and done - plate_updated > 30):
                        plate_dets = []
                    orig_frame = (frame.copy() if self.manual_regions
                                  else None)
                    faces_now = face_s[done]
                    max_active = max(max_active, len(faces_now))
                    debug_marks = []
                    drawn_faces_frame = []
                    ordered_faces = sorted(
                        faces_now.items(),
                        key=lambda kv: (
                            (kv[1][1] + kv[1][3]) / 2.0,
                            -((kv[1][2] - kv[1][0])
                              * (kv[1][3] - kv[1][1]))))
                    for tid, box in ordered_faces:
                        # 위쪽(얼굴) 박스를 먼저 그린다. 몸통/가슴으로
                        # 내려간 중복 트랙이 얼굴 마스크를 밀어내지 않게 한다.
                        # 겹침이 85%를 넘는 완전 중복만 건너뛴다.
                        if not (isinstance(tid, int)
                                and tid >= 10 ** 9):
                            if any(_overlap_ratio(box, d) > 0.85
                                   for d in drawn_faces_frame):
                                continue
                        ang = angle_s[done].get(tid, 0.0)
                        side = face_cache.stable_side(
                            tid, box[2] - box[0], box[3] - box[1])
                        bs = mosaic_block_size(side, side, blocks)
                        if tid in ghost_skip[done]:
                            # 오래 못 보고 멀리 흘러간 박스는 날것의 mesh로
                            # 먼저 검증한다. mesh가 얼굴을 찾으면 그 모양으로
                            # 그리고, 못 찾았다고 마스크를 빼지 않는다.
                            # mesh 실패=None은 '얼굴 아님'이 아니라 불확실이라
                            # 노출(깜빡임)보다 유지가 우선이다. 아래 일반 경로의
                            # 캐시 모양/박스 마스크로 그린다.
                            raw = None
                            if mesh_refiner is not None:
                                try:
                                    raw = mesh_refiner.polygon_for_box(
                                        frame, box)
                                except Exception:
                                    raw = None
                            if raw is not None:
                                if render_face_polygon(
                                        frame, raw, style=style, blocks=blocks,
                                        blur=blur, block_size=bs):
                                    rendered_tids.add(tid)
                                    track_frames[tid] += 1
                                    rendered_cats.add("face")
                                    drawn_faces_frame.append(box)
                                    debug_marks.append(
                                        (box, (0, 255, 0), f"F{tid}"))
                                    continue
                        poly, _fresh = face_cache.polygon(frame, tid, box)
                        if (poly is not None
                                and _poly_covers_box(
                                    poly, _padded_head_box(frame, box), 0.80)
                                and _poly_size_ok(poly, box)):
                            if render_face_polygon(
                                    frame, poly, style=style, blocks=blocks,
                                    blur=blur, block_size=bs):
                                rendered_tids.add(tid)
                                track_frames[tid] += 1
                                rendered_cats.add("face")
                                drawn_faces_frame.append(box)
                                debug_marks.append(
                                    (box, (0, 255, 0), f"F{tid}"))
                                continue
                        if _render_face_box(
                                frame, box, style, blocks, blur,
                                angle=ang, mesh=None, block_size=bs):
                            rendered_tids.add(tid)
                            track_frames[tid] += 1
                            rendered_cats.add("face")
                            drawn_faces_frame.append(box)
                        debug_marks.append((box, (0, 255, 0), f"F{tid}"))
                    if DRAW_HEAD_FALLBACK:
                        for tid, head in head_s[done].items():
                            # YuNet 얼굴이 이미 잡힌 사람에게 머리 보완 트랙이
                            # 겹치면 같은 얼굴을 두 번 가려 원/사각형이 겹친다.
                            if any(_overlap_ratio(head, fbox) > 0.15
                                   for fbox in drawn_faces_frame):
                                continue
                            hb = _clip_head(head, W, H)
                            if hb is None:
                                continue
                            hside = head_cache.stable_side(
                                tid, hb[2] - hb[0], hb[3] - hb[1])
                            hbs = mosaic_block_size(hside, hside, blocks)
                            hpoly, _fresh = head_cache.polygon(frame, tid, hb)
                            if (hpoly is not None
                                    and _poly_covers_box(
                                        hpoly, _padded_head_box(frame, hb), 0.80)
                                    and _poly_size_ok(hpoly, hb)
                                    and render_face_polygon(
                                    frame, hpoly, style=style, blocks=blocks,
                                    blur=blur, block_size=hbs)):
                                fallback_heads += 1
                                rendered_cats.add("face")
                            elif _render_face_box(
                                    frame, hb, style, blocks, blur, mesh=None,
                                    block_size=hbs):
                                fallback_heads += 1
                                rendered_cats.add("face")
                            debug_marks.append((head, (255, 0, 255), f"H{tid}"))

                    sidx = done // stride
                    persons_now = person_frames.get(done) or []
                    for cat in targets:
                        if cat == "face" or _cat_type(cat) != "region":
                            continue
                        boxes = _flow_region_boxes(
                            region_flow, cat, region_samples.get(cat),
                            sidx, done, stride)
                        if cat == "license_plate" and plate_dets:
                            try:
                                boxes = RegionFlow.merge(
                                    boxes,
                                    [dict(xyxy=list(b), conf=0.9,
                                          category="license_plate",
                                          region_type="region",
                                          source="local-plate")
                                     for b in plate_dets])
                            except Exception:
                                pass
                        for b in _dedupe_boxes(boxes):
                            x1, y1, x2, y2 = b["xyxy"]
                            if not plausible_box(cat, x1, y1, x2, y2, W, H):
                                continue
                            if not _region_area_ok(
                                    (x1, y1, x2, y2), W, H,
                                    region_max_area):
                                continue
                            # 옷/가방 위 텍스트가 주소·간판으로 잡히면
                            # 사람 몸통을 크게 덮는다. 사람 박스 안 영역은
                            # 배경 간판/문서로 보지 않는다.
                            if persons_now and _region_on_person(
                                    (x1, y1, x2, y2), persons_now):
                                continue
                            if not region_has_detail(frame, (x1, y1, x2, y2),
                                                     cat):
                                continue
                            bsz = (max(4, min(12, min(x2 - x1, y2 - y1) // 4))
                                   if cat == "license_plate" else None)
                            if render_region(frame, x1, y1, x2, y2,
                                             style=style, blocks=blocks,
                                             blur=blur,
                                             pad=region_pad_for(
                                                 cat, x2 - x1, y2 - y1),
                                             block_size=bsz):
                                rendered_region_cats.add(cat)
                                rendered_cats.add(cat)

                    region_flow.end()
                    self._apply_manual(frame, orig_frame, done, fps)
                    if debug_writer is not None:
                        dbg = frame.copy()
                        self._draw_debug_marks(dbg, debug_marks)
                        debug_writer.write(dbg)
                    elif DEBUG_OVERLAY:
                        self._draw_debug_marks(frame, debug_marks)
                    if writer is not None:
                        writer.write(frame)
                    else:
                        out.write(frame)
                    done += 1
                    if progress_fn and total > 0:
                        try:
                            progress_fn(
                                0.5 + 0.5 * min(1.0, done / max(1, total)),
                                done, total, time.time() - t0)
                        except Exception:
                            pass
            finally:
                cap.release()

            if done == 0:
                raise RuntimeError("처리된 프레임이 없습니다.")
        except Exception:
            if debug_writer is not None:
                debug_writer.abort()
            if writer is not None:
                writer.abort()
            else:
                cleanup_tmp(tmp_noaudio)
            raise

        if debug_writer is not None:
            try:
                debug_writer.close()
            except Exception:
                log.exception("디버그 영상 마무리 실패")
        if writer is not None:
            writer.close()
        else:
            out.release()
            try:
                if self.mux_audio:
                    mux_audio_h264(tmp_noaudio, input_path, output_path,
                                   preset=preset)
                else:
                    cleanup_tmp(output_path)
                    shutil.copy(tmp_noaudio, output_path)
            except Exception:
                cleanup_tmp(output_path)
                shutil.copy(tmp_noaudio, output_path)
            finally:
                cleanup_tmp(tmp_noaudio)

        stats = {}
        for tid in sorted(rendered_tids):
            stats[tid] = {
                "mosaic": True,
                "direct": True,
                "frames": int(track_frames.get(tid, 0)),
                "selector": "local-face",
                "targets": targets,
                "hit_targets": ["face"],
                "cat_on": {"face": True},
                "frac": 1.0,
            }
        stats["_regions"] = len(rendered_region_cats)
        stats["_cats"] = sorted(rendered_cats)
        stats["_untracked"] = []
        stats["_faces"] = int(max_active)
        stats["_head_fallbacks"] = int(fallback_heads)
        stats["_face_gaps"] = face_gaps
        stats["_vlm_calls"] = 0
        stats["_vlm_errors"] = 0
        return stats

    # ------------------------------------------------------------------
    # run() 분할: 동작은 분할 전과 같다(순서/부작용 포함). 단계별 상태는
    # _Pass1State / _Plan / _RenderState / _FrameCtx 에 담는다.
    # ------------------------------------------------------------------
    def run(self, progress_fn=None):
        if self._face_only_mode():
            return self._run_face_only(progress_fn)

        p = self._start_run()
        self._load_cache(p)
        if not p.cache_hit:
            p.observed_types = {}
        p.pending_person_priors = defaultdict(lambda: defaultdict(list))
        if not p.cache_hit:
            self._probe_input(p)
        if not p.cache_hit:
            self._seed_priors(p)
        self._pass1(p, progress_fn)
        self._save_cache(p)
        _reset_tracker(p.model)
        plan = self._make_plan(p)

        # ---- 2패스: 선택된 카테고리만 렌더 ----
        writer, out, tmp_noaudio = self._open_writer(p)
        debug_writer = self._open_debug_writer(p.W, p.H, p.fps)
        rs = _RenderState()
        # 얼굴만 모드에서는 사람 트랙이 끊긴 뒤 오래 남기면 마지막 위치에
        # 고스트 모자이크가 생긴다. 짧게만 유지한다.
        rs.MAX_MISS = 4 if plan.blanket_cats else 8
        rs.progress_base = 0.0 if p.fast_person_only else 0.5
        rs.region_flow = RegionFlow()
        rs.face_trk = FaceTracker(
            max_miss=max(24, int(round(p.fps * 3.0))),
            seed_miss=max(8, int(round(p.fps * 1.0))))
        try:
            rs.face_mesh = FaceMeshRefiner.shared()
            if not rs.face_mesh.available():
                rs.face_mesh = None
        except Exception:
            rs.face_mesh = None
        try:
            rs.face_poly_cache = FacePolyCache(rs.face_mesh)
        except Exception:
            rs.face_poly_cache = None
        try:
            rs.head_poly_cache = FacePolyCache(rs.face_mesh)
        except Exception:
            rs.head_poly_cache = None
        try:
            from .plates import PlateTracker
            rs.plate_tracker = PlateTracker(
                max_miss=max(12, int(round(p.fps * 0.5))))
        except Exception:
            rs.plate_tracker = None
        try:
            frames = self._frame_source(p, plan, rs, progress_fn)
        except Exception:
            if rs.cap2 is not None:
                rs.cap2.release()
            if debug_writer is not None:
                debug_writer.abort()
            if writer is not None:
                writer.abort()
            else:
                cleanup_tmp(tmp_noaudio)
            raise
        self._render_pass(p, plan, rs, frames, writer, out, tmp_noaudio,
                          progress_fn, debug_writer)
        if debug_writer is not None:
            try:
                debug_writer.close()
            except Exception:
                log.exception("디버그 영상 마무리 실패")
        self._close_writer(p, writer, out, tmp_noaudio)
        return self._build_stats(p, plan, rs)

    # ---- 시작: 설정, 모델, 출력 폴더 ----
    def _start_run(self):
        p = _Pass1State()
        targets = self.targets
        invert = self.invert
        p.conf = self.conf
        p.imgsz = self.imgsz
        p.keyframe_stride = self.keyframe_stride
        person_priors = self.person_priors
        if self.ultra:
            p.imgsz = max(p.imgsz, 960)
            p.conf = min(p.conf, 0.18)

        p.fast_person_only = bool(
            self.mask_all_persons and not invert and targets
            and all(c in CATEGORY_TYPE
                    and CATEGORY_TYPE.get(c) == "person"
                    and c != "name_tag" for c in targets))

        p.model = None
        p.needs_person_pass1 = (
            invert or bool(person_priors)
            or any(c not in CATEGORY_TYPE
                   or CATEGORY_TYPE.get(c) == "person" for c in targets))
        if p.needs_person_pass1:
            p.model = get_thread_model(self.model_name)
            _reset_tracker(p.model)

        out_dir = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(out_dir, exist_ok=True)
        return p

    # ---- 1패스 결과 캐시 로드 (JSON 우선, pickle 하위 호환) ----
    def _load_cache(self, p):
        p.cache_hit = False
        if (not p.fast_person_only and self.use_cache and self.cache_path
                and os.path.exists(self.cache_path)):
            try:
                with open(self.cache_path, "rb") as f:
                    raw_bytes = f.read()
                if raw_bytes[:1] == b"{":
                    cache = _deserialize_cache_json(
                        json.loads(raw_bytes.decode("utf-8")))
                else:
                    cache = pickle.loads(raw_bytes)
                if cache.get("version") != 2:
                    raise ValueError("cache version mismatch")
                p.votes = cache["votes"]
                p.person_box_samples = cache["person_box_samples"]
                p.region_samples = cache["region_samples"]
                p.person_cats_seen = cache.get("person_cats_seen", set())
                p.all_detail = cache.get("all_detail", {})
                p.observed_types = cache.get("observed_types", {})
                p.total = cache["meta"]["total"]
                p.fps = cache["meta"]["fps"]
                p.H = cache["meta"]["H"]
                p.W = cache["meta"]["W"]
                p.keyframe_stride = cache["meta"]["keyframe_stride"]
                p.max_sidx = (max(0, (p.total - 1) // p.keyframe_stride)
                              if p.total > 0 else 0)
                p.cache_hit = True
            except Exception:
                p.cache_hit = False
                p.votes = None
                p.person_box_samples = None
                p.region_samples = None

    # ---- 캐시 미스: 입력 영상 정보와 VLM 선택기 준비 ----
    def _probe_input(self, p):
        p.votes = defaultdict(lambda: defaultdict(list))
        p.person_box_samples = defaultdict(lambda: defaultdict(list))
        p.region_samples = defaultdict(lambda: defaultdict(list))

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {self.input_path}")
        p.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(p.fps) or p.fps <= 0.0:
            p.fps = 30.0
        p.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, first = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
        p.H, p.W = first.shape[:2]
        del first

        p.selector = None
        if not p.fast_person_only:
            p.selector = VLMSelector(query_raw=self.query_raw,
                                     targets=self.targets,
                                     api_key=self.api_key,
                                     vlm_model=self.vlm_model,
                                     include_all=False,
                                     vlm_base_url=self.vlm_base_url)

        p.region_samples = defaultdict(lambda: defaultdict(list))
        p.person_box_samples = defaultdict(lambda: defaultdict(list))
        p.pending_person_priors = defaultdict(lambda: defaultdict(list))
        p.max_sidx = (max(0, (p.total - 1) // p.keyframe_stride)
                      if p.total > 0 else 0)

    # ---- 체크리스트 priors → region_samples / pending_person_priors ----
    def _seed_priors(self, p):
        targets = self.targets
        keyframe_stride = p.keyframe_stride
        max_sidx = p.max_sidx
        for cat, by_frame in (self.region_priors or {}).items():
            if cat not in targets or _cat_type(cat) != "region":
                continue
            for fidx, boxes in (by_frame or {}).items():
                try:
                    fidx = int(fidx)
                except (TypeError, ValueError):
                    continue
                sidx = min(max_sidx, max(0, fidx // keyframe_stride))
                for coords in _prior_coords(boxes):
                    p.region_samples[cat][sidx].append({
                        "xyxy": coords, "conf": 1.0, "category": cat,
                        "region_type": "region", "source": "vlm",
                        "matches_prompt": True, "due": fidx,
                    })

        for cat, by_frame in (self.person_priors or {}).items():
            if cat not in targets or _cat_type(cat) != "person":
                continue
            p.observed_types.setdefault(cat, "person")
            for fidx, boxes in (by_frame or {}).items():
                try:
                    fidx = int(fidx)
                except (TypeError, ValueError):
                    continue
                sidx = min(max_sidx, max(0, fidx // keyframe_stride))
                p.pending_person_priors[cat][sidx].extend(
                    (coords, fidx) for coords in _prior_coords(boxes))

    # ---- 1패스: 모든 프레임 추적 + 키프레임 VLM 탐지 ----
    def _pass1(self, p, progress_fn):
        targets = self.targets
        keyframe_stride = p.keyframe_stride
        total = p.total
        if p.cache_hit:
            _reset_tracker(p.model)
        else:
            p.votes = defaultdict(lambda: defaultdict(list))
            p.person_cats_seen = {c for c in targets
                                  if CATEGORY_TYPE.get(c) == "person"}
            p.all_detail = defaultdict(dict)
        p.t0 = time.time()

        p.vlm_budget = VLMBudget(per_video_limit=self.vlm_call_limit)
        p.vlm_targets = None
        if self.vlm_call_limit is not None:
            if self.vlm_call_limit <= 0:
                p.vlm_targets = set()
            else:
                n_keyframes = max(
                    1, (max(1, total) + keyframe_stride - 1)
                    // keyframe_stride)
                k = min(int(self.vlm_call_limit), n_keyframes)
                if k <= 1:
                    p.vlm_targets = {0}
                else:
                    p.vlm_targets = {
                        int(round(i * (n_keyframes - 1) / float(k - 1)))
                        for i in range(k)
                    }
        p.vlm_errors = 0
        p.vlm_disabled = False

        run_pass1_stream = (not p.fast_person_only
                            and (p.needs_person_pass1
                                 or p.vlm_targets is None
                                 or bool(p.vlm_targets)))

        p.replay_on = os.environ.get("MOSAIC_REPLAY", "1") != "0"
        p.recorder = _DetRecorder() if p.replay_on else None

        if not p.cache_hit and run_pass1_stream:
            self._pass1_stream(p, progress_fn)

    def _pass1_stream(self, p, progress_fn):
        input_path = self.input_path
        model = p.model
        recorder = p.recorder
        needs_person_pass1 = p.needs_person_pass1
        keyframe_stride = p.keyframe_stride
        total = p.total
        t0 = p.t0
        person_box_samples = p.person_box_samples

        cap1 = None
        if needs_person_pass1:
            _reset_tracker(model)
            results1 = iter(model.track(
                source=input_path, stream=True, persist=True,
                tracker=_TRACKER_CFG, conf=p.conf, classes=[0],
                verbose=False, imgsz=p.imgsz, iou=0.5, max_det=30))

            def _iter1():
                idx = -1
                while True:
                    # 스레드마다 모델은 분리했지만 GPU 추론은 한 번에 하나만
                    # 실행해 VRAM/커널 경합을 막는다.
                    with MODEL_LOCK:
                        try:
                            res = next(results1)
                        except StopIteration:
                            break
                    idx += 1
                    yield idx, res
        else:
            cap1 = cv2.VideoCapture(input_path)

            def _iter1():
                idx = -1
                while True:
                    ok1, frm = cap1.read()
                    if not ok1:
                        break
                    idx += 1
                    yield idx, SimpleNamespace(
                        orig_img=frm, boxes=None, masks=None)

        try:
            chunk = []
            for frame_idx, r in _iter1():
                if self.cancel_fn is not None and self.cancel_fn():
                    raise RenderCancelled()
                if recorder is not None and needs_person_pass1:
                    recorder.add(r)
                if progress_fn and total > 0 and frame_idx % 5 == 0:
                    try:
                        progress_fn(
                            0.5 * min(1.0, (frame_idx + 1) / total),
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
                           if r.boxes.id is not None
                           else np.full((len(boxes),), -1, dtype=int))

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

                chunk.append({
                    "frame_idx": frame_idx,
                    "sample_idx": sample_idx,
                    "cand": cand,
                    "frame": frame.copy(),
                })
                if len(chunk) >= 6:
                    self._process_chunk(p, chunk)
                    chunk = []
            self._process_chunk(p, chunk)
        finally:
            if cap1 is not None:
                cap1.release()

        for cat, by_sidx in p.pending_person_priors.items():
            for _sidx, entries in by_sidx.items():
                for coords, due in entries:
                    person_box_samples[cat][_sidx].append(
                        {"tid": -1, "rel": None, "abs": coords,
                         "due": int(due)})

    def _process_chunk(self, p, chunk):
        """키프레임 묶음의 VLM을 병렬 호출하고 순서대로 집계한다."""
        targets = self.targets
        selector = p.selector
        vlm_targets = p.vlm_targets
        vlm_budget = p.vlm_budget
        votes = p.votes
        person_box_samples = p.person_box_samples
        region_samples = p.region_samples
        all_detail = p.all_detail
        observed_types = p.observed_types
        person_cats_seen = p.person_cats_seen
        pending_person_priors = p.pending_person_priors
        if not chunk:
            return
        vlm_results = {}
        if not p.vlm_disabled and (vlm_targets is None or vlm_targets):
            run_now = []
            for rec in chunk:
                if (vlm_targets is not None
                        and rec["sample_idx"] not in vlm_targets):
                    continue
                if not vlm_budget.can_call():
                    continue
                vlm_budget.record(1)
                run_now.append(rec)
            if run_now:
                with ThreadPoolExecutor(
                        max_workers=min(4, len(run_now))) as ex:
                    futs = {id(rec): ex.submit(selector.detect_boxes,
                                               rec["frame"])
                            for rec in run_now}
                    for rec in run_now:
                        try:
                            vlm_results[id(rec)] = futs[id(rec)].result()
                        except Exception as e:
                            p.vlm_errors += 1
                            vlm_results[id(rec)] = []
                            if classify_vlm_error(e) in (
                                    "quota_exceeded", "auth_error",
                                    "model_not_found"):
                                p.vlm_disabled = True

        for rec in chunk:
            cand = rec["cand"]
            sample_idx = rec["sample_idx"]
            vlm_boxes = vlm_results.get(id(rec), [])

            if pending_person_priors:
                live_person = defaultdict(list)
                for b in vlm_boxes:
                    bcat = canon_category(b.get("category")
                                          or b.get("phrase"))
                    if (bcat in targets
                            and _cat_type(bcat, b.get("region_type"))
                            == "person"):
                        live_person[bcat].append(b)
                for cat, by_sidx in list(pending_person_priors.items()):
                    if sample_idx not in by_sidx:
                        continue
                    for coords, due in by_sidx.pop(sample_idx):
                        if any(iou(coords, b.get("xyxy", [0, 0, 0, 0])) >= 0.25
                               for b in live_person.get(cat, [])):
                            continue
                        _store_person_box(
                            person_box_samples, cat, sample_idx,
                            {"xyxy": coords}, cand, due=due)

            judged = selector.match_items(cand, vlm_boxes)
            hits_by_key = {}
            detail_by_key = {}
            for it in cand:
                _hit, _detail = judged.get(
                    it["key"], (False, {"selector": "vlm", "hits": []}))
                hits_by_key[it["key"]] = set(_detail.get("hits", []))
                detail_by_key[it["key"]] = _detail

            for it in cand:
                tid = it["tid"]
                hits = {c for c in hits_by_key.get(it["key"], set())
                        if c in targets}
                person_cats_seen.update(hits)
                for cat in hits:
                    votes[tid][cat].append(True)
                    all_detail[tid].setdefault(
                        cat, detail_by_key.get(it["key"], {}))

            for b in vlm_boxes:
                cat = canon_category(b.get("category") or b.get("phrase"))
                if cat not in targets:
                    continue
                rtype = _cat_type(cat, b.get("region_type"))
                observed_types.setdefault(cat, rtype)
                if rtype == "region":
                    region_samples[cat][sample_idx].append(b)
                else:
                    _store_person_box(person_box_samples, cat, sample_idx,
                                      b, cand)
            rec["frame"] = None

    # ---- 1패스 결과 캐시 저장 (JSON 형식으로 안전하게 기록) ----
    def _save_cache(self, p):
        if (not p.fast_person_only and self.use_cache and self.cache_path
                and not p.cache_hit):
            try:
                cache_dir = os.path.dirname(self.cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                cache = {
                    "version": 2,
                    "meta": {
                        "total": p.total,
                        "fps": p.fps,
                        "H": p.H,
                        "W": p.W,
                        "keyframe_stride": p.keyframe_stride,
                    },
                    "votes": dict(p.votes),
                    "person_box_samples": dict(p.person_box_samples),
                    "region_samples": dict(p.region_samples),
                    "person_cats_seen": set(p.person_cats_seen),
                    "all_detail": dict(p.all_detail),
                    "observed_types": dict(p.observed_types),
                }
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(_serialize_cache_json(cache), f,
                              ensure_ascii=False)
            except Exception:
                pass

    # ---- 전역 확정 ----
    def _make_plan(self, p):
        targets = self.targets
        invert = self.invert
        person_priors = self.person_priors
        votes = p.votes
        all_detail = p.all_detail
        observed_types = p.observed_types
        person_box_samples = p.person_box_samples
        region_samples = p.region_samples
        plan = _Plan()
        prior_person_cats = {c for c in person_priors if c in targets}
        for _cat in prior_person_cats:
            observed_types.setdefault(_cat, "person")
        person_cats = {c for c in targets
                       if c in p.person_cats_seen or c in prior_person_cats
                       or _cat_type(c, observed_types.get(c)) == "person"}
        region_cats = {c for c in targets if c not in person_cats}
        region_cats |= {c for c in region_samples
                        if c in targets and c not in person_cats}
        direct_cats = {c for c in person_cats if person_box_samples.get(c)}
        # 얼굴은 전용 얼굴 트래커가 그린다. 사람 박스 상단 30%를 얼굴로
        # 칠하면 몸을 숙인 사람/부분 검출에서 배·가슴이 가려지므로 뺀다.
        mask_cats = {c for c in person_cats
                     if c not in ("name_tag", "face")}

        cat_on = {}
        for tid, by_cat in votes.items():
            cat_on[tid] = {}
            for cat, lst in by_cat.items():
                if not lst:
                    continue
                frac = sum(1 for v in lst if v) / len(lst)
                cat_on[tid][cat] = frac >= 0.2
        for cat in direct_cats:
            for by_sidx in person_box_samples.get(cat, {}).values():
                for entry in by_sidx:
                    tid = int(entry.get("tid", -1))
                    if tid >= 0:
                        cat_on.setdefault(tid, {})[cat] = True

        global_on = {}
        mask_on = {}
        global_hit_targets = {}
        global_detail = {}
        for tid in sorted(set(votes) | set(cat_on)):
            by_cat = votes.get(tid, {})
            on_cats = [c for c in targets
                       if cat_on.get(tid, {}).get(c) and c in person_cats]
            global_on[tid] = bool(on_cats)
            mask_on[tid] = any(c in mask_cats for c in on_cats)
            global_hit_targets[tid] = on_cats
            fracs = [sum(1 for v in lst if v) / len(lst)
                     for lst in by_cat.values() if lst]
            frac_any = max(fracs) if fracs else (1.0 if on_cats else 0.0)
            samp = {}
            for c in on_cats or list(cat_on.get(tid, {}).keys()):
                if tid in all_detail and c in all_detail[tid]:
                    samp = all_detail[tid][c]
                    break
            global_detail[tid] = {"frac": round(float(frac_any), 3),
                                  "cat_on": {k: v for k, v in
                                             cat_on.get(tid, {}).items() if v},
                                  "sample": samp}

        # 얼굴은 전용 얼굴 트래커가 모든 사람 트랙에 적용한다. 사람 박스
        # 상단을 일괄로 가리는 blanket은 배/가슴/몸 전체 오마스크를 만들어
        # 사용하지 않는다.
        blanket_cats = set()

        plan.person_cats = person_cats
        plan.region_cats = region_cats
        plan.direct_cats = direct_cats
        plan.mask_cats = mask_cats
        plan.cat_on = cat_on
        plan.global_on = global_on
        plan.mask_on = mask_on
        plan.global_hit_targets = global_hit_targets
        plan.global_detail = global_detail
        plan.blanket_cats = blanket_cats
        plan.needs_person = bool(person_cats) or invert
        plan.face_only_person = bool(self.area_mode == "face_only"
                                     and not invert and plan.needs_person)
        return plan

    # ---- writer 열기 / 닫기 ----
    def _open_writer(self, p):
        tmp_noaudio = self.output_path + ".noaudio.mp4"
        writer = None
        out = None
        try:
            writer = FFmpegFrameWriter(
                self.output_path, p.W, p.H, p.fps,
                audio_src=self.input_path if self.mux_audio else None,
                preset=self.encode_preset)
        except Exception:
            writer = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_noaudio, fourcc, p.fps, (p.W, p.H))
            if not out.isOpened():
                cleanup_tmp(tmp_noaudio)
                raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")
        return writer, out, tmp_noaudio

    def _close_writer(self, p, writer, out, tmp_noaudio):
        if writer is not None:
            writer.close()
        else:
            out.release()
            try:
                if self.mux_audio:
                    mux_audio_h264(tmp_noaudio, self.input_path,
                                   self.output_path,
                                   preset=self.encode_preset)
                else:
                    cleanup_tmp(self.output_path)
                    shutil.copy(tmp_noaudio, self.output_path)
            except Exception:
                cleanup_tmp(self.output_path)
                shutil.copy(tmp_noaudio, self.output_path)
            finally:
                cleanup_tmp(tmp_noaudio)

    # ---- 2패스 프레임 소스 (기록 재생 / 재추적 / 추적 없음) ----
    def _frame_source(self, p, plan, rs, progress_fn):
        input_path = self.input_path
        model = p.model
        recorder = p.recorder
        total = p.total
        t0 = p.t0
        needs_person = plan.needs_person
        replay_on = p.replay_on
        rs.cap2 = None
        have_rec = bool(recorder is not None and recorder.frames
                        and p.needs_person_pass1 and not p.cache_hit)
        if needs_person and replay_on and not have_rec:
            recorder = _DetRecorder()
            p.recorder = recorder
            _reset_tracker(model)
            rs.progress_base = 0.5
            rec_stream = model.track(
                source=input_path, stream=True, persist=True,
                tracker=_TRACKER_CFG, conf=p.conf, classes=[0],
                verbose=False, imgsz=p.imgsz, iou=0.5, max_det=30)
            for _ri, _res in enumerate(rec_stream):
                if self.cancel_fn is not None and self.cancel_fn():
                    raise RenderCancelled()
                recorder.add(_res)
                if progress_fn and total > 0 and _ri % 5 == 0:
                    try:
                        progress_fn(0.5 * min(1.0, (_ri + 1) / total),
                                    _ri + 1, total, time.time() - t0)
                    except Exception:
                        pass
            have_rec = bool(recorder.frames)
        if needs_person and replay_on and have_rec:
            try:
                n_added = recorder.finalize()
                log.info("검출 후처리: 합성 박스 %d개 추가 (프레임 %d)",
                         n_added, len(recorder.frames))
            except Exception:
                log.exception("검출 후처리 실패")

            def _iter_frames():
                for res in recorder.replay(input_path):
                    yield res
        elif needs_person:
            _reset_tracker(model)
            results2 = model.track(source=input_path, stream=True,
                                   persist=True, tracker=_TRACKER_CFG,
                                   conf=p.conf, classes=[0], verbose=False,
                                   imgsz=p.imgsz, iou=0.5, max_det=30)

            def _iter_frames():
                for res in results2:
                    yield res
        else:
            rs.cap2 = cv2.VideoCapture(input_path)
            cap2 = rs.cap2

            def _iter_frames():
                while True:
                    ok2, frm = cap2.read()
                    if not ok2:
                        break
                    yield SimpleNamespace(orig_img=frm, boxes=None,
                                          masks=None)
        return _iter_frames()

    # ---- 2패스 프레임 루프 ----
    def _render_pass(self, p, plan, rs, frames, writer, out, tmp_noaudio,
                     progress_fn, debug_writer=None):
        try:
            for r in frames:
                frame = self._render_frame(p, plan, rs, r)
                if debug_writer is not None:
                    dbg = frame.copy()
                    self._draw_debug_marks(dbg, rs.debug_boxes)
                    debug_writer.write(dbg)
                elif DEBUG_OVERLAY:
                    self._draw_debug_marks(frame, rs.debug_boxes)
                if writer is not None:
                    writer.write(frame)
                else:
                    out.write(frame)
                rs.done += 1
                if progress_fn and p.total > 0:
                    try:
                        progress_fn(
                            rs.progress_base + (1.0 - rs.progress_base)
                            * min(1.0, rs.done / max(1, p.total)),
                            rs.done, p.total, time.time() - p.t0)
                    except Exception:
                        pass
            if rs.done == 0:
                raise RuntimeError("처리된 프레임이 없습니다.")
        except Exception:
            if debug_writer is not None:
                debug_writer.abort()
            if writer is not None:
                writer.abort()
            else:
                cleanup_tmp(tmp_noaudio)
            raise
        finally:
            if rs.cap2 is not None:
                rs.cap2.release()

    def _render_frame(self, p, plan, rs, r):
        """프레임 1장을 그려 반환한다. 합성 프레임/스텁 검출로 테스트 가능."""
        frame = r.orig_img.copy()
        rs.debug_boxes = []
        rs.region_flow.begin(r.orig_img)
        fc = _FrameCtx(frame=frame, r=r, sidx=rs.done // p.keyframe_stride)

        if plan.face_only_person:
            self._update_faces(p, plan, rs, fc)
        self._collect_items(p, plan, rs, fc)
        self._compute_keep_union(fc)
        self._render_items(p, plan, rs, fc)
        self._render_ghosts(p, plan, rs, fc)
        self._render_untracked_direct(p, plan, rs, fc)
        self._render_regions(p, plan, rs, fc)

        rs.region_flow.end()
        rs.debug_boxes = list(fc.debug_boxes)
        self._apply_manual(frame, r.orig_img, rs.done, p.fps)
        return frame

    def _draw_face(self, p, rs, fc, fbox):
        for drawn in fc.drawn_faces:
            if _overlap_ratio(fbox, drawn) > 0.65:
                return True
        ang = getattr(fbox, "angle", 0.0)
        cache = getattr(rs, "face_poly_cache", None)
        tid = getattr(fbox, "tid", None)
        bs = None
        if cache is not None and tid is not None:
            try:
                side = cache.stable_side(
                    tid, fbox[2] - fbox[0], fbox[3] - fbox[1])
                bs = mosaic_block_size(side, side, self.blocks)
            except Exception:
                bs = None
            poly, _fresh = cache.polygon(fc.frame, tid, fbox)
            if (poly is not None
                    and _poly_covers_box(
                        poly, _padded_head_box(fc.frame, fbox), 0.80)
                    and _poly_size_ok(poly, fbox)
                    and render_face_polygon(
                    fc.frame, poly, style=self.style, blocks=self.blocks,
                    blur=self.blur, block_size=bs)):
                fc.drawn_faces.add(tuple(int(v) for v in fbox))
                fc.debug_boxes.append(
                    (fbox, (0, 255, 0), f"F{int(fbox[0])}"))
                rs.rendered_cats.add("face")
                return True
        mesh = getattr(rs, "face_mesh", None)
        if _render_face_box(fc.frame, fbox, self.style, self.blocks,
                            self.blur, angle=ang, mesh=mesh,
                            block_size=bs):
            fc.drawn_faces.add(tuple(int(v) for v in fbox))
            fc.debug_boxes.append((fbox, (0, 255, 0), f"F{int(fbox[0])}"))
            rs.rendered_cats.add("face")
            return True
        return False

    def _update_faces(self, p, plan, rs, fc):
        r = fc.r
        face_trk = rs.face_trk
        try:
            if is_scene_cut(rs.prev_sig_run, r.orig_img):
                face_trk.reset()
            rs.prev_sig_run = scene_signature(r.orig_img)
            _g = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2GRAY)
            _ff = ((lambda b, _p=rs.prev_gray_run, _c=_g:
                    shift_box(_p, _c, b))
                   if rs.prev_gray_run is not None else None)
            _dets = detect_faces(r.orig_img)
            if _face_rotate_enabled() and (
                    rs.done % _face_rotate_every() == 0 or not _dets):
                _dets = detect_faces_rotated(r.orig_img, base=_dets)
            try:
                if r.boxes is not None and len(r.boxes) > 0:
                    _persons = r.boxes.xyxy.cpu().numpy()
                    _h, _w = r.orig_img.shape[:2]
                    _dets = filter_faces_by_person(
                        _dets, _persons, frame_area=float(_w * _h))
                    # 얼굴이 없는 사람(원거리 군중)은 사람 상단 크롭 확대.
                    missing = persons_without_faces(_persons, _dets)
                    if missing:
                        extra = detect_faces_in_person_crops(
                            r.orig_img, missing, max_crops=10)
                        if extra:
                            _dets = list(_dets) + list(extra)
                            _dets.sort(key=lambda it: it[1], reverse=True)
            except Exception:
                log.exception("얼굴-사람 게이트 실패")
            # YuNet이 놓친 마스크/누운/측면 얼굴을 YOLO가 보완한다.
            # 얼굴+번호판 같은 일반 경로도 깜빡임 없이 가리게 한다.
            try:
                _yolo_dets = detect_faces_yolo(r.orig_img)
                if _yolo_dets:
                    _merged = list(_dets)
                    for _yb, _ys in _yolo_dets:
                        try:
                            if float(_ys) < 0.55:
                                continue
                        except (TypeError, ValueError):
                            continue
                        if any(_overlap_ratio(_yb, _b) > 0.45
                               for _b, _s in _merged):
                            continue
                        _merged.append((_yb, _ys))
                    _merged.sort(key=lambda it: it[1], reverse=True)
                    _dets = _merged
            except Exception:
                log.exception("YOLO 얼굴 보완 실패(사람 경로)")
            face_trk.update(_dets, flow_fn=_ff)
            rs.prev_gray_run = _g
            _fbs = []
            for t in face_trk.active():
                _fb = FaceBox(t.box, angle=getattr(t, "angle", 0.0))
                _fb.tid = t.tid
                _fbs.append(_fb)
            fc.faces_now = sorted(
                _fbs,
                key=lambda b: ((b[2] - b[0]) * (b[3] - b[1])),
                reverse=True)
        except Exception:
            log.exception("얼굴 검출 실패(사람 경로)")
            fc.faces_now = []
        if "face" in plan.blanket_cats:
            for _fb in fc.faces_now:
                self._draw_face(p, rs, fc, _fb)

    def _collect_items(self, p, plan, rs, fc):
        """검출된 사람 → 평활 박스, 마스크, 렌더 대상 items."""
        r = fc.r
        frame = fc.frame
        invert = self.invert
        area_mode = self.area_mode
        ALPHA = rs.ALPHA
        smooth, hist_mask, hist_miss = rs.smooth, rs.hist_mask, rs.hist_miss
        track_frames = rs.track_frames
        global_on, mask_on = plan.global_on, plan.mask_on
        blanket_cats = plan.blanket_cats
        seen = fc.seen
        items = fc.items
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None
                   else np.full((len(boxes),), -1, dtype=int))
            masks_full = None
            if r.masks is not None:
                try:
                    masks_full = list(r.masks.xy)
                except Exception:
                    log.exception("masks.xy 읽기 실패")
                    masks_full = None
                if masks_full is not None \
                        and len(masks_full) != len(boxes):
                    log.warning("mask/box 개수 불일치: %d/%d",
                                len(masks_full), len(boxes))
                    masks_full = None
            keep = filter_contained(boxes)
            for _idx, _keep in enumerate(keep):
                if _keep:
                    continue
                _tid = int(ids[_idx])
                if _tid >= 0:
                    smooth.pop(_tid, None)
                    hist_mask.pop(_tid, None)
                    hist_miss.pop(_tid, None)
            boxes = boxes[keep]
            ids = ids[keep]
            if masks_full is not None:
                masks_full = [m for m, k in zip(masks_full, keep)
                              if k]
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                tid = int(ids[i])
                seen.add(tid)
                track_frames[tid] += 1
                if tid in smooth:
                    px1, py1, px2, py2 = smooth[tid]
                    x1 = int(ALPHA * x1 + (1 - ALPHA) * px1)
                    y1 = int(ALPHA * y1 + (1 - ALPHA) * py1)
                    x2 = int(ALPHA * x2 + (1 - ALPHA) * px2)
                    y2 = int(ALPHA * y2 + (1 - ALPHA) * py2)
                smooth[tid] = (x1, y1, x2, y2)
                hist_miss[tid] = 0
                raw_box = (int(x1), int(y1), int(x2), int(y2))
                ax1, ay1, ax2, ay2 = apply_area_limit(
                    *raw_box, area_mode)
                now_on = bool(global_on.get(tid, False))
                should = ((not now_on) if invert
                          else (bool(mask_on.get(tid, False))
                                or bool(blanket_cats)))
                mf = None
                if masks_full is not None and i < len(masks_full):
                    try:
                        poly = masks_full[i]
                        if poly is None:
                            mf = None
                        elif invert:
                            mfull = _poly_roi_mask(
                                poly, 0, 0, frame.shape[1],
                                frame.shape[0])
                            if mfull is None:
                                raise ValueError("빈 폴리곤")
                            if area_mode != "full":
                                tmp = np.zeros_like(mfull)
                                tmp[max(0, ay1):ay2,
                                    max(0, ax1):ax2] = 255
                                mfull = cv2.bitwise_and(
                                    mfull, mfull, mask=tmp)
                            hist_mask[tid] = mfull
                            mf = mfull
                        else:
                            raw_mf = _poly_roi_mask(
                                poly, ax1, ay1, ax2, ay2)
                            if raw_mf is not None:
                                mf = _blend_mask(
                                    raw_mf, hist_mask.get(tid))
                                hist_mask[tid] = raw_mf
                    except Exception:
                        log.exception("마스크 생성 실패 tid=%s", tid)
                        mf = hist_mask.get(tid)
                else:
                    mf = hist_mask.get(tid)
                items.append((tid, ax1, ay1, ax2, ay2, should, mf,
                              raw_box))

    def _compute_keep_union(self, fc):
        keep_union = None
        if self.invert:
            for (_tid, _x1, _y1, _x2, _y2, _should, _mf,
                 _raw) in fc.items:
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
        fc.keep_union = keep_union

    def _render_items(self, p, plan, rs, fc):
        frame = fc.frame
        invert = self.invert
        style, blocks, blur = self.style, self.blocks, self.blur
        area_mode = self.area_mode
        direct_cats = plan.direct_cats
        person_box_samples = p.person_box_samples
        sidx = fc.sidx
        done = rs.done
        W, H = p.W, p.H
        keep_union = fc.keep_union
        face_only_person = plan.face_only_person
        faces_now = fc.faces_now
        for (tid, x1, y1, x2, y2, should, mf, raw_box) in fc.items:
            fb_in = (_faces_in_person(faces_now, raw_box)
                     if face_only_person else [])
            if direct_cats and not invert:
                for cat in direct_cats:
                    # 얼굴은 로컬 검출에서만, 동승자는 얼굴이 이미 모든
                    # 사람에게 적용된 경우 VLM 사각형을 중복으로 그리지 않는다.
                    if cat == "face":
                        continue
                    if cat == "passenger" and "face" in plan.blanket_cats:
                        continue
                    if should and cat == "passenger" and mf is not None:
                        continue
                    rels = _nearest_rel_boxes(
                        person_box_samples.get(cat), sidx, tid,
                        frame_idx=done)
                    if _render_rel_boxes(
                            frame, raw_box, rels, style, blocks,
                            blur, area_mode=area_mode,
                            clip_area=(cat not in ("name_tag",
                                                   "face")),
                            cat=cat):
                        rs.rendered_tids.add(tid)
                        rs.rendered_direct_tids.add(tid)
                        rs.rendered_cats.add(cat)
            if not should:
                continue
            if face_only_person:
                if fb_in:
                    for _fb in fb_in:
                        self._draw_face(p, rs, fc, _fb)
                    rs.rendered_tids.add(tid)
                    continue
                hb = _clip_head(_head_fallback_box(
                    raw_box, mf, (x1, y1)), W, H)
                if hb:
                    hc = getattr(rs, "head_poly_cache", None)
                    hp = None
                    if hc is not None:
                        hp, _f = hc.polygon(frame, tid, hb)
                    if (hp is not None
                            and _poly_covers_box(
                                hp, _padded_head_box(frame, hb), 0.80)
                            and _poly_size_ok(hp, hb)
                            and render_face_polygon(
                            frame, hp, style=style, blocks=blocks,
                            blur=blur)):
                        rs.rendered_tids.add(tid)
                        rs.rendered_cats.add("face")
                        continue
                    if _render_face_box(
                            frame, hb, style, blocks, blur,
                            mesh=getattr(rs, "face_mesh", None)):
                        rs.rendered_tids.add(tid)
                        rs.rendered_cats.add("face")
                        continue
            render_mask = mf
            if invert and render_mask is not None \
                    and keep_union is not None:
                try:
                    render_mask = cv2.bitwise_and(
                        render_mask, render_mask,
                        mask=cv2.bitwise_not(keep_union))
                except Exception:
                    pass
            if invert:
                ok_render = render_person_strict(
                    frame, x1, y1, x2, y2, render_mask,
                    style=style, blocks=blocks, blur=blur,
                    area_mode=area_mode)
            else:
                ok_render = render_person_roi(
                    frame, x1, y1, x2, y2, render_mask,
                    style=style, blocks=blocks, blur=blur,
                    area_mode=area_mode)
            if ok_render:
                rs.rendered_tids.add(tid)

        for _it in fc.items:
            fc.debug_boxes.append((_it[7], (0, 255, 255), f"P{_it[0]}"))

    def _render_ghosts(self, p, plan, rs, fc):
        """트랙이 잠깐 사라진 프레임도 같은 카테고리면 계속 가린다."""
        frame = fc.frame
        invert = self.invert
        style, blocks, blur = self.style, self.blocks, self.blur
        area_mode = self.area_mode
        direct_cats = plan.direct_cats
        blanket_cats = plan.blanket_cats
        global_on, mask_on = plan.global_on, plan.mask_on
        person_box_samples = p.person_box_samples
        smooth, hist_mask, hist_miss = rs.smooth, rs.hist_mask, rs.hist_miss
        MAX_MISS = rs.MAX_MISS
        sidx = fc.sidx
        done = rs.done
        W, H = p.W, p.H
        seen = fc.seen
        keep_union = fc.keep_union
        face_only_person = plan.face_only_person
        faces_now = fc.faces_now
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
            _mf = hist_mask.get(tid)
            if face_only_person and hist_miss[tid] > 2:
                # 얼굴 전용 모드에서 사라진 사람의 마지막 머리 위치에
                # 고스트 모자이크를 남기지 않는다.
                continue
            if direct_cats and not invert:
                for cat in direct_cats:
                    if cat in ("face", "passenger"):
                        continue
                    rels = _nearest_rel_boxes(
                        person_box_samples.get(cat), sidx, tid,
                        frame_idx=done)
                    if _render_rel_boxes(
                            frame, raw_box, rels, style, blocks,
                            blur, area_mode=area_mode,
                            clip_area=(cat not in ("name_tag",
                                                   "face")),
                            cat=cat):
                        rs.rendered_tids.add(tid)
                        rs.rendered_direct_tids.add(tid)
                        rs.rendered_cats.add(cat)
            if invert:
                if global_on.get(tid, False):
                    continue
            elif (not mask_on.get(tid, False)
                  and not blanket_cats):
                continue
            x1, y1, x2, y2 = apply_area_limit(
                *raw_box, area_mode)
            if face_only_person:
                _fbs = _faces_in_person(faces_now, raw_box)
                if _fbs:
                    for _fb in _fbs:
                        self._draw_face(p, rs, fc, _fb)
                    rs.rendered_tids.add(tid)
                    continue
                _hb = _clip_head(_head_fallback_box(
                    raw_box, _mf, (x1, y1)), W, H)
                if _hb:
                    _hc = getattr(rs, "head_poly_cache", None)
                    _hp = None
                    if _hc is not None:
                        _hp, _hf = _hc.polygon(frame, tid, _hb)
                    if (_hp is not None
                            and _poly_covers_box(
                                _hp, _padded_head_box(frame, _hb), 0.80)
                            and _poly_size_ok(_hp, _hb)
                            and render_face_polygon(
                            frame, _hp, style=style, blocks=blocks,
                            blur=blur)):
                        rs.rendered_tids.add(tid)
                        rs.rendered_cats.add("face")
                        continue
                    if _render_face_box(
                            frame, _hb, style, blocks, blur,
                            mesh=getattr(rs, "face_mesh", None)):
                        rs.rendered_tids.add(tid)
                        rs.rendered_cats.add("face")
                        continue
            if invert and _mf is not None and keep_union is not None:
                try:
                    _mf = cv2.bitwise_and(
                        _mf, _mf, mask=cv2.bitwise_not(keep_union))
                except Exception:
                    pass
            if invert:
                ok_render = render_person_strict(
                    frame, x1, y1, x2, y2, _mf,
                    style=style, blocks=blocks, blur=blur,
                    area_mode=area_mode)
            else:
                ok_render = render_person_roi(
                    frame, x1, y1, x2, y2, _mf,
                    style=style, blocks=blocks, blur=blur,
                    area_mode=area_mode)
            if ok_render:
                rs.rendered_tids.add(tid)

    def _render_untracked_direct(self, p, plan, rs, fc):
        """사람 트랙에 매칭되지 않은 사람 부착 박스도 최근 절대 좌표로 가린다."""
        frame = fc.frame
        direct_cats = plan.direct_cats
        if direct_cats and not self.invert:
            for cat in direct_cats:
                # 얼굴·동승자는 사람 추적/YuNet 경로에서만 그린다. 트랙에
                # 붙지 않은 VLM 좌표를 사각형으로 그리면 몸통·손·배경에
                # 모자이크가 생긴다.
                if cat in ("face", "passenger"):
                    continue
                boxes = _missing_abs_boxes(
                    p.person_box_samples.get(cat), fc.sidx, fc.seen,
                    max_age=2, frame_idx=rs.done)
                for bx in _dedupe_boxes(
                        [{"xyxy": b} for _tid, b in boxes]):
                    if not plausible_box(
                            cat, *bx["xyxy"],
                            frame.shape[1], frame.shape[0]):
                        continue
                    if render_region(frame, *bx["xyxy"],
                                     style=self.style, blocks=self.blocks,
                                     blur=self.blur, pad=4):
                        rs.rendered_direct_tids.add(-1)
                        rs.rendered_cats.add(cat)
                        rs.untracked_cats.add(cat)

    def _render_regions(self, p, plan, rs, fc):
        """배경 영역 렌더 (체크리스트 VLM 박스 + 번호판 로컬 검출)."""
        frame = fc.frame
        W, H = p.W, p.H
        if "license_plate" in plan.region_cats:
            try:
                from .plates import _every as _plate_every
                from .plates import detect_plates as _detect_plates
                gray = cv2.cvtColor(fc.r.orig_img, cv2.COLOR_BGR2GRAY)
                dets = []
                if rs.done % max(1, _plate_every()) == 0:
                    dets = _detect_plates(fc.r.orig_img) or []
                if getattr(rs, "plate_tracker", None) is not None:
                    rs.plate_tracker.update(
                        dets, gray=gray,
                        prev_gray=getattr(rs, "prev_gray_plate", None))
                    rs.plate_dets = rs.plate_tracker.boxes()
                else:
                    if dets:
                        rs.plate_dets = dets
                rs.prev_gray_plate = gray
            except Exception:
                log.exception("번호판 트랙 갱신 실패")
        if getattr(rs, "plate_dets", None) \
                and getattr(rs, "plate_tracker", None) is None \
                and rs.done - getattr(rs, "plate_updated", 0) > 30:
            rs.plate_dets = []
        for cat in plan.region_cats:
            boxes = _flow_region_boxes(
                rs.region_flow, cat, p.region_samples.get(cat),
                fc.sidx, rs.done, p.keyframe_stride)
            if cat == "license_plate" and getattr(rs, "plate_dets", None):
                try:
                    boxes = RegionFlow.merge(
                        boxes,
                        [dict(xyxy=list(b), conf=0.9,
                              category="license_plate",
                              region_type="region", source="local-plate")
                         for b in rs.plate_dets])
                except Exception:
                    pass
            persons_now = []
            for _it in getattr(fc, "items", []) or []:
                try:
                    persons_now.append(_it[7])
                except (TypeError, IndexError):
                    pass
            for b in _dedupe_boxes(boxes):
                x1, y1, x2, y2 = b["xyxy"]
                if not plausible_box(cat, x1, y1, x2, y2, W, H):
                    continue
                # 옷/가방/몸통 위 간판·휴대폰 박스가 사람을 크게 덮으면
                # 배경 영역으로 보지 않는다. (얼굴만 모드와 같은 기준)
                if persons_now and _region_on_person(
                        (x1, y1, x2, y2), persons_now):
                    continue
                if not region_has_detail(frame, (x1, y1, x2, y2),
                                         cat):
                    continue
                if render_region(frame, x1, y1, x2, y2,
                                 style=self.style, blocks=self.blocks,
                                 blur=self.blur,
                                 pad=region_pad_for(
                                     cat, x2 - x1, y2 - y1)):
                    rs.rendered_region_cats.add(cat)
                    rs.rendered_cats.add(cat)
                    fc.debug_boxes.append(
                        ((x1, y1, x2, y2), (0, 255, 0), cat))

    # ---- 결과 stats ----
    def _build_stats(self, p, plan, rs):
        votes = p.votes
        cat_on = plan.cat_on
        blanket_cats = plan.blanket_cats
        if blanket_cats:
            for tid in rs.rendered_tids:
                cat_on.setdefault(tid, {})
                for cat in blanket_cats:
                    cat_on[tid].setdefault(cat, True)
            rs.rendered_cats.update(blanket_cats)

        stats = {}
        for tid in sorted(set(votes) | set(cat_on) | rs.rendered_tids):
            stats[tid] = {
                "mosaic": tid in rs.rendered_tids,
                "direct": tid in rs.rendered_direct_tids,
                "frames": int(rs.track_frames.get(tid, 0)),
                "selector": "vlm",
                "query_raw": self.query_raw,
                "targets": list(self.targets),
                "hit_targets": plan.global_hit_targets.get(tid, []),
                "cat_on": {k: v for k, v in cat_on.get(tid, {}).items()
                           if v},
                "frac": plan.global_detail.get(tid, {}).get("frac", 0.0),
                "sample": plan.global_detail.get(tid, {}).get("sample", {}),
            }
        stats["_regions"] = len(rs.rendered_region_cats)
        stats["_cats"] = sorted(rs.rendered_cats)
        stats["_untracked"] = sorted(rs.untracked_cats)
        stats["_vlm_calls"] = int(p.vlm_budget.used_this_video)
        stats["_vlm_errors"] = int(p.vlm_errors)
        return stats


def mosaic_video(input_path, output_path, targets=None, invert=False,
                 conf=0.20, model_name="yolo11n-seg.pt", imgsz=480,
                 blocks=30, style="pixel", blur=61, ultra=False,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=15, progress_fn=None,
                 query_raw="", api_key=None, vlm_model=VLM_MODEL,
                 vlm_base_url=None, region_priors=None, person_priors=None,
                 cache_path=None, use_cache=False,
                 vlm_call_limit=None, mask_all_persons=False,
                 manual_regions=None, debug_path=None, cancel_fn=None,
                 mux_audio=True):
    """호환 래퍼."""
    job = MosaicJob(input_path, output_path, targets=targets, invert=invert,
                    conf=conf, model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur, ultra=ultra,
                    area_mode=area_mode, encode_preset=encode_preset,
                    keyframe_stride=keyframe_stride, query_raw=query_raw,
                    api_key=api_key, vlm_model=vlm_model,
                    vlm_base_url=vlm_base_url,
                    region_priors=region_priors,
                    person_priors=person_priors,
                    cache_path=cache_path, use_cache=use_cache,
                    vlm_call_limit=vlm_call_limit,
                    mask_all_persons=mask_all_persons,
                    manual_regions=manual_regions,
                    debug_path=debug_path, cancel_fn=cancel_fn,
                    mux_audio=mux_audio)
    return job.run(progress_fn)
