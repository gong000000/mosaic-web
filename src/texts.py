"""간판·문서 글자 영역 검출 (OpenCV EAST).

VLM이 찾은 주소·신분증·휴대폰 박스는 위치가 어긋나거나 너무 넓다.
박스 주변을 잘라 EAST로 글자 쿼드를 찾고, VLM 박스와 겹치는 글자들의
합집합으로 박스를 고친다. 글자가 없으면 VLM 박스를 유지한다(노출 방지).

모델: weights/frozen_east_text_detection.pb (약 96MB, Unlicense).
없으면 항상 None을 돌려주고 VLM 경로만 쓴다.
"""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("mosaic.texts")

MODEL_NAME = "frozen_east_text_detection.pb"
LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "weights", MODEL_NAME)
DEPLOY_PATH = "/opt/weights/" + MODEL_NAME

# 이 카테고리의 VLM 박스에만 텍스트 정제를 적용한다.
REFINE_CATS = {"address", "id_card", "phone_screen"}

_NET_SINGLETON = None
_net_lock = threading.Lock()


def _enabled():
    return os.environ.get("MOSAIC_TEXT_DET", "1") != "0"


def _conf():
    try:
        return float(os.environ.get("MOSAIC_TEXT_CONF", "0.6"))
    except (TypeError, ValueError):
        return 0.6


def _model_path():
    if os.path.exists(DEPLOY_PATH):
        return DEPLOY_PATH
    return LOCAL_PATH


def _get_net():
    global _NET_SINGLETON
    with _net_lock:
        if _NET_SINGLETON is not None:
            return _NET_SINGLETON
        path = _model_path()
        if not os.path.exists(path):
            return None
        try:
            detector = cv2.dnn.TextDetectionModel_EAST(
                cv2.dnn.readNet(path))
            detector.setConfidenceThreshold(_conf())
            detector.setNMSThreshold(0.4)
            globals()["_NET_SINGLETON"] = detector
            return detector
        except Exception:
            log.exception("EAST 모델 로드 실패")
            return None


def detect_text_quads(crop_bgr, conf=None):
    """크롭에서 글자 쿼드 목록. returns [4x2 array, ...] (크롭 좌표).

    EAST는 입력을 640x640으로 강제로 늘리면 세로 간판/작은 글자가
    찌그러져 검출되지 않는다. 비율을 유지해 확대/축소하고, 32의 배수로
    패딩한 뒤 원래 좌표로 되돌린다.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    det = _get_net()
    if det is None:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        return []
    # 작은 간판은 최대 6배까지 확대하고, 큰 크롭은 640 기준으로 축소한다.
    scale = min(6.0, 640.0 / float(max(w, h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    nw = max(32, ((nw + 31) // 32) * 32)
    nh = max(32, ((nh + 31) // 32) * 32)
    # 원본 비율을 유지한 채 nw x nh 안에 맞춘다(패딩은 바깥).
    fit = min(nw / float(w), nh / float(h))
    rw = max(1, int(round(w * fit)))
    rh = max(1, int(round(h * fit)))
    interp = cv2.INTER_AREA if fit < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(crop_bgr, (rw, rh), interpolation=interp)
    canvas = np.zeros((nh, nw, 3), np.uint8)
    canvas[:rh, :rw] = small
    try:
        with _net_lock:
            try:
                det.setConfidenceThreshold(
                    float(conf if conf is not None else _conf()))
                det.setInputParams(1.0, (nw, nh),
                                    (123.68, 116.78, 103.94), True)
            except Exception:
                pass
            boxes, _confs = det.detect(canvas)
    except Exception:
        return []
    if boxes is None:
        return []
    out = []
    try:
        for b in boxes:
            pts = np.asarray(b, np.float32).reshape(-1, 2)
            pts[:, 0] *= w / float(rw)
            pts[:, 1] *= h / float(rh)
            out.append(pts)
    except (TypeError, ValueError):
        return []
    return out


def detect_text_regions(frame_bgr, persons=None, vehicles=None, plates=None,
                        min_area=800, tiles=3, max_area_ratio=0.02,
                        exclude=None):
    """EAST 글자 영역을 주소·간판 후보 박스로 만든다.

    연결 컴포넌트로 크게 묶으면 건물 벽 전체가 하나의 박스가 되어 과도하게
    가려진다. 글자 쿼드 단위로 만들고, 아주 가까운 글자만 합치되 박스 면적이
    프레임의 max_area_ratio를 넘으면 더 합치지 않는다.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    h, w = frame_bgr.shape[:2]
    if h < 64 or w < 64:
        return []
    try:
        tiles = max(1, int(tiles))
    except (TypeError, ValueError):
        tiles = 3
    step_x, step_y = max(1, w // tiles), max(1, h // tiles)
    raw = []
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            x2 = min(w, x + step_x + 40)
            y2 = min(h, y + step_y + 40)
            for q in detect_text_quads(frame_bgr[y:y2, x:x2]):
                q = q.copy()
                q[:, 0] += x
                q[:, 1] += y
                bx = [int(q[:, 0].min()), int(q[:, 1].min()),
                      int(q[:, 0].max()), int(q[:, 1].max())]
                bx = [max(0, bx[0]), max(0, bx[1]),
                      min(w, bx[2]), min(h, bx[3])]
                if (bx[2] - bx[0] < 20 or bx[3] - bx[1] < 8
                        or _box_area(bx) < min_area):
                    continue
                raw.append(bx)
    max_area = max_area_ratio * float(w * h)
    merged = []
    for b in sorted(raw, key=lambda z: -_box_area(z)):
        placed = False
        for i, m in enumerate(merged):
            if not (_box_overlap(b, m) > 0.60
                    or _box_overlap(m, b) > 0.60):
                continue
            u = [min(b[0], m[0]), min(b[1], m[1]),
                 max(b[2], m[2]), max(b[3], m[3])]
            if _box_area(u) > max_area:
                continue
            if (u[2] - u[0]) > 0.55 * w or (u[3] - u[1]) > 0.40 * h:
                continue
            merged[i] = u
            placed = True
            break
        if not placed:
            merged.append(list(b))
    persons = list(persons or [])
    vehicles = list(vehicles or [])
    plates = list(plates or [])
    exclude = list(exclude or [])
    out = []
    for box in merged:
        if box[3] / 2.0 + box[1] / 2.0 > h * 0.88:
            continue
        if any(_box_overlap(box, p) > 0.35 for p in persons):
            continue
        if any(_box_overlap(box, v) > 0.50 for v in vehicles):
            continue
        if any(_box_overlap(box, p) > 0.40 for p in plates):
            continue
        if any(_box_overlap(box, e) > 0.30 for e in exclude):
            continue
        out.append(box)
    out.sort(key=lambda b: -_box_area(b))
    return out


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _box_overlap(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / float(max(1, _box_area(inner)))


def refine_box_with_text(frame_bgr, xyxy, pad_ratio=0.25):
    """VLM 박스를 안의 글자 합집합으로 고친다. 실패하면 None.

    글자가 하나도 안 보이면 None을 돌려 호출부가 VLM 박스를 유지하게 한다.
    """
    if not _enabled() or frame_bgr is None:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        ex1 = max(0, int(x1 - bw * pad_ratio))
        ey1 = max(0, int(y1 - bh * pad_ratio))
        ex2 = min(w, int(x2 + bw * pad_ratio))
        ey2 = min(h, int(y2 + bh * pad_ratio))
        if ex2 - ex1 < 32 or ey2 - ey1 < 32:
            return None
        crop = frame_bgr[ey1:ey2, ex1:ex2]
        quads = detect_text_quads(crop)
        if not quads:
            return None
        # VLM 박스(원본 좌표)와 겹치는 글자만 쓴다.
        vx1, vy1, vx2, vy2 = x1, y1, x2, y2
        kept = []
        for q in quads:
            qx = q[:, 0] + ex1
            qy = q[:, 1] + ey1
            qx1, qy1 = float(qx.min()), float(qy.min())
            qx2, qy2 = float(qx.max()), float(qy.max())
            ix1, iy1 = max(qx1, vx1), max(qy1, vy1)
            ix2, iy2 = min(qx2, vx2), min(qy2, vy2)
            if (ix2 - ix1) > 0 and (iy2 - iy1) > 0:
                kept.append((qx1, qy1, qx2, qy2))
        if not kept:
            return None
        ux1 = max(ex1, min(k[0] for k in kept))
        uy1 = max(ey1, min(k[1] for k in kept))
        ux2 = min(ex2, max(k[2] for k in kept))
        uy2 = min(ey2, max(k[3] for k in kept))
        if ux2 - ux1 < 8 or uy2 - uy1 < 6:
            return None
        return [int(round(ux1)), int(round(uy1)),
                int(round(ux2)), int(round(uy2))]
    except Exception:
        log.exception("텍스트 정제 실패")
        return None
