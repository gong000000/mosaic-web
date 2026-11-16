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
    """크롭에서 글자 쿼드 목록. returns [4x2 array, ...] (크롭 좌표)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    det = _get_net()
    if det is None:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        return []
    small = cv2.resize(crop_bgr, (640, 640), interpolation=cv2.INTER_AREA)
    try:
        with _net_lock:
            try:
                det.setConfidenceThreshold(
                    float(conf if conf is not None else _conf()))
                det.setInputParams(1.0, (640, 640),
                                    (123.68, 116.78, 103.94), True)
            except Exception:
                pass
            boxes, _confs = det.detect(small)
    except Exception:
        return []
    if boxes is None:
        return []
    out = []
    try:
        for b in boxes:
            pts = np.asarray(b, np.float32).reshape(-1, 2)
            pts[:, 0] *= w / 640.0
            pts[:, 1] *= h / 640.0
            out.append(pts)
    except (TypeError, ValueError):
        return []
    return out


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
