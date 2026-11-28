"""HTTP client for the optional separate GPU refiner."""
from __future__ import annotations

import base64
import logging
import os
import threading
import time

import cv2
import numpy as np
import requests

log = logging.getLogger("mosaic.gpu_client")

_lock = threading.Lock()
_stats = {
    "http_requests": 0,
    "face_requests": 0,
    "face_success": 0,
    "face_failures": 0,
    "ocr_requests": 0,
    "ocr_success": 0,
    "ocr_failures": 0,
    "disabled": 0,
    "last_ms": 0.0,
    "last_error": "",
}


def worker_url():
    return os.environ.get("MOSAIC_GPU_WORKER", "").strip().rstrip("/")


def enabled():
    return bool(worker_url())


def reset_stats():
    with _lock:
        for key in _stats:
            if isinstance(_stats[key], (int, float)):
                _stats[key] = 0
        _stats["last_error"] = ""


def snapshot():
    with _lock:
        out = dict(_stats)
    out["enabled"] = enabled()
    out["used"] = bool(out["face_success"] or out["ocr_success"])
    out["fallback"] = bool(
        out["face_failures"] or out["ocr_failures"] or out["disabled"])
    out["ok"] = bool(out["used"] and not out["fallback"])
    return out


def _mark_failure(kind, error=""):
    with _lock:
        _stats[f"{kind}_failures"] += 1
        if error:
            _stats["last_error"] = str(error)[:240]


def _post(path, payload, timeout=90):
    base = worker_url()
    if not base:
        with _lock:
            _stats["disabled"] += 1
        return None
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MOSAIC_WORKER_TOKEN", "").strip()
    if token:
        headers["X-Worker-Token"] = token
    t0 = time.time()
    try:
        r = requests.post(base + path, json=payload, headers=headers,
                          timeout=float(timeout))
        with _lock:
            _stats["http_requests"] += 1
            _stats["last_ms"] = round((time.time() - t0) * 1000.0, 1)
        if r.status_code != 200:
            body = (r.text or "")[:160]
            log.warning("GPU worker %s HTTP %s", path, r.status_code)
            with _lock:
                _stats["last_error"] = f"HTTP {r.status_code}: {body}"
            return None
        return r.json()
    except Exception as exc:
        with _lock:
            _stats["http_requests"] += 1
            _stats["last_ms"] = round((time.time() - t0) * 1000.0, 1)
            _stats["last_error"] = str(exc)[:240]
        log.warning("GPU worker %s unavailable: %s", path, exc)
        return None


def _encode(frame_bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def remote_face_masks(frame_bgr, boxes, timeout=90, use_parse=True):
    """Return [{polygon or None}] or None when the worker is unavailable."""
    with _lock:
        _stats["face_requests"] += 1
    if not enabled() or frame_bgr is None or not boxes:
        if not enabled():
            with _lock:
                _stats["disabled"] += 1
        _mark_failure("face", "worker disabled" if not enabled() else "bad input")
        return None
    img = _encode(frame_bgr)
    if not img:
        _mark_failure("face", "jpeg encode failed")
        return None
    data = _post("/v1/face-masks", {
        "image_b64": img,
        "boxes": [[float(v) for v in b[:4]] for b in boxes],
        "use_parse": bool(use_parse),
    }, timeout=timeout)
    if not data or not isinstance(data.get("polygons"), list):
        _mark_failure("face", "invalid worker response")
        return None
    out = []
    for p in data["polygons"]:
        if p is None:
            out.append(None)
        else:
            try:
                out.append(np.asarray(p, np.int32).reshape(-1, 2))
            except Exception:
                out.append(None)
    if len(out) != len(boxes):
        _mark_failure("face", "polygon count mismatch")
        return None
    with _lock:
        _stats["face_success"] += 1
    return out


def remote_ocr_lines(frame_bgr, region, kind="doc", timeout=60):
    with _lock:
        _stats["ocr_requests"] += 1
    if not enabled() or frame_bgr is None:
        if not enabled():
            with _lock:
                _stats["disabled"] += 1
        _mark_failure("ocr", "worker disabled" if not enabled() else "bad input")
        return None
    img = _encode(frame_bgr)
    if not img:
        _mark_failure("ocr", "jpeg encode failed")
        return None
    data = _post("/v1/ocr/lines", {
        "image_b64": img,
        "region": ([float(v) for v in region[:4]]
                   if region is not None else None),
        "kind": str(kind),
    }, timeout=timeout)
    if not data or not isinstance(data.get("boxes"), list):
        _mark_failure("ocr", "invalid worker response")
        return None
    out = []
    try:
        for b in data["boxes"]:
            out.append([int(round(float(v))) for v in b[:4]])
    except (TypeError, ValueError, IndexError):
        _mark_failure("ocr", "invalid box")
        return None
    with _lock:
        _stats["ocr_success"] += 1
    return out
