"""HTTP client for the optional separate GPU refiner."""
from __future__ import annotations

import base64
import logging
import os

import cv2
import numpy as np
import requests

log = logging.getLogger("mosaic.gpu_client")


def worker_url():
    return os.environ.get("MOSAIC_GPU_WORKER", "").strip().rstrip("/")


def enabled():
    return bool(worker_url())


def _post(path, payload, timeout=90):
    base = worker_url()
    if not base:
        return None
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MOSAIC_WORKER_TOKEN", "").strip()
    if token:
        headers["X-Worker-Token"] = token
    try:
        r = requests.post(base + path, json=payload, headers=headers,
                          timeout=float(timeout))
        if r.status_code != 200:
            log.warning("GPU worker %s HTTP %s", path, r.status_code)
            return None
        return r.json()
    except Exception as exc:
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
    if not enabled() or frame_bgr is None or not boxes:
        return None
    img = _encode(frame_bgr)
    if not img:
        return None
    data = _post("/v1/face-masks", {
        "image_b64": img,
        "boxes": [[float(v) for v in b[:4]] for b in boxes],
        "use_parse": bool(use_parse),
    }, timeout=timeout)
    if not data or not isinstance(data.get("polygons"), list):
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
        return None
    return out


def remote_ocr_lines(frame_bgr, region, kind="doc", timeout=60):
    if not enabled() or frame_bgr is None:
        return None
    img = _encode(frame_bgr)
    if not img:
        return None
    data = _post("/v1/ocr/lines", {
        "image_b64": img,
        "region": ([float(v) for v in region[:4]]
                   if region is not None else None),
        "kind": str(kind),
    }, timeout=timeout)
    if not data or not isinstance(data.get("boxes"), list):
        return None
    out = []
    try:
        for b in data["boxes"]:
            out.append([int(round(float(v))) for v in b[:4]])
    except (TypeError, ValueError, IndexError):
        return None
    return out
