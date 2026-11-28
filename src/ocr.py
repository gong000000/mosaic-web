"""OCR text-line masks for plates and document/address regions.

The project used EAST only to correct VLM boxes.  This module adds a
PaddleOCR-style detector/recognizer through RapidOCR and returns the *text
line* boxes, not the whole plate/paper rectangle.  A plate therefore keeps its
border and background untouched, and a sign/document masks only the glyphs.

The OCR engine is optional.  If RapidOCR is not installed, or the crop is too
blurry to recognize, callers fall back to the existing full-region mask.
"""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("mosaic.ocr")

_LOCK = threading.RLock()
_ENGINE = None
_ENGINE_ERR = False


def _enabled(kind="general"):
    if kind == "plate":
        return os.environ.get("MOSAIC_OCR_PLATE",
                              os.environ.get("MOSAIC_OCR", "1")) != "0"
    if kind == "doc":
        return os.environ.get("MOSAIC_OCR_DOC",
                              os.environ.get("MOSAIC_OCR", "1")) != "0"
    return os.environ.get("MOSAIC_OCR", "1") != "0"


def _conf(default=0.40):
    try:
        return float(os.environ.get("MOSAIC_OCR_CONF", str(default)))
    except (TypeError, ValueError):
        return float(default)


def available():
    return _get_engine() is not None


def _get_engine():
    global _ENGINE, _ENGINE_ERR
    if os.environ.get("MOSAIC_APP_HEAVY", "1") == "0":
        return None
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_ERR or not _enabled():
        return None
    with _LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _ENGINE_ERR:
            return None
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ENGINE = RapidOCR()
            log.info("RapidOCR loaded")
        except Exception:
            log.exception("RapidOCR load failed; falling back to region masks")
            _ENGINE_ERR = True
            _ENGINE = None
        return _ENGINE


def _clip_box(box, w, h, pad=0):
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    x1 = max(0, x1 - int(pad))
    y1 = max(0, y1 - int(pad))
    x2 = min(int(w), x2 + int(pad))
    y2 = min(int(h), y2 + int(pad))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def _run(crop, scale=1.0):
    eng = _get_engine()
    if eng is None or crop is None or crop.size == 0:
        return []
    try:
        use = crop
        if scale > 1.01:
            use = cv2.resize(crop, None, fx=float(scale), fy=float(scale),
                             interpolation=cv2.INTER_CUBIC)
        with _LOCK:
            res, _elapsed = eng(use)
        if not res:
            return []
        out = []
        for row in res:
            try:
                quad, text, score = row[0], str(row[1]), float(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            q = np.asarray(quad, np.float32).reshape(-1, 2) / float(scale)
            x1, y1 = float(q[:, 0].min()), float(q[:, 1].min())
            x2, y2 = float(q[:, 0].max()), float(q[:, 1].max())
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append({
                "xyxy": [x1, y1, x2, y2],
                "text": text.strip(),
                "score": score,
            })
        return out
    except Exception:
        log.exception("OCR inference failed")
        return []


def _dedupe(boxes, overlap=0.55):
    out = []
    for b in sorted(boxes, key=lambda z: -(
            (z[2] - z[0]) * (z[3] - z[1]))):
        dup = False
        for o in out:
            ix1, iy1 = max(b[0], o[0]), max(b[1], o[1])
            ix2, iy2 = min(b[2], o[2]), min(b[3], o[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            aa = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            oa = max(1.0, (o[2] - o[0]) * (o[3] - o[1]))
            if inter / min(aa, oa) >= float(overlap):
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def _pad_line(box, w, h, x_pad=0.06, y_pad=0.16, frame_w=None,
              frame_h=None):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * float(x_pad), bh * float(y_pad)
    bx1, by1 = x1 - px, y1 - py
    bx2, by2 = x2 + px, y2 + py
    if frame_w is not None and frame_h is not None:
        bx1 = max(0.0, bx1)
        by1 = max(0.0, by1)
        bx2 = min(float(frame_w), bx2)
        by2 = min(float(frame_h), by2)
    if bx2 - bx1 < 3 or by2 - by1 < 3:
        return None
    return [int(round(bx1)), int(round(by1)),
            int(round(bx2)), int(round(by2))]


def _overlap_ratio(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1.0, (inner[2] - inner[0])
                       * (inner[3] - inner[1]))


def detect_lines(frame_bgr, region_box=None, min_conf=None,
                 max_side=1600):
    """Return OCR text lines inside ``region_box`` (frame coordinates)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    h, w = frame_bgr.shape[:2]
    if region_box is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        bb = _clip_box(region_box, w, h)
        if bb is None:
            return []
        x1, y1, x2, y2 = bb
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    ch, cw = crop.shape[:2]
    if ch < 12 or cw < 12:
        return []
    scale = 1.0
    m = max(ch, cw)
    if m > int(max_side):
        scale = int(max_side) / float(m)
    elif max(ch, cw) < 320:
        scale = min(4.0, 320.0 / float(max(1, max(ch, cw))))
    rows = _run(crop, scale=scale)
    conf = _conf(0.40 if min_conf is None else min_conf)
    out = []
    for r in rows:
        if r["score"] < conf:
            continue
        text = r["text"]
        if not text:
            continue
        bx = [r["xyxy"][0] + x1, r["xyxy"][1] + y1,
              r["xyxy"][2] + x1, r["xyxy"][3] + y1]
        out.append((bx, text, r["score"]))
    return out


def plate_text_boxes(frame_bgr, plate_box, min_conf=0.30):
    """OCR the plate crop and return padded *character/line* boxes."""
    if not _enabled("plate") or frame_bgr is None:
        return []
    h, w = frame_bgr.shape[:2]
    bb = _clip_box(plate_box, w, h)
    if bb is None:
        return []
    x1, y1, x2, y2 = bb
    bw, bh = x2 - x1, y2 - y1
    if bw < 14 or bh < 7:
        return []
    try:
        from .gpu_client import enabled as _remote_on
        from .gpu_client import remote_ocr_lines
        if _remote_on():
            _remote = remote_ocr_lines(frame_bgr, bb, "plate")
            if _remote is not None:
                return _remote
    except Exception:
        log.exception("GPU worker 번호판 OCR 실패 - 로컬 대체")
    pad = int(max(4, min(18, 0.18 * max(bw, bh))))
    cx1, cy1, cx2, cy2 = _clip_box(bb, w, h, pad=pad)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []
    ch = max(1, crop.shape[0])
    scale = max(2.0, min(8.0, 200.0 / float(ch)))
    rows = _run(crop, scale=scale)
    conf = float(min_conf if min_conf is not None else _conf(0.30))
    boxes = []
    for r in rows:
        if r["score"] < conf:
            continue
        text = r["text"].strip()
        if not text:
            continue
        q = [r["xyxy"][0] + cx1, r["xyxy"][1] + cy1,
             r["xyxy"][2] + cx1, r["xyxy"][3] + cy1]
        # Keep lines that sit on the plate, allowing a few pixels outside
        # the detector box because plate detectors often trim the border.
        cx = (q[0] + q[2]) / 2.0
        cy = (q[1] + q[3]) / 2.0
        if not (x1 - 0.25 * bw <= cx <= x2 + 0.25 * bw
                and y1 - 0.25 * bh <= cy <= y2 + 0.25 * bh):
            continue
        if _overlap_ratio(q, [x1, y1, x2, y2]) < 0.25:
            continue
        # A single gigantic line is the full plate; there is no benefit in
        # splitting it, and using it would cover the whole plate anyway.
        if ((q[2] - q[0]) * (q[3] - q[1])
                > 0.90 * max(1.0, bw * bh)):
            continue
        pb = _pad_line(q, w, h, x_pad=0.03, y_pad=0.08,
                       frame_w=w, frame_h=h)
        if pb is not None:
            boxes.append(pb)
    boxes = _dedupe(boxes, overlap=0.45)
    # If OCR produced too many fragments, keep the strongest lines but do not
    # exceed the plate area.  This prevents a noisy recognition pass from
    # turning a plate into a solid mosaic.
    boxes.sort(key=lambda b: (b[1], b[0]))
    keep = []
    total = 0.0
    for b in boxes:
        area = (b[2] - b[0]) * (b[3] - b[1])
        if total + area > 0.95 * bw * bh and keep:
            continue
        keep.append(b)
        total += area
    return keep


class OcrLineCache:
    """Small per-render cache so a moving sign is not OCR'd every frame."""

    def __init__(self, refresh_every=8, max_items=32):
        self.refresh_every = max(1, int(refresh_every))
        self.max_items = max(4, int(max_items))
        self.items = []

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def lines_for(self, frame_bgr, cat, box, frame_idx,
                  refresh_every=None, min_conf=0.35):
        if not _enabled("doc") or frame_bgr is None:
            return []
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except (TypeError, ValueError, IndexError):
            return []
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = self._center(box)
        best = None
        best_score = 1e9
        for it in self.items:
            if it["cat"] != cat:
                continue
            ob = it["box"]
            ocx, ocy = self._center(ob)
            ow, oh = max(1.0, ob[2] - ob[0]), max(1.0, ob[3] - ob[1])
            if 0.45 > bw / ow or bw / ow > 2.2:
                continue
            if 0.45 > bh / oh or bh / oh > 2.2:
                continue
            d = (((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
                 / max(1.0, max(bw, bh)))
            if d > 0.55:
                continue
            if d < best_score:
                best, best_score = it, d
        every = (self.refresh_every if refresh_every is None
                 else max(1, int(refresh_every)))
        if best is not None and frame_idx - int(best["frame"]) <= every:
            # Shift cached lines with the region box; flow already moved it.
            ob = best["box"]
            sx = (x2 - x1) / max(1.0, ob[2] - ob[0])
            sy = (y2 - y1) / max(1.0, ob[3] - ob[1])
            ox, oy = ob[0], ob[1]
            out = []
            for lb in best["lines"]:
                nx1 = x1 + (lb[0] - ox) * sx
                ny1 = y1 + (lb[1] - oy) * sy
                nx2 = x1 + (lb[2] - ox) * sx
                ny2 = y1 + (lb[3] - oy) * sy
                out.append([int(round(nx1)), int(round(ny1)),
                            int(round(nx2)), int(round(ny2))])
            return out
        lines = document_text_boxes(frame_bgr, [x1, y1, x2, y2],
                                    min_conf=min_conf)
        it = {"cat": cat, "box": [x1, y1, x2, y2],
              "lines": [list(v) for v in lines], "frame": int(frame_idx)}
        self.items = [z for z in self.items if z is not best]
        self.items.append(it)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        return lines


def project_text_boxes(ref_plates, plates):
    """Move OCR line boxes from reference plate boxes to current plates."""
    refs = list(ref_plates or [])
    cur = list(plates or [])
    if not refs or not cur:
        return []
    out = []
    for pb in cur:
        try:
            px1, py1, px2, py2 = [float(v) for v in pb[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        best, best_score = None, 1e9
        for rb, _lines in refs:
            try:
                rx1, ry1, rx2, ry2 = [float(v) for v in rb[:4]]
            except (TypeError, ValueError, IndexError):
                continue
            rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            size = max(1.0, px2 - px1, py2 - py1,
                       rx2 - rx1, ry2 - ry1)
            d = (((pcx - rcx) ** 2 + (pcy - rcy) ** 2) ** 0.5) / size
            if d < best_score:
                best, best_score = (rb, _lines), d
        if best is None or best_score > 0.75:
            continue
        rb, lines = best
        rx1, ry1, rx2, ry2 = [float(v) for v in rb[:4]]
        sx = (px2 - px1) / max(1.0, rx2 - rx1)
        sy = (py2 - py1) / max(1.0, ry2 - ry1)
        for lb in lines:
            nx1 = px1 + (float(lb[0]) - rx1) * sx
            ny1 = py1 + (float(lb[1]) - ry1) * sy
            nx2 = px1 + (float(lb[2]) - rx1) * sx
            ny2 = py1 + (float(lb[3]) - ry1) * sy
            if nx2 - nx1 < 3 or ny2 - ny1 < 3:
                continue
            out.append([int(round(nx1)), int(round(ny1)),
                        int(round(nx2)), int(round(ny2))])
    return _dedupe(out, overlap=0.55)


def document_text_boxes(frame_bgr, region_box, min_conf=0.35,
                        max_lines=40):
    """OCR a document/sign region and return padded text-line boxes."""
    if not _enabled("doc") or frame_bgr is None:
        return []
    try:
        from .gpu_client import enabled as _remote_on
        from .gpu_client import remote_ocr_lines
        if _remote_on() and region_box is not None:
            _remote = remote_ocr_lines(frame_bgr, region_box, "doc")
            if _remote is not None:
                return _remote
    except Exception:
        log.exception("GPU worker 문서 OCR 실패 - 로컬 대체")
    rows = detect_lines(frame_bgr, region_box=region_box,
                        min_conf=min_conf)
    if not rows:
        return []
    h, w = frame_bgr.shape[:2]
    bb = _clip_box(region_box, w, h)
    if bb is None:
        return []
    x1, y1, x2, y2 = bb
    rw, rh = max(1, x2 - x1), max(1, y2 - y1)
    boxes = []
    for q, text, score in rows:
        text = (text or "").strip()
        if not text:
            continue
        if _overlap_ratio(q, [x1, y1, x2, y2]) < 0.50:
            continue
        # Reject a line that is almost the whole region: OCR has merged a
        # sign into one giant quad and a text-line mask would be no better
        # than the old full-box mask.
        if (q[2] - q[0]) > 0.92 * rw and (q[3] - q[1]) > 0.55 * rh:
            continue
        pb = _pad_line(q, w, h, x_pad=0.04, y_pad=0.16,
                       frame_w=w, frame_h=h)
        if pb is not None:
            boxes.append(pb)
    boxes = _dedupe(boxes, overlap=0.60)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes[:int(max_lines)]
