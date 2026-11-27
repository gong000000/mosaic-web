"""Face-shape refinement with SAM 2.1 and a face-parsing SegFormer.

The existing MediaPipe mesh is fast but its convex hull includes hair and part
of the neck on many faces.  This module adds two optional refiners:

* SAM 2.1 box-prompted masks, which follow the actual face boundary.
* A face-parsing SegFormer that labels skin / eyes / nose / mouth / ears
  separately from hair, hat, neck and clothing.

When both are available the masks are intersected.  The parser removes hair,
hats and neck from the SAM mask, while SAM keeps the outer face boundary that
the parser sometimes under-segments.  Either model can be missing; callers fall
back to the existing MediaPipe mesh or skin model.

Everything is lazy and optional.  A missing checkpoint, package, GPU or score
simply returns ``None`` so a render never fails because of this refiner.
"""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

from .config import ROOT_DIR

log = logging.getLogger("mosaic.face_seg")

SAM2_CKPT = os.path.join(ROOT_DIR, "weights", "sam2.1_hiera_tiny.pt")
SAM2_CONFIG = os.environ.get(
    "MOSAIC_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SAM2_MAX_SIDE = 1280
SAM2_MIN_SCORE = 0.45

# face-parsing label ids used by jonathandinu/face-parsing
PARSE_FACE_IDS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15})


def _env_on(name, default="1"):
    return os.environ.get(name, default) != "0"


def _clip_box(box, w, h):
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(w), x2), min(int(h), y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def _poly_area(poly):
    try:
        return abs(float(cv2.contourArea(
            np.asarray(poly, np.float32).reshape(-1, 1, 2))))
    except Exception:
        return 0.0


def _poly_bbox(poly):
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    return (float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max()))


def _polygon_from_mask(mask, box=None, dilate=2, approx=0.0035):
    """Largest component of a binary mask -> approximated polygon."""
    if mask is None or mask.size == 0:
        return None
    m = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    if m.shape[0] < 4 or m.shape[1] < 4:
        return None
    if int((m > 0).sum()) < 0.002 * m.size:
        return None
    if dilate > 0:
        m = cv2.dilate(
            m, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    cx = cy = None
    if box is not None:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
    best = None
    best_score = -1.0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 0.0015 * m.size:
            continue
        score = float(area)
        if cx is not None:
            dist = ((cents[lab][0] - cx) ** 2
                    + (cents[lab][1] - cy) ** 2) ** 0.5
            score /= (1.0 + dist * 0.02)
        if score > best_score:
            best, best_score = lab, score
    if best is None:
        return None
    comp = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.0012 * m.size:
        return None
    eps = max(0.8, float(approx) * cv2.arcLength(cnt, True))
    ap = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(ap) < 6:
        return None
    return np.round(ap).astype(np.int32)


def _polygon_quality(poly, box, center_margin=0.10):
    """Reject masks that drifted to hair, hands, paper or background."""
    if poly is None or len(poly) < 6:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        bx1, by1, bx2, by2 = _poly_bbox(poly)
        if bx2 - bx1 < 8 or by2 - by1 < 8:
            return False
        if bx2 - bx1 > 1.65 * bw or by2 - by1 > 1.65 * bh:
            return False
        if bx1 < x1 - 0.35 * bw or by1 < y1 - 0.45 * bh:
            return False
        if bx2 > x2 + 0.35 * bw or by2 > y2 + 0.45 * bh:
            return False
        area = _poly_area(poly)
        if area < 0.18 * bw * bh or area > 1.40 * bw * bh:
            return False
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cv2.pointPolygonTest(
                np.asarray(poly, np.int32).reshape(-1, 1, 2),
                (cx, cy), False) < 0:
            # A valid face can be slightly off-centre inside a loose detector
            # box; allow it when the polygon still covers the box centre area.
            mx = bw * center_margin
            my = bh * center_margin
            hit = 0
            for px in (cx - mx, cx, cx + mx):
                for py in (cy - my, cy, cy + my):
                    if cv2.pointPolygonTest(
                            np.asarray(poly, np.int32).reshape(-1, 1, 2),
                            (float(px), float(py)), False) >= 0:
                        hit += 1
            if hit < 3:
                return False
        return True
    except Exception:
        return False


def _intersect_polygons(a, b):
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = _poly_bbox(a)
    bx1, by1, bx2, by2 = _poly_bbox(b)
    x1, y1 = int(max(0, min(ax1, bx1))) - 2, int(max(0, min(ay1, by1))) - 2
    x2, y2 = int(max(ax2, bx2)) + 3, int(max(ay2, by2)) + 3
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    ma = np.zeros((y2 - y1, x2 - x1), np.uint8)
    mb = np.zeros_like(ma)
    cv2.fillPoly(ma, [np.asarray(a, np.int32) - np.array([x1, y1])], 255)
    cv2.fillPoly(mb, [np.asarray(b, np.int32) - np.array([x1, y1])], 255)
    inter = cv2.bitwise_and(ma, mb)
    if int((inter > 0).sum()) < 30:
        return None
    poly = _polygon_from_mask(inter, box=None, dilate=1, approx=0.0025)
    if poly is None:
        return None
    return poly + np.array([x1, y1], np.int32)


class FaceSegmentation:
    """Lazy SAM 2.1 + face-parsing wrapper shared by all render threads."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._sam = None
        self._sam_predictor = None
        self._sam_ok = None
        self._sam_frame_ref = None
        self._sam_scale = 1.0
        self._sam_shape = None
        self._parse_proc = None
        self._parse_model = None
        self._parse_ok = None
        self._dev = "cpu"
        self._run_lock = threading.RLock()

    @classmethod
    def shared(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self):
        return self._sam_available() or self._parse_available()

    # ------------------------------------------------------------------
    # SAM 2.1
    # ------------------------------------------------------------------
    def _sam_available(self):
        if self._sam_ok is not None:
            return self._sam_ok
        self._sam_ok = False
        if not _env_on("MOSAIC_SAM2"):
            return False
        ckpt = os.environ.get("MOSAIC_SAM2_CKPT", SAM2_CKPT)
        if not os.path.exists(ckpt):
            return False
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception:
            log.info("SAM 2 package unavailable; using mesh/skin fallback")
            return False
        try:
            dev = os.environ.get("MOSAIC_SAM2_DEVICE")
            if not dev:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._dev = dev
            model = build_sam2(
                os.environ.get("MOSAIC_SAM2_CONFIG", SAM2_CONFIG),
                ckpt, device=dev)
            self._sam = model
            self._sam_predictor = SAM2ImagePredictor(model)
            self._sam_ok = True
            log.info("SAM 2.1 loaded on %s", dev)
        except Exception:
            log.exception("SAM 2.1 load failed; using mesh/skin fallback")
            self._sam = None
            self._sam_predictor = None
            self._sam_ok = False
        return self._sam_ok

    def _set_sam_image(self, frame_bgr):
        if self._sam_predictor is None:
            return False
        try:
            h, w = frame_bgr.shape[:2]
            m = max(h, w)
            max_side = int(os.environ.get("MOSAIC_SAM2_MAX_SIDE",
                                          str(SAM2_MAX_SIDE)))
            scale = min(1.0, float(max_side) / float(max(1, m)))
            if scale < 0.999:
                sw = max(32, int(round(w * scale)))
                sh = max(32, int(round(h * scale)))
                small = cv2.resize(frame_bgr, (sw, sh),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame_bgr
                sw, sh = w, h
            if (self._sam_frame_ref is frame_bgr
                    and self._sam_shape == (h, w)):
                return True
            import torch
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                self._sam_predictor.set_image(rgb)
            self._sam_frame_ref = frame_bgr
            self._sam_scale = (sw / float(w), sh / float(h))
            self._sam_shape = (h, w)
            return True
        except Exception:
            log.exception("SAM 2.1 image embedding failed")
            self._sam_frame_ref = None
            return False

    def sam2_polygon(self, frame_bgr, box):
        if not self._sam_available():
            return None
        if not self._set_sam_image(frame_bgr):
            return None
        try:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            sx, sy = self._sam_scale
            prompt = np.array([[x1 * sx, y1 * sy,
                                x2 * sx, y2 * sy]], np.float32)
            import torch
            with self._run_lock, torch.inference_mode():
                masks, scores, _ = self._sam_predictor.predict(
                    box=prompt, multimask_output=False)
            if masks is None or len(masks) == 0:
                return None
            score = float(np.asarray(scores).reshape(-1)[0])
            if score < float(os.environ.get("MOSAIC_SAM2_MIN_SCORE",
                                            str(SAM2_MIN_SCORE))):
                return None
            m = np.asarray(masks[0]).squeeze()
            if m.ndim != 2:
                return None
            sw, sh = int(round(w * sx)), int(round(h * sy))
            if m.shape != (sh, sw):
                m = cv2.resize(m.astype(np.float32), (sw, sh),
                               interpolation=cv2.INTER_LINEAR)
            m = (m > 0.0).astype(np.uint8) * 255
            if sx < 0.999 or sy < 0.999:
                m = cv2.resize(m, (w, h),
                               interpolation=cv2.INTER_NEAREST)
            poly = _polygon_from_mask(m, box=box, dilate=1, approx=0.0028)
            if poly is None or not _polygon_quality(poly, box):
                return None
            return poly
        except Exception:
            log.exception("SAM 2.1 box prediction failed")
            return None

    # ------------------------------------------------------------------
    # Face parsing
    # ------------------------------------------------------------------
    def _parse_available(self):
        if self._parse_ok is not None:
            return self._parse_ok
        self._parse_ok = False
        if not _env_on("MOSAIC_FACE_PARSE"):
            return False
        try:
            import torch
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
        except Exception:
            return False
        try:
            if self._dev == "cpu" and torch.cuda.is_available():
                self._dev = "cuda"
            name = os.environ.get("MOSAIC_FACE_PARSE_MODEL",
                                  "jonathandinu/face-parsing")
            self._parse_proc = SegformerImageProcessor.from_pretrained(name)
            self._parse_model = (
                SegformerForSemanticSegmentation.from_pretrained(name)
                .to(self._dev).eval())
            self._parse_ok = True
            log.info("face-parsing model loaded on %s", self._dev)
        except Exception:
            log.exception("face-parsing model load failed")
            self._parse_proc = None
            self._parse_model = None
            self._parse_ok = False
        return self._parse_ok

    def face_parse_polygon(self, frame_bgr, box):
        if not self._parse_available():
            return None
        try:
            h, w = frame_bgr.shape[:2]
            bb = _clip_box(box, w, h)
            if bb is None:
                return None
            x1, y1, x2, y2 = bb
            bw, bh = x2 - x1, y2 - y1
            pad = int(0.30 * max(bw, bh))
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None
            import torch
            import torch.nn.functional as F
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = self._parse_proc(
                images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._dev) for k, v in inputs.items()}
            with self._run_lock, torch.inference_mode():
                logits = self._parse_model(**inputs).logits
                up = F.interpolate(
                    logits, size=crop.shape[:2], mode="bilinear",
                    align_corners=False)
                lab = up.argmax(1)[0].detach().cpu().numpy()
            mask = np.isin(lab, list(PARSE_FACE_IDS)).astype(np.uint8)
            if int(mask.sum()) < 0.15 * mask.size:
                return None
            poly = _polygon_from_mask(mask, box=(x1 - cx1, y1 - cy1,
                                                x2 - cx1, y2 - cy1),
                                      dilate=1, approx=0.0025)
            if poly is None:
                return None
            poly = poly + np.array([cx1, cy1], np.int32)
            if not _polygon_quality(poly, box):
                return None
            return poly
        except Exception:
            log.exception("face-parsing prediction failed")
            return None

    # ------------------------------------------------------------------
    def polygon_for_box(self, frame_bgr, box, use_parse=True,
                        use_sam2=True):
        """Best available face polygon, or None for the mesh fallback."""
        self._sam_available()
        if use_parse and self._parse_ok is None:
            self._parse_available()
        sam = (self.sam2_polygon(frame_bgr, box)
               if (use_sam2 and self._sam_ok) else None)
        parse = (self.face_parse_polygon(frame_bgr, box)
                 if (use_parse and self._parse_ok) else None)
        if sam is not None and parse is not None:
            # Intersection removes hair/hat/neck from SAM while keeping the
            # outer boundary the parser may under-segment.
            inter = _intersect_polygons(sam, parse)
            if inter is not None:
                inter = _grow_polygon(inter, box, ratio=0.012)
                if _polygon_quality(inter, box):
                    return inter
        if parse is not None and _polygon_quality(parse, box):
            # The parser gives the cleanest hair/neck boundary.  Grow a
            # little to cover the jaw/ear edge that is often one pixel thin.
            parse = _grow_polygon(parse, box, ratio=0.020)
            if _polygon_quality(parse, box):
                return parse
        if sam is not None:
            return _grow_polygon(sam, box, ratio=0.010)
        return None


def _grow_polygon(poly, box, ratio=0.015):
    try:
        p = np.asarray(poly, np.float32).reshape(-1, 2)
        c = p.mean(axis=0, keepdims=True)
        # Grow around the face center, not a distant polygon centroid.
        bx = (float(box[0]) + float(box[2])) / 2.0
        by = (float(box[1]) + float(box[3])) / 2.0
        center = np.array([[bx, by]], np.float32) * 0.35 + c * 0.65
        return np.round(center + (p - center) * (1.0 + float(ratio))
                        ).astype(np.int32)
    except Exception:
        return poly


def segment_face(frame_bgr, box):
    """Convenience wrapper used by tests and external workers."""
    try:
        return FaceSegmentation.shared().polygon_for_box(frame_bgr, box)
    except Exception:
        return None
