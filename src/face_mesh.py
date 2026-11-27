"""MediaPipe 얼굴 랜드마크로 사람마다 다른 얼굴 외곽 폴리곤을 만든다.

YuNet 박스 + 고정 23점 폴리곤은 모든 사람에게 같은 모양이라 '도장을 찍은 것
처럼 보인다. FaceLandmarker(478점)의 convex hull을 쓰면 이마·광대·턱선이
사람마다 달라진다. YuNet/기하/사람 게이트를 통과한 얼굴에만 실행하고,
실패하면 고정 폴리곤으로 대체한다(노출 방지).

모델: weights/face_landmarker.task (약 3.7MB, 빌드 때 함께 업로드).
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .config import ROOT_DIR

MODEL_PATH = os.path.join(ROOT_DIR, "weights", "face_landmarker.task")
# 배포 이미지는 /opt/weights에 모델을 빌드 때 내려받는다.
DEPLOY_MODEL_PATH = "/opt/weights/face_landmarker.task"

# hull을 얼마나 부풀릴지. 랜드마크 hull은 피부에 딱 붙어서 이마·헤어라인이
# 남으면 재검출·신원 파악이 가능하므로 25% 키워 모발 경계까지 덮는다.
GROW_RATIO = 0.25
MIN_SIDE = 40


class FaceMeshRefiner:
    """지연 로딩 싱글톤. 모델이 없으면 항상 None을 돌려준다."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path=None):
        if model_path is None:
            if os.path.exists(DEPLOY_MODEL_PATH):
                model_path = DEPLOY_MODEL_PATH
            else:
                model_path = MODEL_PATH
        self.model_path = model_path
        self._landmarker = None
        self._ok = None

    @classmethod
    def shared(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self):
        if self._ok is not None:
            return self._ok
        if not os.path.exists(self.model_path):
            self._ok = False
            return False
        if os.environ.get("MOSAIC_FACE_MESH", "1") == "0":
            self._ok = False
            return False
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as bopts
            opts = vision.FaceLandmarkerOptions(
                base_options=bopts.BaseOptions(
                    model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False)
            self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            self._ok = True
        except Exception:
            self._landmarker = None
            self._ok = False
        return self._ok

    def polygon_for_box(self, frame_bgr, box):
        """얼굴 박스에서 외곽 폴리곤(프레임 좌표). 실패하면 None.

        박스보다 살짝 넓게 잘라 랜드마커에 주고, hull을 12% 키워 돌려준다.
        """
        if not self.available():
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame_bgr.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                return None
            pad = int(0.30 * max(bw, bh))
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if min(crop.shape[:2]) < MIN_SIDE:
                return None
            # 작은 얼굴은 원본 crop 그대로 넣으면 FaceLandmarker가
            # 놓치기 쉽다. 최소 192px까지 확대한 뒤 좌표를 원래
            # 프레임 좌표로 되돌린다. (검출 결과만 확대, 원본은 유지)
            scale = max(1.0, min(3.0, 192.0 / max(1.0, min(crop.shape[:2]))))
            if scale > 1.05:
                crop_for_mesh = cv2.resize(
                    crop, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
            else:
                crop_for_mesh = crop
                scale = 1.0
            import mediapipe as mp
            rgb = cv2.cvtColor(crop_for_mesh, cv2.COLOR_BGR2RGB)
            res = self._landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                return None
            fl = res.face_landmarks[0]
            pts = np.array(
                [[p.x * (cx2 - cx1) / scale + cx1,
                  p.y * (cy2 - cy1) / scale + cy1]
                 for p in fl], np.float32)
            hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) < 8:
                return None
            # 중심 기준 25% 확대 + 단순화(떨림 방지).
            c = hull.mean(axis=0, keepdims=True)
            grown = c + (hull - c) * (1.0 + GROW_RATIO)
            eps = 0.008 * cv2.arcLength(
                grown.reshape(-1, 1, 2).astype(np.float32), True)
            approx = cv2.approxPolyDP(
                grown.reshape(-1, 1, 2).astype(np.float32),
                float(max(1.0, eps)), True).reshape(-1, 2)
            if len(approx) < 6:
                approx = grown
            poly = np.round(approx).astype(np.int32)
            # 박스에서 너무 벗어나면 랜드마크 오류로 보고 버린다.
            px1, py1 = float(poly[:, 0].min()), float(poly[:, 1].min())
            px2, py2 = float(poly[:, 0].max()), float(poly[:, 1].max())
            if px1 < x1 - 0.5 * bw or py1 < y1 - 0.6 * bh:
                return None
            if px2 > x2 + 0.5 * bw or py2 > y2 + 0.6 * bh:
                return None
            return poly
        except Exception:
            return None


def _box_iou(a, b):
    ax1, ay1 = float(a[0]), float(a[1])
    ax2, ay2 = float(a[2]), float(a[3])
    bx1, by1 = float(b[0]), float(b[1])
    bx2, by2 = float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    ba = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return inter / max(1.0, aa + ba - inter)


def _poly_iou(a, b):
    try:
        inter, _ = cv2.intersectConvexConvex(
            np.asarray(a, np.float32).reshape(-1, 1, 2),
            np.asarray(b, np.float32).reshape(-1, 1, 2))
        inter = float(inter)
    except Exception:
        inter = 0.0
    if inter <= 0:
        return 0.0
    aa = abs(cv2.contourArea(np.asarray(a, np.float32).reshape(-1, 1, 2)))
    ba = abs(cv2.contourArea(np.asarray(b, np.float32).reshape(-1, 1, 2)))
    return inter / max(1.0, aa + ba - inter)


def _resample_closed(poly, n=48):
    """닫힌 폴리곤을 둘레 길이 기준 n개 점으로 다시 샘플링.

    시작점을 '가장 위쪽 점'으로 맞춰 프레임마다 점 대응이 뒤섞이지 않게
    한다. (대응이 틀리면 EMA가 모양을 망가뜨린다.)
    """
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(p) < 3:
        return None
    # 방향을 일정하게(면적 부호) 맞춘다.
    area = 0.0
    for i in range(len(p)):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % len(p)]
        area += x1 * y2 - x2 * y1
    if area < 0:
        p = p[::-1]
    # 가장 위쪽(같으면 왼쪽) 점에서 시작.
    start = int(np.lexsort((p[:, 0], p[:, 1]))[0])
    p = np.roll(p, -start, axis=0)
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-6:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, int(n), endpoint=False)
    out = np.empty((int(n), 2), np.float32)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(len(seg) - 1, j))
        r = (t - cum[j]) / max(1e-6, seg[j])
        out[i] = closed[j] + (closed[j + 1] - closed[j]) * r
    return out


def _to_rel(poly, box):
    """프레임 좌표 폴리곤을 박스 기준 0~1 상대좌표로."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    p = np.asarray(poly, np.float32).reshape(-1, 2).copy()
    p[:, 0] = (p[:, 0] - x1) / w
    p[:, 1] = (p[:, 1] - y1) / h
    return p


def _from_rel(rel, box):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    p = np.asarray(rel, np.float32).copy()
    p[:, 0] = p[:, 0] * w + x1
    p[:, 1] = p[:, 1] * h + y1
    return np.round(p).astype(np.int32)


class FacePolyCache:
    """트랙별 얼굴 폴리곤 캐시. 깜빡임/지연 방지.

    폴리곤을 '현재 얼굴 박스 기준 0~1 상대좌표(float32)'로 보관한다.
    매 프레임 현재 박스에 투영하므로 이동·확대를 지연 0px로 따라간다.
    (예전처럼 int32로 이동량을 누적하면 0.7px 이동이 매번 잘려
     72프레임 뒤 25px까지 뒤처졌다.)
    mesh는 refresh_every 프레임마다만 돌리고, 새 모양은 둘레 기준으로
    같은 점 개수로 리샘플한 뒤 EMA로 섞어 모양 떨림을 없앤다.
    """

    def __init__(self, mesh=None, refresh_every=0, alpha=0.30):
        self.mesh = mesh
        self.refresh_every = max(0, int(refresh_every))
        self.alpha = min(0.9, max(0.05, float(alpha)))
        self.mem = {}
        self.buckets = {}

    def stable_side(self, tid, w, h):
        """트랙별 모자이크 블록 크기 기준(짧은 축)을 안정화한다.

        얼굴 크기가 1~2px 변할 때마다 블록 크기가 바뀌면 타일 전체가
        다시 그려져 깜빡인다. 16px 버킷으로 양자화하고, 25% 이상
        커지거나 작아질 때만 갱신한다.
        """
        side = max(1, int(min(int(w), int(h))))
        b = self.buckets.get(tid)
        if b and 0.75 * b <= side <= 1.35 * b:
            return b
        b = max(16, int(round(side / 16.0)) * 16)
        self.buckets[tid] = b
        return b

    def polygon(self, frame_bgr, tid, box, force=False):
        """(폴리곤 int32 또는 None, 이번 프레임에 mesh를 새로 돌렸는지)."""
        try:
            bx = [float(v) for v in box]
        except (TypeError, ValueError):
            return None, False
        if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
            return None, False
        m = self.mem.get(tid)
        # 1) 캐시가 있고 아직 새로 돌릴 때가 아니면 현재 박스에 투영만.
        if m is not None and not force and m["age"] < self.refresh_every:
            m["age"] += 1
            m["box"] = bx
            return _from_rel(m["rel"], bx), False
        # 2) mesh 새로 시도.
        poly = None
        if self.mesh is not None:
            try:
                poly = self.mesh.polygon_for_box(frame_bgr, bx)
            except Exception:
                poly = None
        if poly is not None:
            rel_new = _resample_closed(_to_rel(poly, bx), 48)
            if rel_new is not None:
                blend = False
                if m is not None and m["age"] <= 4:
                    try:
                        old_proj = _from_rel(m["rel"], bx)
                        blend = _poly_iou(old_proj, poly) >= 0.72
                    except Exception:
                        blend = False
                if blend:
                    rel = (1.0 - self.alpha) * m["rel"] + self.alpha * rel_new
                else:
                    rel = rel_new
                self.mem[tid] = {"rel": rel, "box": bx, "age": 0}
                if len(self.mem) > 128:
                    self.mem.pop(next(iter(self.mem)))
                return _from_rel(rel, bx), True
        # 3) mesh 실패 → 옛 상대 모양을 현재 박스에 투영(고정형 점프 방지).
        if m is not None:
            m["age"] += 1
            m["box"] = bx
            return _from_rel(m["rel"], bx), False
        return None, False

    def drop(self, tid):
        self.mem.pop(tid, None)
        self.buckets.pop(tid, None)
