"""차량번호판 전용 로컬 검출기.

VLM 앵커(최대 16장)만으로는 움직이는 번호판을 놓친다. YOLO11n 번호판 모델로
3프레임마다 직접 찾아 광학 흐름으로 이어 붙인다. VLM 박스는 백업으로 유지한다.
모델이 없으면 빈 목록을 돌려주고 VLM 경로만 쓴다.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("mosaic.plates")

MODEL_NAME = os.environ.get("MOSAIC_PLATE_MODEL",
                            "license-plate-finetune-v1n.pt")
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_EVERY = 2
VEHICLE_CLASSES = (2, 3, 5, 7)


# 과거 오타("MOSAIC_PLATE_DET"처럼 중간 공백)도 읽는다.
def _env(name, default):
    return os.environ.get(name, default)


def _enabled():
    return _env("MOSAIC_PLATE_DET", "1") != "0"


def _conf():
    try:
        return float(_env("MOSAIC_PLATE_CONF", str(DEFAULT_CONF)))
    except (TypeError, ValueError):
        return float(DEFAULT_CONF)


def _every():
    try:
        return max(1, int(_env("MOSAIC_PLATE_EVERY",
                               str(DEFAULT_EVERY))))
    except (TypeError, ValueError):
        return int(DEFAULT_EVERY)


def detect_plates(frame_bgr, conf=None, imgsz=DEFAULT_IMGSZ):
    """프레임에서 번호판 박스 [[x1,y1,x2,y2], ...]. 실패하면 []."""
    if not _enabled() or frame_bgr is None:
        return []
    try:
        from .models import MODEL_LOCK, get_model
    except ImportError:
        from models import MODEL_LOCK, get_model  # type: ignore
    try:
        model = get_model(MODEL_NAME)
    except Exception:
        log.exception("번호판 모델 로드 실패 - VLM만 사용")
        return []
    try:
        with MODEL_LOCK:
            res = list(model.predict(source=frame_bgr, verbose=False,
                                     imgsz=int(imgsz),
                                     conf=float(conf if conf is not None
                                                else _conf()),
                                     max_det=10))[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
    except Exception:
        log.exception("번호판 검출 실패")
        return []
    h, w = frame_bgr.shape[:2]
    out = []
    for (x1, y1, x2, y2), c in zip(xy, cf):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 6:
            continue
        # 번호판은 가로로 길다. 너무 통통한 오검출을 버린다.
        # (US식 박스는 탐지 박스가 헐렁해 1.5 기준이면 진짜를 버린다.)
        if bw < 1.2 * bh:
            continue
        if bw > 0.9 * w or bh > 0.4 * h:
            continue
        out.append([x1, y1, x2, y2])
    # 큰 것(가까운 차) 먼저. 겹치면 큰 것만 남긴다.
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in out:
        if any(_iou(b, k) > 0.5 for k in kept):
            continue
        kept.append(b)
    return kept


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
    ba = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    return inter / float(max(1, aa + ba - inter))


def detect_vehicles(frame_bgr, model, conf=0.30, imgsz=640):
    """차량 박스 [[x1,y1,x2,y2], ...]. 실패하면 []."""
    if frame_bgr is None or model is None:
        return []
    try:
        from .models import MODEL_LOCK
    except ImportError:
        from models import MODEL_LOCK  # type: ignore
    try:
        with MODEL_LOCK:
            res = list(model.predict(source=frame_bgr, verbose=False,
                                     imgsz=int(imgsz), conf=float(conf),
                                     classes=list(VEHICLE_CLASSES),
                                     max_det=15))[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xy = res.boxes.xyxy.cpu().numpy()
    except Exception:
        log.exception("차량 검출 실패")
        return []
    h, w = frame_bgr.shape[:2]
    out = []
    for x1, y1, x2, y2 in xy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 >= 24 and y2 - y1 >= 18:
            out.append([x1, y1, x2, y2])
    return out


def filter_by_vehicles(plates, vehicles, margin=0.10):
    """차량 안에 중심이 있는 번호판만 남긴다.

    차량이 하나도 안 잡히면(검출 실패) 모두 살린다. 간판·벽보 같은
    배경 오검출을 줄이기 위한 것으로, 놓치는 쪽보다 안전하다.
    """
    if not plates:
        return []
    if not vehicles:
        return list(plates)
    out = []
    for b in plates:
        cx = (float(b[0]) + float(b[2])) / 2.0
        cy = (float(b[1]) + float(b[3])) / 2.0
        keep = False
        for v in vehicles:
            vx1, vy1, vx2, vy2 = [float(x) for x in v[:4]]
            mw, mh = (vx2 - vx1) * margin, (vy2 - vy1) * margin
            if vx1 - mw <= cx <= vx2 + mw \
                    and vy1 - mh <= cy <= vy2 + mh:
                keep = True
                break
        if keep:
            out.append(b)
    return out


class _PlateTrack:
    __slots__ = ("pid", "box", "score", "misses", "hits", "vel", "age",
                 "committed_box", "pending_box", "pending_vel", "pending_age")

    def __init__(self, pid, box, score):
        self.pid = int(pid)
        self.box = [float(v) for v in box]
        self.score = float(score)
        self.misses = 0
        self.hits = 1
        self.vel = [0.0, 0.0, 0.0, 0.0]
        self.age = 0
        self.committed_box = [float(v) for v in box]
        self.pending_box = None
        self.pending_vel = [0.0, 0.0, 0.0, 0.0]
        self.pending_age = 0

    def clear_pending(self):
        self.pending_box = None
        self.pending_vel = [0.0, 0.0, 0.0, 0.0]
        self.pending_age = 0

    @staticmethod
    def union_box(a, b):
        return [min(float(a[0]), float(b[0])),
                min(float(a[1]), float(b[1])),
                max(float(a[2]), float(b[2])),
                max(float(a[3]), float(b[3]))]


class PlateTracker:
    """번호판 개별 트랙. 한 대가 1프레임 놓쳐도 다른 차 번호판은 유지한다.

    예전에는 검출 리스트를 통째로 덮어써서, 화면에 2대가 있을 때 1대가
    한 프레임만 놓치면 그 번호판이 즉시 사라져 깜빡였다. 트랙마다
    광학 흐름으로 이동시키고 max_miss까지 유지한다.
    큰 박스 변화는 얼굴과 같은 sticky 규칙으로 2프레임 확인 후 확정한다.
    """

    def __init__(self, max_miss=24, iou_thresh=0.15, max_dist=0.9,
                 sticky=True, sticky_center=0.15, sticky_scale=0.20,
                 pending_center=0.45, pending_scale=0.35):
        self.max_miss = max(1, int(max_miss))
        self.iou_thresh = float(iou_thresh)
        self.max_dist = float(max_dist)
        self.sticky = bool(sticky)
        self.sticky_center = max(0.01, float(sticky_center))
        self.sticky_scale = max(0.01, float(sticky_scale))
        self.pending_center = max(self.sticky_center, float(pending_center))
        self.pending_scale = max(self.sticky_scale, float(pending_scale))
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _pred_box(box, vel):
        return [float(box[i]) + float(vel[i]) for i in range(4)]

    def _change_ok(self, det, ref, center=None, scale=None):
        if ref is None:
            return False
        bw = max(1.0, float(ref[2]) - float(ref[0]))
        bh = max(1.0, float(ref[3]) - float(ref[1]))
        bw2 = max(1.0, float(det[2]) - float(det[0]))
        bh2 = max(1.0, float(det[3]) - float(det[1]))
        cx, cy = self._center(det)
        rx, ry = self._center(ref)
        dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 / max(bw, bh)
        c = self.sticky_center if center is None else float(center)
        s = self.sticky_scale if scale is None else float(scale)
        return (dist <= c
                and abs(bw2 / bw - 1.0) <= s
                and abs(bh2 / bh - 1.0) <= s)

    def _pending_pred(self, t):
        return self._pred_box(t.pending_box, t.pending_vel)

    def _pending_matches(self, det, t):
        if t.pending_box is None:
            return False
        return (self._change_ok(det, t.pending_box,
                                self.pending_center, self.pending_scale)
                or self._change_ok(det, self._pending_pred(t),
                                   self.pending_center, self.pending_scale)
                or _iou(det, t.pending_box) >= 0.35)

    def _commit(self, t, det, smooth=True):
        new = [float(v) for v in det]
        # 검출 지터를 조금만 섞어 타일이 떨리지 않게 한다.
        if smooth and t.hits >= 2:
            new = [0.8 * new[i] + 0.2 * t.committed_box[i]
                   for i in range(4)]
        t.vel = [0.6 * (new[i] - t.committed_box[i])
                 + 0.4 * t.vel[i] for i in range(4)]
        t.committed_box = list(new)
        t.box = list(new)
        t.misses = 0
        t.hits += 1
        t.age = 0
        t.clear_pending()

    def _set_pending(self, t, det):
        t.pending_box = [float(v) for v in det]
        t.pending_vel = [t.pending_box[i] - t.committed_box[i]
                         for i in range(4)]
        t.pending_age = 0
        t.box = t.union_box(t.committed_box, t.pending_box)
        t.misses = 0
        t.hits += 1
        t.age = 0

    def _match(self, det, track, ref=None):
        if ref is None:
            ref = track.committed_box
        iou = _iou(det, ref)
        if iou >= self.iou_thresh:
            return iou
        cx, cy = self._center(det)
        tx, ty = self._center(ref)
        size = max(1.0, det[2] - det[0], det[3] - det[1],
                   ref[2] - ref[0], ref[3] - ref[1])
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / size
        if dist <= self.max_dist:
            return 1.0 - dist
        return 0.0

    def update(self, dets, gray=None, prev_gray=None):
        dets = [d for d in (dets or []) if d and len(d) >= 4]
        used = set()
        for det in sorted(dets, key=lambda b: -(
                (b[2] - b[0]) * (b[3] - b[1]))):
            best, best_score = None, 0.0
            for t in self.tracks.values():
                if t.pid in used:
                    continue
                refs = [t.committed_box]
                if t.pending_box is not None:
                    refs.append(t.pending_box)
                for ref in refs:
                    s = self._match(det, t, ref=ref)
                    if s > best_score:
                        best, best_score = t, s
            if best is not None:
                used.add(best.pid)
                raw = [float(v) for v in det]
                expected = self._pred_box(best.committed_box, best.vel)
                if (not self.sticky
                        or self._change_ok(raw, best.committed_box)
                        or self._change_ok(raw, expected)
                        or self._pending_matches(raw, best)):
                    self._commit(
                        best, raw,
                        smooth=not self._pending_matches(raw, best))
                else:
                    self._set_pending(best, raw)
            else:
                t = _PlateTrack(self.next_id, det, 1.0)
                self.tracks[t.pid] = t
                self.next_id += 1
        from .flow import shift_box
        for pid in list(self.tracks):
            t = self.tracks[pid]
            if pid in used:
                continue
            t.misses += 1
            t.age += 1
            old = list(t.committed_box)
            moved = None
            if gray is not None and prev_gray is not None:
                try:
                    moved = shift_box(prev_gray, gray, old)
                except Exception:
                    moved = None
            if moved is None:
                decay = 0.85 ** t.misses
                moved = [old[i] + t.vel[i] * decay for i in range(4)]
            new_committed = [float(v) for v in moved]
            delta = [new_committed[i] - old[i] for i in range(4)]
            t.committed_box = new_committed
            if t.pending_box is not None:
                t.pending_box = [t.pending_box[i] + delta[i]
                                 for i in range(4)]
                t.pending_age += 1
                if t.pending_age > 2:
                    t.clear_pending()
            t.box = (t.union_box(t.committed_box, t.pending_box)
                     if t.pending_box is not None
                     else list(t.committed_box))
            if t.misses > self.max_miss:
                self.tracks.pop(pid, None)

    def boxes(self):
        return [[int(round(v)) for v in t.box]
                for t in sorted(self.tracks.values(),
                                key=lambda t: t.pid)
                if t.misses <= self.max_miss]

    def reset(self):
        self.tracks.clear()
