"""ByteTrack adapter for the face-only and general face paths.

The rest of the pipeline only needs ``update``, ``active``, ``reset`` and
``seed``.  This adapter keeps the existing FaceTrack smoothing/fallback logic
and uses ``supervision``'s ByteTrack only for data association.  If the package
is missing, the caller falls back to the original FaceTracker.
"""
from __future__ import annotations

import logging

import numpy as np

from .faces import FaceTracker, _box_iou, _clip_box

log = logging.getLogger("mosaic.bytetrack")


def bytetrack_available():
    try:
        import supervision as sv  # noqa: F401
        return hasattr(sv, "ByteTrack")
    except Exception:
        return False


class ByteTrackFaceTracker(FaceTracker):
    """FaceTracker-compatible wrapper around supervision.ByteTrack."""

    def __init__(self, max_miss=45, seed_miss=24, iou_thresh=0.20,
                 frame_rate=30.0, track_activation_threshold=0.20,
                 lost_track_buffer=60, minimum_matching_threshold=0.70,
                 **kwargs):
        kwargs.pop("sticky", None)
        super().__init__(max_miss=max_miss, seed_miss=seed_miss,
                         iou_thresh=iou_thresh, sticky=False, **kwargs)
        self.frame_rate = max(1.0, float(frame_rate))
        self.track_activation_threshold = float(track_activation_threshold)
        self.lost_track_buffer = int(lost_track_buffer)
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self._bt = None
        self._bt_error = False
        self._bt_to_tid = {}
        self._seed_boxes = []

    def _make_bt(self):
        try:
            import supervision as sv
            return sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=int(round(self.frame_rate)))
        except Exception:
            log.exception("ByteTrack 초기화 실패 - FaceTracker로 대체")
            self._bt_error = True
            return None

    def _ensure_bt(self):
        if self._bt is None and not self._bt_error:
            self._bt = self._make_bt()
        return self._bt

    def reset(self):
        super().reset()
        self._bt_to_tid.clear()
        self._seed_boxes.clear()
        if not self._bt_error:
            self._bt = self._make_bt()

    def _map_bt_id(self, bt_id, box, score):
        tid = self._bt_to_tid.get(int(bt_id))
        if tid is not None and tid in self.tracks:
            return self.tracks[tid]
        # A seeded VLM/prior box is not in ByteTrack.  Attach the first fresh
        # detection that overlaps it, keeping the seed's ID and box history.
        best = None
        best_iou = 0.0
        for seed_track, seed_box in list(self._seed_boxes):
            if seed_track.tid in self.tracks:
                iou = _box_iou(box, seed_box)
                if iou > best_iou:
                    best, best_iou = seed_track, iou
        if best is not None and best_iou >= 0.20:
            self._seed_boxes = [
                (t, b) for t, b in self._seed_boxes if t is not best]
            self._bt_to_tid[int(bt_id)] = best.tid
            return best
        track = self._new_track(box, score)
        self._bt_to_tid[int(bt_id)] = track.tid
        return track

    def update(self, detections, alpha=1.0, flow_fn=None):
        bt = self._ensure_bt()
        dets = sorted(detections or [],
                      key=lambda item: float(item[1]), reverse=True)
        used = set()
        if bt is None:
            return super().update(detections, alpha=alpha,
                                  flow_fn=flow_fn)
        try:
            import supervision as sv
            rows = []
            for det in dets:
                box, score = det[0], det[1]
                try:
                    bw = abs(float(box[2]) - float(box[0]))
                    bh = abs(float(box[3]) - float(box[1]))
                    if bw < 8 or bh < 8 or not (0.45 <= bw / bh <= 1.55):
                        continue
                    rows.append((box, float(score)))
                except (TypeError, ValueError, IndexError):
                    continue
            if rows:
                xy = np.array([[float(v) for v in b[:4]]
                               for b, _s in rows], np.float32)
                cf = np.array([s for _b, s in rows], np.float32)
                cl = np.zeros((len(rows),), int)
                det = sv.Detections(xyxy=xy, confidence=cf, class_id=cl)
            else:
                det = sv.Detections.empty()
            out = bt.update_with_detections(det)
            if out is not None and out.tracker_id is not None:
                for b, s, btid in zip(out.xyxy, out.confidence,
                                      out.tracker_id):
                    if btid is None:
                        continue
                    box = [float(v) for v in b[:4]]
                    tr = self._map_bt_id(int(btid), box, float(s))
                    used.add(tr.tid)
                    base = tr.committed_box
                    pred = ([base[i] + tr.vel[i] * 0.5 for i in range(4)]
                            if tr.misses == 0 else list(base))
                    blend = [float(alpha) * box[i]
                             + (1.0 - float(alpha)) * pred[i]
                             for i in range(4)]
                    try:
                        new_box = self._smooth_size(base, blend)
                    except Exception:
                        new_box = box
                    # Keep the old pending/sticky protection: a one-frame
                    # false jump is committed only after a second match.
                    raw = [float(v) for v in box]
                    expected = self._pred_box(base, tr.vel)
                    if (not self.sticky
                            or self._is_small_change(raw, base)
                            or self._is_small_change(raw, expected)
                            or self._pending_matches(raw, tr)):
                        self._commit(tr, new_box, tr.angle, float(s))
                    else:
                        self._set_pending(tr, raw, tr.angle, float(s))
        except Exception:
            log.exception("ByteTrack 갱신 실패 - 기존 FaceTracker로 대체")
            self._bt_error = True
            return super().update(detections, alpha=alpha,
                                  flow_fn=flow_fn)
        for tid in list(self.tracks):
            if tid in used:
                continue
            track = self.tracks[tid]
            track.misses += 1
            decay = 0.8 ** track.misses
            old = list(track.committed_box)
            moved = None
            if flow_fn is not None:
                try:
                    moved = [float(v) for v in flow_fn(track.committed_box)]
                    if all(abs(moved[i] - old[i]) < 1e-6
                           for i in range(4)):
                        moved = None
                    else:
                        track.vel = [0.7 * (moved[i] - old[i])
                                     + 0.3 * track.vel[i]
                                     for i in range(4)]
                except Exception:
                    moved = None
            if moved is None:
                moved = [old[i] + track.vel[i] * decay for i in range(4)]
            new_committed = (_clip_box(
                moved, 10 ** 9, 10 ** 9, angle=track.angle) or old)
            delta = [float(new_committed[i]) - float(old[i])
                     for i in range(4)]
            track.committed_box = [float(v) for v in new_committed]
            if track.pending_box is not None:
                track.pending_box = [float(track.pending_box[i]) + delta[i]
                                     for i in range(4)]
                track.pending_age += 1
                if track.pending_age > 2:
                    track.clear_pending()
            if track.pending_box is not None:
                track.box = self._smooth_size(
                    track.committed_box,
                    track.union_box(track.committed_box,
                                    track.pending_box),
                    max_ratio=0.20)
            else:
                track.box = list(track.committed_box)
            if track.misses > track.max_miss(self.max_miss, self.seed_miss):
                self.tracks.pop(tid, None)
                self._bt_to_tid = {
                    k: v for k, v in self._bt_to_tid.items() if v != tid}

    def seed(self, box, score=1.0, angle=0.0):
        tr = super().seed(box, score=score, angle=angle)
        if tr is not None:
            self._seed_boxes.append((tr, [float(v) for v in box[:4]]))
            # Do not let seed boxes grow without bound.
            if len(self._seed_boxes) > 64:
                self._seed_boxes = self._seed_boxes[-64:]
        return tr


def make_face_tracker(max_miss=45, seed_miss=24, frame_rate=30.0,
                      force=None):
    """Create a ByteTrack tracker when available, else the original one."""
    use = force
    if use is None:
        import os
        use = (os.environ.get("MOSAIC_FACE_BYTETRACK", "1") != "0"
               and bytetrack_available())
    if use:
        try:
            return ByteTrackFaceTracker(
                max_miss=max_miss, seed_miss=seed_miss,
                frame_rate=frame_rate)
        except Exception:
            log.exception("ByteTrack 생성 실패 - FaceTracker 사용")
    return FaceTracker(max_miss=max_miss, seed_miss=seed_miss)
