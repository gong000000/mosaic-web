"""로컬 얼굴 검출/추적.

VLM은 개인정보 항목과 대략적인 위치를 찾고, 실제 '얼굴만' 모자이크는
OpenCV YuNet으로 프레임마다 얼굴을 정확히 찾아 그린다. 얼굴이 잠깐
가려지거나 검출이 끊겨도 짧게 추적을 유지해 모자이크가 깜빡이지 않게 한다.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ROOT_DIR

FACE_MODEL = os.path.join(ROOT_DIR, "weights",
                          "face_detection_yunet_2023mar.onnx")

# YuNet은 강아지 얼굴이나 상체 일부를 사람 얼굴로 오검출하기도 한다.
# 얼굴 박스가 화면의 16%를 넘으면 '얼굴만' 범위를 벗어난 것으로 보고
# 버린다. 다만 실제 근접 얼굴은 8~12%에 달할 수 있어, 큰 박스는
# 높은 신뢰도에서만 허용한다. 낮은 신뢰도의 큰 박스(강아지/상체 오검출)
# 는 제외한다.
MAX_FACE_AREA = 0.16
LARGE_AREA = 0.06
LARGE_AREA_MIN_SCORE = 0.80
MIN_FACE_SCORE = 0.60


def _clip_box(box, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(width), x2), min(int(height), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [x1, y1, x2, y2]


def expand_face_box(box, width, height, side=0.06, top=0.05,
                    bottom=0.15):
    """YuNet/VLM의 타이트한 얼굴 박스를 입·턱까지 덮도록 넓힌다.

    얼굴 검출기는 입꼬리 아래를 조금 남기는 경우가 많다. 좌우보다 아래를
    더 넓혀 P4처럼 입이 살짝 보이는 문제를 막는다. 단, 너무 넓히면
    목·어깨까지 타원으로 번지므로 확장량은 작게 유지한다.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    box = [x1 - w * side, y1 - h * top,
           x2 + w * side, y2 + h * bottom]
    return _clip_box(box, width, height)


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
    ba = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    return inter / float(max(1, aa + ba - inter))


class YuNetFaceDetector:
    """YuNet 얼굴 검출기. 모델은 프로세스당 한 번만 연다."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path=FACE_MODEL):
        self.model_path = model_path
        self._detector = None
        self._size = None
        self._detect_lock = threading.Lock()

    @classmethod
    def shared(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure(self, width, height):
        if not os.path.exists(self.model_path):
            return None
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path, "", (int(width), int(height)),
                0.55, 0.3, 5000)
        if self._size != (int(width), int(height)):
            self._detector.setInputSize((int(width), int(height)))
            self._size = (int(width), int(height))
        return self._detector

    def detect(self, frame_bgr, score_threshold=MIN_FACE_SCORE,
               max_area=MAX_FACE_AREA):
        """returns [(xyxy, score)]. 모델이 없거나 실패하면 빈 목록."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        if w < 16 or h < 16:
            return []
        try:
            # OpenCV DNN 검출기는 스레드 안전하지 않다. 분석 샘플을 병렬로
            # 처리해도 모델 내부 상태가 깨지지 않도록 검출만 직렬화한다.
            with self._detect_lock:
                det = self._ensure(w, h)
                if det is None:
                    return []
                try:
                    det.setScoreThreshold(float(score_threshold))
                except Exception:
                    pass
                _ok, faces = det.detect(frame_bgr)
        except Exception:
            return []
        if faces is None:
            return []
        out = []
        for row in faces:
            if len(row) < 5:
                continue
            x, y, fw, fh = [float(v) for v in row[:4]]
            score = float(row[-1])
            if score < float(score_threshold):
                continue
            box = _clip_box([x, y, x + fw, y + fh], w, h)
            if box is None:
                continue
            area = ((box[2] - box[0]) * (box[3] - box[1])
                    / float(max(1, w * h)))
            if area > float(max_area):
                continue
            if (area > min(LARGE_AREA, float(max_area))
                    and score < LARGE_AREA_MIN_SCORE):
                continue
            out.append((box, score))
        out.sort(key=lambda item: item[1], reverse=True)
        return out


def detect_faces(frame_bgr, score_threshold=MIN_FACE_SCORE,
                 max_area=MAX_FACE_AREA):
    return YuNetFaceDetector.shared().detect(
        frame_bgr, score_threshold, max_area)


def scene_signature(frame_bgr):
    """장면 전환 비교용 저해상도 흑백 서명."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def is_scene_cut(prev_sig, frame_bgr, threshold=45.0):
    """이전 프레임과 크게 달라졌으면 장면 전환으로 본다.

    장면이 바뀌면 이전 장면의 얼굴 트랙이 새 장면의 배경/상체 위에
    남아 큰 고스트 모자이크가 된다. 컷에서 트랙을 비워 막는다.
    """
    if prev_sig is None or frame_bgr is None:
        return False
    cur = scene_signature(frame_bgr)
    diff = float(np.mean(cv2.absdiff(prev_sig, cur)))
    return diff >= float(threshold)


@dataclass
class FaceTrack:
    tid: int
    box: list
    score: float = 1.0
    misses: int = 0
    hits: int = 1
    frames: int = 1
    seeded: bool = False

    def max_miss(self, default=45, seeded=24):
        """이 트랙을 몇 프레임까지 유지할지.

        한두 번 스친 오검출이 1~2초씩 화면에 남아 큰 사각형/고스트
        모자이크가 되는 것을 막는다. 여러 프레임 확정된 얼굴만 오토바이
        등에 가려진 동안 길게 유지한다.
        """
        if self.seeded:
            return seeded
        return min(default, max(6, self.hits * 3))

    def visible(self, default=45, seeded=24):
        """지금 프레임에 그릴지. 짧게 스친 저신뢰 검출은 다음 프레임부터."""
        if self.misses > self.max_miss(default, seeded):
            return False
        return bool(self.seeded or self.hits >= 2 or self.score >= 0.90)


class FaceTracker:
    """짧은 누락을 버티는 얼굴 트랙.

    YOLO 사람 트랙 ID와 무관하게 얼굴 박스만 추적한다. 오토바이나 행인이
    얼굴을 잠깐 가려 검출이 끊겨도 마지막 박스를 유지해 깜빡임을 줄인다.
    """

    def __init__(self, max_miss=45, seed_miss=24, iou_thresh=0.20):
        self.max_miss = max(1, int(max_miss))
        self.seed_miss = max(1, int(seed_miss))
        self.iou_thresh = float(iou_thresh)
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _match_score(self, box, track):
        iou = _box_iou(box, track.box)
        if iou >= self.iou_thresh:
            return iou
        cx, cy = self._center(box)
        tx, ty = self._center(track.box)
        bw = max(1.0, box[2] - box[0], box[3] - box[1])
        tw = max(1.0, track.box[2] - track.box[0],
                 track.box[3] - track.box[1])
        scale = max(bw, tw)
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / scale
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])
                      / max(1.0, (track.box[2] - track.box[0])
                            * (track.box[3] - track.box[1])))
        # VLM 얼굴 박스는 YuNet 타이트 박스보다 2~4배 클 수 있다. 중심이
        # 겹치면 같은 얼굴로 보고, 면적 차이만으로 매칭을 버리지 않는다.
        if dist <= 0.45 and 0.20 <= area_ratio <= 5.0:
            return 1.0 - dist
        return 0.0

    def _new_track(self, box, score, seeded=False):
        track = FaceTrack(self.next_id, list(box), float(score),
                          seeded=bool(seeded))
        self.next_id += 1
        self.tracks[track.tid] = track
        return track

    def seed(self, box, score=1.0):
        """VLM이 찾은 얼굴 박스를 추적 초기값으로 넣는다."""
        if not box:
            return None
        best, best_score = None, 0.0
        for track in self.tracks.values():
            score_match = self._match_score(box, track)
            if score_match > best_score:
                best, best_score = track, score_match
        if best is not None:
            best.box = list(box)
            best.score = max(best.score, float(score))
            best.misses = 0
            best.seeded = False
            return best
        return self._new_track(box, score, seeded=True)

    def reset(self):
        """장면 전환에서 이전 장면의 얼굴 트랙을 모두 버린다."""
        self.tracks.clear()

    def update(self, detections, alpha=0.65):
        """검출 결과를 트랙에 반영한다. detections=[(xyxy, score)]."""
        used = set()
        dets = sorted(detections or [],
                      key=lambda item: float(item[1]), reverse=True)
        for box, score in dets:
            best, best_score = None, 0.0
            for track in self.tracks.values():
                if track.tid in used:
                    continue
                match = self._match_score(box, track)
                if match > best_score:
                    best, best_score = track, match
            if best is not None:
                used.add(best.tid)
                blend = [alpha * float(box[i]) + (1.0 - alpha) * best.box[i]
                         for i in range(4)]
                best.box = _clip_box(blend, 10 ** 9, 10 ** 9) or list(box)
                best.score = float(score)
                best.misses = 0
                best.hits += 1
                best.seeded = False
            else:
                self._new_track(box, score)

        for tid in list(self.tracks):
            if tid in used:
                continue
            track = self.tracks[tid]
            track.misses += 1
            if track.misses > track.max_miss(self.max_miss,
                                             self.seed_miss):
                self.tracks.pop(tid, None)

    def active(self):
        out = []
        for track in self.tracks.values():
            if not track.visible(self.max_miss, self.seed_miss):
                continue
            tx1, ty1, tx2, ty2 = track.box
            # 화면 밖으로 완전히 나간 트랙은 그리지 않는다.
            if tx2 <= 0 or ty2 <= 0:
                continue
            track.frames += 1
            out.append(track)
        out.sort(key=lambda t: t.tid)
        return out
