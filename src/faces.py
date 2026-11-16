"""로컬 얼굴 검출/추적.

VLM은 개인정보 항목과 대략적인 위치를 찾고, 실제 '얼굴만' 모자이크는
OpenCV YuNet으로 프레임마다 얼굴을 정확히 찾아 그린다. 얼굴이 잠깐
가려지거나 검출이 끊겨도 짧게 추적을 유지해 모자이크가 깜빡이지 않게 한다.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import ROOT_DIR

FACE_MODEL = os.path.join(ROOT_DIR, "weights",
                          "face_detection_yunet_2023mar.onnx")

# 프라이버시 우선: 클로즈업 얼굴도 가려야 하므로 상한을 크게 잡는다.
# 오검출(손/상체/배경)은 랜드마크 기하 검증으로 걸러낸다.
MAX_FACE_AREA = 0.50
# 큰 박스는 훨씬 높은 점수를 요구한다. 개/고양이 얼굴이나 몸통 오검출은
# 0.65~0.72에 몰려 있고, 실제 사람 얼굴은 0.85 이상인 경우가 대부분이다.
LARGE_AREA = 0.10
LARGE_AREA_MIN_SCORE = 0.80
MIN_FACE_SCORE = 0.55
# 화면에서 너무 작은 검출은 배경 무늬 오검출일 가능성이 높다.
MIN_FACE_SIDE_RATIO = 0.012
MIN_FACE_SIDE_PX = 12


class FaceBox(list):
    """[x1, y1, x2, y2] 리스트이면서 눈 기울기(angle) 속성을 함께 담는다."""

    def __init__(self, iterable=(), angle=0.0):
        super().__init__(iterable)
        self.angle = float(angle or 0.0)


def _clip_box(box, width, height, angle=None):
    if angle is None:
        angle = getattr(box, "angle", 0.0)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(width), x2), min(int(height), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return FaceBox([x1, y1, x2, y2], angle=angle)


def expand_face_box(box, width, height, side=0.05, top=0.03,
                    bottom=0.10, motion=0.0, angle=None):
    """YuNet/VLM의 타이트한 얼굴 박스를 입·턱까지 덮도록 조금 넓힌다.

    예전에는 이동 속도에 비례해 최대 35%까지 넓혀, 빠르게 지나가는
    얼굴이 거대한 덩어리로 보였다. 이제는 이동량을 최대 5%로 제한하고
    좌우/위 여백도 얼굴 크기에 맞게 작게 유지한다.
    """
    if angle is None:
        angle = getattr(box, "angle", 0.0)
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    tilt = min(1.0, abs(float(angle or 0.0)) / 45.0)
    side = float(side) + 0.02 * tilt
    top = float(top) + 0.02 * tilt
    bottom = float(bottom) + 0.03 * tilt
    m = min(max(0.0, float(motion)) * 0.15, 0.05 * max(w, h))
    out = [x1 - w * side - m, y1 - h * top - m,
           x2 + w * side + m, y2 + h * bottom + m]
    return _clip_box(out, width, height, angle=angle)


def _face_geometry_ok(x, y, fw, fh, row):
    """YuNet 랜드마크가 '얼굴'로 설명되는 배치인지 확인한다.

    손/팔/배경 무늬를 얼굴로 잘못 잡으면 눈·코·입 위치가 얼굴 비율을
    크게 벗어난다. 이런 박스를 버려 얼굴이 아닌 곳의 모자이크를 막는다.
    """
    if row is None or len(row) < 14:
        return False
    try:
        rex, rey = float(row[4]), float(row[5])
        lex, ley = float(row[6]), float(row[7])
        nx, ny = float(row[8]), float(row[9])
        rmx, rmy = float(row[10]), float(row[11])
        lmx, lmy = float(row[12]), float(row[13])
    except (TypeError, ValueError):
        return False
    w, h = max(1.0, float(fw)), max(1.0, float(fh))
    d = math.hypot(lex - rex, ley - rey)
    if d < 3.0:
        return False
    # 정면은 w/d 1.2~3.1, 옆모습(반측면)은 두 눈이 겹쳐 보여 3.4~4.3까지
    # 나온다. 점수가 높은 옆모습을 버리면 그 사람 얼굴이 통째로 노출되므로
    # 상한을 넓힌다. (실측: face.mp4 오른쪽 남성 w/d=3.38, h/d=4.08, 0.88)
    score = float(row[-1]) if len(row) > 14 else 1.0
    w_max = 4.30 if score >= 0.75 else 3.60
    h_max = 5.20 if score >= 0.75 else 4.60
    if not (1.05 <= w / d <= w_max):
        return False
    if not (1.45 <= h / d <= h_max):
        return False
    ey = (rey + ley) / 2.0
    if not (y + 0.08 * h <= ey <= y + 0.70 * h):
        return False
    # 눈 → 코 → 입 순서. 옆모습은 코가 눈 옆으로 크게 벗어난다.
    if ny <= ey + 0.05 * d:
        return False
    mouth_y = max(rmy, lmy)
    if mouth_y <= ny + 0.02 * d:
        return False
    eye_mouth = mouth_y - ey
    if not (0.30 * d <= eye_mouth <= 1.80 * d):
        return False
    # 두 눈의 높이 차이가 너무 크면 얼굴이 아니다(기울어진 얼굴은 허용).
    if abs(ley - rey) > 0.8 * d:
        return False
    if not (min(rex, lex) - 1.1 * d <= nx <= max(rex, lex) + 1.1 * d):
        return False
    # 랜드마크가 박스 안(약간의 여유 포함)에 있어야 한다.
    tol_x, tol_y = 0.18 * w, 0.18 * h
    for px, py in ((rex, rey), (lex, ley), (nx, ny),
                   (rmx, rmy), (lmx, lmy)):
        if not (x - tol_x <= px <= x + w + tol_x):
            return False
        if not (y - tol_y <= py <= y + h + tol_y):
            return False
    return True


def _eye_angle(row):
    """YuNet 랜드마크(오른눈/왼눈)로 얼굴 기울기(도)를 구한다."""
    if row is None or len(row) < 8:
        return 0.0
    try:
        rex, rey = float(row[4]), float(row[5])
        lex, ley = float(row[6]), float(row[7])
    except (TypeError, ValueError):
        return 0.0
    if abs(lex - rex) < 1e-3 and abs(ley - rey) < 1e-3:
        return 0.0
    ang = math.degrees(math.atan2(ley - rey, lex - rex))
    if ang > 90.0:
        ang -= 180.0
    elif ang < -90.0:
        ang += 180.0
    return float(ang)


def _blend_angle(old, new, alpha=0.45):
    """-180~180 경계를 넘어도 짧은 방향으로 각도를 섞는다."""
    delta = (float(new) - float(old) + 180.0) % 360.0 - 180.0
    return float(old) + float(alpha) * delta


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
    """YuNet 얼굴 검출기.

    다중 스케일 검출에서 매번 setInputSize를 부르면 OpenCV DNN 내부 버퍼가
    매 프레임 3번씩 재할당되어 속도가 20~30배 느려진다. 해상도별 검출기
    인스턴스를 캐시해 재할당을 막는다.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path=FACE_MODEL):
        self.model_path = model_path
        self._detectors = {}
        self._detect_lock = threading.Lock()
        self._call_count = 0

    @classmethod
    def shared(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure(self, width, height):
        if not os.path.exists(self.model_path):
            return None
        if cv2.getNumThreads() <= 1:
            try:
                cv2.setNumThreads(min(8, max(2, os.cpu_count() or 4)))
            except Exception:
                pass
        key = (int(width), int(height))
        det = self._detectors.get(key)
        if det is None:
            if len(self._detectors) >= 12:
                self._detectors.clear()
            det = cv2.FaceDetectorYN.create(
                self.model_path, "", key, 0.50, 0.3, 5000)
            self._detectors[key] = det
        return det

    def _detect_once(self, frame_bgr, score_threshold=MIN_FACE_SCORE,
                     max_area=MAX_FACE_AREA):
        """한 해상도에서 검출. returns [(FaceBox, score)]."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        if w < 16 or h < 16:
            return []
        try:
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
        min_side = max(MIN_FACE_SIDE_PX,
                       int(round(MIN_FACE_SIDE_RATIO * min(w, h))))
        for row in faces:
            if len(row) < 5:
                continue
            x, y, fw, fh = [float(v) for v in row[:4]]
            score = float(row[-1])
            if score < float(score_threshold):
                continue
            if fw < min_side or fh < min_side:
                continue
            if not _face_geometry_ok(x, y, fw, fh, row):
                continue
            ang = _eye_angle(row)
            box = _clip_box([x, y, x + fw, y + fh], w, h, angle=ang)
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

    def detect(self, frame_bgr, score_threshold=MIN_FACE_SCORE,
               max_area=MAX_FACE_AREA, upscale=None):
        """다중 스케일 + 가장자리 패딩 검출. returns [(FaceBox, score)].

        - 화면 가득 찬 근접 얼굴: YuNet 은 큰 얼굴에 약하므로 축소본에서도 찾는다.
        - 화면 가장자리에 걸친 얼굴: 테두리를 덧대 잘린 얼굴도 찾는다.
        - 멀리 있는 작은 얼굴: MOSAIC_FACE_UPSCALE 로 확대 검출을 켤 수 있다.
          upscale 인자로 직접 넘기면 환경변수보다 우선한다.
        환경변수 MOSAIC_FACE_MULTISCALE=0 이면 기존 단일 스케일만 쓴다.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        mode = os.environ.get("MOSAIC_FACE_MULTISCALE", "1")
        if mode == "0":
            return self._detect_once(frame_bgr, score_threshold, max_area)
        h, w = frame_bgr.shape[:2]
        if w < 16 or h < 16:
            return []
        self._call_count += 1
        base_dets = self._detect_once(frame_bgr, score_threshold, max_area)
        # 기본 해상도에서 이미 얼굴이 잡혔으면 4프레임마다(또는 얼굴이 하나도
        # 안 잡혔을 때 즉시) 패딩+축소 스케일을 추가로 돌아 클로즈업/가장자리
        # 얼굴을 보강한다. 추적기와 광학 흐름이 사이 프레임을 메운다.
        if base_dets and mode != "all" and (self._call_count % 4 != 1):
            return base_dets
        pad = int(round(0.08 * min(w, h)))
        padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_REPLICATE)
        ph, pw = padded.shape[:2]
        scales = [0.5, 0.25] if base_dets else [1.0, 0.5, 0.25]
        if upscale is None:
            try:
                up = float(os.environ.get("MOSAIC_FACE_UPSCALE", "0") or 0)
            except ValueError:
                up = 0.0
        else:
            try:
                up = float(upscale)
            except (TypeError, ValueError):
                up = 0.0
        if up > 1.0 and max(w, h) <= 1280:
            scales.append(up)
        cands = [(bx, sc) for bx, sc in base_dets]
        for sc in scales:
            if sc == 1.0:
                img = padded
                thr = float(score_threshold)
            else:
                img = cv2.resize(
                    padded, (max(16, int(pw * sc)), max(16, int(ph * sc))),
                    interpolation=(cv2.INTER_AREA if sc < 1.0
                                   else cv2.INTER_LINEAR))
                thr = (max(float(score_threshold), 0.62) if sc == 0.5
                       else max(float(score_threshold), 0.68) if sc < 0.5
                       else float(score_threshold))
            for box, score in self._detect_once(img, thr, max_area=1.0):
                ang = getattr(box, "angle", 0.0)
                bx = FaceBox([v / sc - pad for v in box], angle=ang)
                cands.append((bx, score))
        merged = []
        for bx, score in sorted(cands, key=lambda c: c[1], reverse=True):
            if any(_box_iou(bx, m[0]) > 0.35 for m in merged):
                continue
            merged.append((bx, score))
        out = []
        for bx, score in merged:
            box = _clip_box(bx, w, h, angle=getattr(bx, "angle", 0.0))
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
                 max_area=MAX_FACE_AREA, upscale=None):
    return YuNetFaceDetector.shared().detect(
        frame_bgr, score_threshold, max_area, upscale=upscale)


def detect_faces_in_person_crops(frame_bgr, persons, scale=2.0,
                                 max_crops=8,
                                 score_threshold=MIN_FACE_SCORE):
    """사람 박스 상단(머리 영역)을 확대해 작은 얼굴을 찾는다.

    1080p에서 20px 미만 얼굴은 전체 프레임에서 YuNet이 못 잡는다.
    사람이 잡혔는데 그 사람 안에 얼굴이 없을 때, 사람 박스 위쪽 55%만
    잘라 scale배 확대해 검출한다. 배경 얼굴을 끌어오지 않도록 상단만 본다.
    returns [(FaceBox, score)].
    """
    if frame_bgr is None or frame_bgr.size == 0 or not persons:
        return []
    h, w = frame_bgr.shape[:2]
    try:
        boxes = sorted(([float(v) for v in p[:4]] for p in persons),
                       key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    except (TypeError, ValueError):
        return []
    try:
        sc = min(3.0, max(1.25, float(scale)))
    except (TypeError, ValueError):
        sc = 2.0
    out = []
    for pb in boxes[:max(1, int(max_crops))]:
        px1, py1, px2, py2 = pb
        bw, bh = px2 - px1, py2 - py1
        if bw < 18 or bh < 24:
            continue
        m = 0.10 * max(bw, bh)
        cx1 = max(0, int(px1 - m))
        cy1 = max(0, int(py1 - m))
        cx2 = min(w, int(px2 + m))
        cy2 = min(h, int(py1 + 0.60 * bh + m))
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
        cw = min(720, max(32, int((cx2 - cx1) * sc)))
        ch = min(720, max(32, int((cy2 - cy1) * sc)))
        big = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
        fx = (cx2 - cx1) / float(cw)
        fy = (cy2 - cy1) / float(ch)
        for box, score in detect_faces(big, score_threshold):
            x1 = cx1 + float(box[0]) * fx
            y1 = cy1 + float(box[1]) * fy
            x2 = cx1 + float(box[2]) * fx
            y2 = cy1 + float(box[3]) * fy
            clipped = _clip_box([x1, y1, x2, y2], w, h,
                                angle=getattr(box, "angle", 0.0))
            if clipped is None:
                continue
            out.append((clipped, score))
    out.sort(key=lambda item: item[1], reverse=True)
    merged = []
    for bx, score in out:
        if any(_box_iou(bx, m[0]) > 0.35 for m in merged):
            continue
        merged.append((bx, score))
    return merged


def persons_without_faces(persons, faces, margin=0.15):
    """얼굴 중심이 사람 박스 안에 없는 사람만 돌려준다(원거리 보완 대상)."""
    if not persons:
        return []
    out = []
    for p in persons:
        px1, py1, px2, py2 = [float(v) for v in p[:4]]
        mw, mh = (px2 - px1) * margin, (py2 - py1) * margin
        has = False
        for box, _score in faces or []:
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            if (px1 - mw <= cx <= px2 + mw
                    and py1 - mh <= cy <= py2 + mh):
                has = True
                break
        if not has:
            out.append(p)
    return out


def filter_faces_by_person(faces, persons, min_score=0.82, margin=0.15,
                           frame_area=None, gate_area_ratio=0.0005):
    """사람 박스와 무관한 '큰 동물 얼굴' 오검출을 거른다.

    YuNet은 사람 얼굴 검출기지만 개/고양이 얼굴에도 0.6~0.7 점수를 낸다.
    점수가 아주 높은 사람 얼굴과 아주 작은 원거리 얼굴은 그대로 두고,
    애매한 크기의 검출만 YOLO 사람 박스 안에 중심이 있는지 확인한다.
    사람 박스가 하나도 없으면(검출 실패) 얼굴을 버리지 않는다.
    """
    if persons is None:
        return faces
    try:
        if len(persons) == 0:
            return faces
    except TypeError:
        return faces
    out = []
    for box, score in faces:
        if float(score) >= float(min_score):
            out.append((box, score))
            continue
        fw = max(0.0, float(box[2]) - float(box[0]))
        fh = max(0.0, float(box[3]) - float(box[1]))
        if frame_area and (fw * fh) < gate_area_ratio * float(frame_area):
            out.append((box, score))
            continue
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        for p in persons:
            px1, py1, px2, py2 = [float(v) for v in p[:4]]
            if px2 - px1 < 8 or py2 - py1 < 16:
                continue
            mw, mh = (px2 - px1) * margin, (py2 - py1) * margin
            if (px1 - mw <= cx <= px2 + mw
                    and py1 - mh <= cy <= py2 + mh):
                out.append((box, score))
                break
    return out


def scene_signature(frame_bgr):
    """장면 전환 비교용 저해상도 흑백 서명."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def is_scene_cut(prev_sig, frame_bgr, threshold=45.0):
    """이전 프레임과 크게 달라졌으면 장면 전환으로 본다."""
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
    angle: float = 0.0
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def speed(self):
        """박스 중심의 프레임당 이동 속도(px)."""
        return float(max(abs(self.vel[0] + self.vel[2]),
                         abs(self.vel[1] + self.vel[3])) / 2.0)

    def max_miss(self, default=45, seeded=24):
        if self.seeded:
            return seeded
        return min(default, max(6, self.hits * 3))

    def visible(self, default=45, seeded=24):
        """지금 프레임에 그릴지. 프라이버시 우선이라 첫 프레임부터 가린다."""
        if self.misses > self.max_miss(default, seeded):
            return False
        return True


class FaceTracker:
    """짧은 누락을 버티는 얼굴 트랙."""

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
        if dist <= 0.95 and 0.10 <= area_ratio <= 10.0:
            return 1.0 - dist
        return 0.0

    def _new_track(self, box, score, seeded=False, angle=0.0):
        b = FaceBox(box, angle=angle)
        track = FaceTrack(self.next_id, b, float(score),
                          seeded=bool(seeded), angle=float(angle or 0.0))
        self.next_id += 1
        self.tracks[track.tid] = track
        return track

    def seed(self, box, score=1.0, angle=0.0):
        """VLM이 찾은 얼굴 박스를 추적 초기값으로 넣는다."""
        if not box:
            return None
        ang = float(angle or getattr(box, "angle", 0.0))
        best, best_score = None, 0.0
        for track in self.tracks.values():
            score_match = self._match_score(box, track)
            if score_match > best_score:
                best, best_score = track, score_match
        if best is not None:
            best.box = FaceBox(box, angle=ang or best.angle)
            best.score = max(best.score, float(score))
            best.misses = 0
            best.angle = float(ang or best.angle)
            best.seeded = False
            return best
        return self._new_track(box, score, seeded=True, angle=ang)

    def reset(self):
        """장면 전환에서 이전 장면의 얼굴 트랙을 모두 버린다."""
        self.tracks.clear()

    def update(self, detections, alpha=1.0, flow_fn=None):
        """검출 결과를 트랙에 반영한다. detections=[(xyxy, score[, angle])].

        검출이 있으면 평활 없이 그 위치를 그대로 쓴다(지연 0). 박스 안정화는
        1패스 후처리(앞뒤 프레임 합집합)가 담당하고, 여기서는 검출이 끊긴
        프레임만 속도/광학 흐름으로 이어 붙인다.
        """
        used = set()
        dets = sorted(detections or [],
                      key=lambda item: float(item[1]), reverse=True)
        for det in dets:
            box, score = det[0], det[1]
            angle = (float(det[2]) if len(det) > 2
                     else float(getattr(box, "angle", 0.0)))
            best, best_score = None, 0.0
            for track in self.tracks.values():
                if track.tid in used:
                    continue
                match = self._match_score(box, track)
                if match > best_score:
                    best, best_score = track, match
            if best is not None:
                used.add(best.tid)
                pred = [best.box[i] + best.vel[i] * 0.5 for i in range(4)] \
                    if best.misses == 0 else list(best.box)
                blend = [alpha * float(box[i]) + (1.0 - alpha) * pred[i]
                         for i in range(4)]
                new_ang = _blend_angle(best.angle, angle)
                new_box = (_clip_box(blend, 10 ** 9, 10 ** 9, angle=new_ang)
                           or FaceBox(box, angle=new_ang))
                raw_vel = [float(new_box[i]) - float(best.box[i])
                           for i in range(4)]
                best.vel = [0.6 * raw_vel[i] + 0.4 * best.vel[i]
                            for i in range(4)]
                best.box = new_box
                best.score = float(score)
                best.misses = 0
                best.hits += 1
                best.angle = new_ang
                best.seeded = False
            else:
                self._new_track(box, score, angle=angle)

        for tid in list(self.tracks):
            if tid in used:
                continue
            track = self.tracks[tid]
            track.misses += 1
            decay = 0.8 ** track.misses
            moved = None
            if flow_fn is not None:
                try:
                    moved = [float(v) for v in flow_fn(track.box)]
                    if all(abs(moved[i] - track.box[i]) < 1e-6
                           for i in range(4)):
                        moved = None
                    else:
                        track.vel = [0.7 * (moved[i] - track.box[i])
                                     + 0.3 * track.vel[i]
                                     for i in range(4)]
                except Exception:
                    moved = None
            if moved is None:
                moved = [track.box[i] + track.vel[i] * decay
                         for i in range(4)]
            track.box = (_clip_box(moved, 10 ** 9, 10 ** 9, angle=track.angle)
                         or track.box)
            if track.misses > track.max_miss(self.max_miss,
                                             self.seed_miss):
                self.tracks.pop(tid, None)

    def active(self):
        out = []
        for track in self.tracks.values():
            if not track.visible(self.max_miss, self.seed_miss):
                continue
            tx1, ty1, tx2, ty2 = track.box
            if tx2 <= 0 or ty2 <= 0:
                continue
            track.frames += 1
            out.append(track)
        out.sort(key=lambda t: t.tid)
        return out
