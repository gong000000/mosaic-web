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


def _map_box_back(box, rotation, width, height):
    """회전시켜 검출한 얼굴 박스를 원본 좌표로 되돌린다."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if rotation == 90:
        nx1, ny1 = y1, height - 1 - x2
        nx2, ny2 = y2, height - 1 - x1
        angle = getattr(box, "angle", 0.0) - 90.0
    elif rotation == 180:
        nx1, ny1 = width - 1 - x2, height - 1 - y2
        nx2, ny2 = width - 1 - x1, height - 1 - y1
        angle = getattr(box, "angle", 0.0) + 180.0
    else:  # 270 (counter-clockwise)
        nx1, ny1 = width - 1 - y2, x1
        nx2, ny2 = width - 1 - y1, x2
        angle = getattr(box, "angle", 0.0) + 90.0
    return _clip_box([nx1, ny1, nx2, ny2], width, height, angle=angle)


def _map_angle_box_back(box, matrix, width, height, rotation=0.0):
    """임의 각도로 회전한 프레임의 얼굴 박스를 원본 좌표로 되돌린다.

    박스 크기는 회전 전 얼굴 크기를 유지하고 중심만 역변환한다. 코너를
    그대로 역변환하면 45도에서 bbox가 1.4배로 커져 과마스크가 된다.
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        inv = cv2.invertAffineTransform(matrix)
        pt = cv2.transform(np.array([[[cx, cy]]], np.float32), inv)[0][0]
        w, h = max(4.0, x2 - x1), max(4.0, y2 - y1)
        angle = float(getattr(box, "angle", 0.0)) - float(rotation)
        return _clip_box([pt[0] - w / 2.0, pt[1] - h / 2.0,
                          pt[0] + w / 2.0, pt[1] + h / 2.0],
                         width, height, angle=angle)
    except Exception:
        return None


def _rotate_frame_angle(frame_bgr, angle):
    h, w = frame_bgr.shape[:2]
    c = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(c, float(angle), 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - c[0]
    m[1, 2] += nh / 2.0 - c[1]
    out = cv2.warpAffine(frame_bgr, m, (nw, nh),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, m


def _merge_face_dets(base, extra, iou=0.45):
    out = list(base or [])
    for box, score in extra or []:
        dup = False
        for i, (old, old_score) in enumerate(out):
            if _box_iou(box, old) > iou:
                if score > old_score:
                    out[i] = (box, score)
                dup = True
                break
        if not dup:
            out.append((box, score))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


_YOLO_FACE = None
_YOLO_FACE_LOCK = threading.Lock()
YOLO_FACE_MODEL = os.path.join(ROOT_DIR, "weights", "yolov11n-face.pt")


def detect_faces_yolo(frame_bgr, conf=0.25, imgsz=960):
    """YOLO 얼굴 검출기. 마스크 쓴 얼굴/누운 얼굴을 YuNet보다 잘 잡는다."""
    global _YOLO_FACE
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    try:
        with _YOLO_FACE_LOCK:
            if _YOLO_FACE is None:
                from ultralytics import YOLO
                _YOLO_FACE = YOLO(YOLO_FACE_MODEL)
            model = _YOLO_FACE
            results = list(model.predict(
                source=frame_bgr, conf=float(conf), verbose=False,
                imgsz=int(imgsz), max_det=200))
        if not results:
            return []
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
    except Exception:
        return []
    h, w = frame_bgr.shape[:2]
    out = []
    for box, score in zip(boxes, scores):
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            continue
        if not (0.35 <= bw / bh <= 2.2):
            continue
        area = bw * bh / float(max(1, w * h))
        if area > MAX_FACE_AREA:
            continue
        clipped = _clip_box([x1, y1, x2, y2], w, h, angle=0.0)
        if clipped is not None:
            out.append((clipped, float(score)))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def detect_faces_rotated(frame_bgr, base=None, rotations=(90, 270),
                         score_threshold=MIN_FACE_SCORE,
                         max_area=MAX_FACE_AREA):
    """정면 검출 + 90/270도 회전 검출을 합친다.

    YuNet은 누워 있거나 옆으로 기울어진 얼굴을 잘 못 찾는다. 프레임을
    회전시켜 한 번 더 찾고 박스를 원래 좌표로 되돌린다. 호출부에서 매
    프레임이 아니라 몇 프레임 간격으로 호출해 비용을 제한한다.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    if base is None:
        base = detect_faces(frame_bgr, score_threshold, max_area)
    rots = tuple(rotations or ())
    if not base and 180 not in rots:
        rots = tuple(dict.fromkeys(rots + (180,)))
    if not rots:
        return list(base)
    h, w = frame_bgr.shape[:2]
    extra = []
    for rot in rots:
        if rot in (90, 180, 270):
            if rot == 90:
                rframe = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                rframe = cv2.rotate(frame_bgr, cv2.ROTATE_180)
            else:
                rframe = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for box, score in detect_faces(
                    rframe, score_threshold, max_area):
                mapped = _map_box_back(box, rot, w, h)
                if mapped is not None:
                    extra.append((mapped, score))
            continue
        # 45도 같은 임의 각도: 누운 얼굴/기울어진 얼굴 검출
        rframe, m = _rotate_frame_angle(frame_bgr, rot)
        for box, score in detect_faces(
                rframe, score_threshold, max_area):
            mapped = _map_angle_box_back(box, m, w, h, rot)
            if mapped is not None:
                extra.append((mapped, score))
    return _merge_face_dets(base, extra)


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

    def _scan(crop, cx1, cy1, cx2, cy2, angles=(90, 270)):
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return []
        cw = min(720, max(32, int((cx2 - cx1) * sc)))
        ch = min(720, max(32, int((cy2 - cy1) * sc)))
        big = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
        fx = (cx2 - cx1) / float(cw)
        fy = (cy2 - cy1) / float(ch)
        found = []
        for box, score in detect_faces_rotated(
                big, score_threshold=score_threshold,
                rotations=tuple(angles)):
            x1 = cx1 + float(box[0]) * fx
            y1 = cy1 + float(box[1]) * fy
            x2 = cx1 + float(box[2]) * fx
            y2 = cy1 + float(box[3]) * fy
            clipped = _clip_box([x1, y1, x2, y2], w, h,
                                angle=getattr(box, "angle", 0.0))
            if clipped is not None:
                found.append((clipped, score))
        return found

    for pb in boxes[:max(1, int(max_crops))]:
        px1, py1, px2, py2 = pb
        bw, bh = px2 - px1, py2 - py1
        if bw < 18 or bh < 24:
            continue
        # 큰 전경 사람은 YuNet/포즈가 얼굴을 직접 잡는다. 사람 크롭 확대는
        # 옷/명치/허리 피부를 얼굴로 오검출하기 쉬우므로 건너뛴다.
        # 단, 가로로 넓은 박스(누운 사람/여러 명)는 얼굴이 위쪽에 없을 수
        # 있어 크롭 재검출을 허용한다.
        if bh > 1.5 * bw and (bh > 0.38 * h or bw > 0.35 * w):
            continue
        m = 0.10 * max(bw, bh)
        cx1 = max(0, int(px1 - m))
        cx2 = min(w, int(px2 + m))
        # 기존: 사람 박스 상단 60% 확대.
        top2 = min(h, int(py1 + 0.60 * bh + m))
        found = _scan(frame_bgr[max(0, int(py1 - m)):top2, cx1:cx2],
                      cx1, max(0, int(py1 - m)), cx2, top2)
        # 누워 있거나 옆으로 긴 사람은 머리가 박스 상단에 없을 수 있다.
        # 상단에서 못 찾았거나 박스가 가로로 길면 전체 사람 박스를 본다.
        if not found or bw > 1.3 * bh:
            full2 = min(h, int(py2 + m))
            found += _scan(frame_bgr[max(0, int(py1 - m)):full2, cx1:cx2],
                           cx1, max(0, int(py1 - m)), cx2, full2)
        # 세로로 긴(서 있는) 사람 박스의 아래 28%에서 나온 '얼굴'은
        # 신발/바닥 반사/옷자락 오검출일 가능성이 크다.
        if bh > 1.5 * bw:
            cut_y = py1 + 0.72 * bh
            found = [(bx, sc) for bx, sc in found
                     if (bx[1] + bx[3]) / 2.0 <= cut_y]
        out.extend(found)
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
    # sticky box: 큰 변화는 2프레임 확인 후 확정한다. 확인 전에는
    # committed와 pending의 합집합을 그려서 움직인 쪽도 노출하지 않는다.
    committed_box: list = None
    pending_box: list = None
    pending_angle: float = 0.0
    pending_vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    pending_hits: int = 0
    pending_age: int = 0

    def __post_init__(self):
        if not self.committed_box:
            self.committed_box = [float(v) for v in self.box]

    def clear_pending(self):
        self.pending_box = None
        self.pending_angle = 0.0
        self.pending_vel = [0.0, 0.0, 0.0, 0.0]
        self.pending_hits = 0
        self.pending_age = 0

    @staticmethod
    def union_box(a, b):
        return [min(float(a[0]), float(b[0])),
                min(float(a[1]), float(b[1])),
                max(float(a[2]), float(b[2])),
                max(float(a[3]), float(b[3]))]

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

    def __init__(self, max_miss=45, seed_miss=24, iou_thresh=0.20,
                 sticky=True, sticky_center=0.13, sticky_scale=0.15,
                 pending_center=0.40, pending_scale=0.30):
        self.max_miss = max(1, int(max_miss))
        self.seed_miss = max(1, int(seed_miss))
        self.iou_thresh = float(iou_thresh)
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

    @staticmethod
    def _smooth_size(old, new, max_ratio=0.15, min_delta=8.0,
                     max_aspect=1.7):
        """검출 박스 크기가 한 프레임에 급변하지 않게 제한한다.

        얼굴을 옷/가방으로 잘못 잡은 큰 박스가 한 프레임만 들어와도
        모자이크가 갑자기 커지는 것을 막는다. 중심은 새 검출을 따르고
        폭/높이만 최대 ±15%씩 변한다. 작은 원거리 얼굴은 15%가 몇 px에
        불과해 금방 뒤처지므로 최소 ±8px은 허용한다. 얼굴 박스가 세로로
        길어지면(몸통 오검출) 가로/세로 비율도 2.2 이내로 제한한다.
        """
        if old is None:
            return [float(v) for v in new]
        ow = max(1.0, float(old[2]) - float(old[0]))
        oh = max(1.0, float(old[3]) - float(old[1]))
        nw = max(1.0, float(new[2]) - float(new[0]))
        nh = max(1.0, float(new[3]) - float(new[1]))
        r = max(0.01, float(max_ratio))
        dw = max(ow * r, float(min_delta))
        dh = max(oh * r, float(min_delta))
        w = min(ow + dw, max(ow - dw, nw))
        h = min(oh + dh, max(oh - dh, nh))
        if h > max_aspect * w:
            h = max_aspect * w
        elif w > max_aspect * h:
            w = max_aspect * h
        cx = (float(new[0]) + float(new[2])) / 2.0
        cy = (float(new[1]) + float(new[3])) / 2.0
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def _change(self, box, ref):
        bw = max(1.0, float(ref[2]) - float(ref[0]))
        bh = max(1.0, float(ref[3]) - float(ref[1]))
        bw2 = max(1.0, float(box[2]) - float(box[0]))
        bh2 = max(1.0, float(box[3]) - float(box[1]))
        cx, cy = self._center(box)
        rx, ry = self._center(ref)
        dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 / max(bw, bh)
        return dist, bw2 / bw, bh2 / bh

    def _change_ok(self, box, ref, center=None, scale=None):
        if ref is None:
            return False
        dist, wr, hr = self._change(box, ref)
        c = self.sticky_center if center is None else float(center)
        s = self.sticky_scale if scale is None else float(scale)
        return (dist <= c
                and abs(wr - 1.0) <= s
                and abs(hr - 1.0) <= s)

    def _is_small_change(self, box, ref):
        return self._change_ok(box, ref)

    def _pending_pred(self, track):
        return self._pred_box(track.pending_box, track.pending_vel)

    def _pending_matches(self, box, track):
        if track.pending_box is None:
            return False
        return (self._change_ok(box, track.pending_box,
                                self.pending_center, self.pending_scale)
                or self._change_ok(box, self._pending_pred(track),
                                   self.pending_center, self.pending_scale)
                or _box_iou(box, track.pending_box) >= 0.35)

    def _set_pending(self, track, box, angle, score):
        track.pending_box = [float(v) for v in box]
        track.pending_angle = float(angle)
        track.pending_vel = [track.pending_box[i] - track.committed_box[i]
                             for i in range(4)]
        track.pending_hits = 1
        track.pending_age = 0
        # 확인 전 프레임은 기존 위치와 새 검출의 중간을 덮되, 박스 크기는
        # 한 프레임에 20%까지만 키운다. 큰 오검출이 들어와도 모자이크가
        # 갑자기 옷/가방 크기로 커졌다가 풀리는 현상을 막는다.
        union = track.union_box(track.committed_box, track.pending_box)
        track.box = self._smooth_size(track.committed_box, union,
                                      max_ratio=0.20)
        track.misses = 0
        track.hits += 1
        track.score = float(score)

    def _commit(self, track, box, angle, score):
        box = self._smooth_size(track.committed_box, box)
        raw_vel = [float(box[i]) - float(track.committed_box[i])
                   for i in range(4)]
        track.vel = [0.6 * raw_vel[i] + 0.4 * track.vel[i]
                     for i in range(4)]
        track.committed_box = [float(v) for v in box]
        track.box = list(track.committed_box)
        track.angle = float(angle)
        track.score = float(score)
        track.misses = 0
        track.hits += 1
        track.seeded = False
        track.clear_pending()

    def _match_score(self, box, track, ref=None):
        if ref is None:
            ref = track.committed_box
        iou = _box_iou(box, ref)
        if iou >= self.iou_thresh:
            return iou
        cx, cy = self._center(box)
        tx, ty = self._center(ref)
        bw = max(1.0, box[2] - box[0], box[3] - box[1])
        tw = max(1.0, ref[2] - ref[0], ref[3] - ref[1])
        scale = max(bw, tw)
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / scale
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])
                      / max(1.0, (ref[2] - ref[0])
                            * (ref[3] - ref[1])))
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
        try:
            bw = abs(float(box[2]) - float(box[0]))
            bh = abs(float(box[3]) - float(box[1]))
        except (TypeError, ValueError, IndexError):
            return None
        if bw < 8 or bh < 8 or not (0.45 <= bw / bh <= 1.55):
            return None
        ang = float(angle or getattr(box, "angle", 0.0))
        best, best_score = None, 0.0
        for track in self.tracks.values():
            score_match = self._match_score(box, track)
            if score_match > best_score:
                best, best_score = track, score_match
        if best is not None:
            ang = float(ang or best.angle)
            seeded = self._smooth_size(best.committed_box, box)
            best.committed_box = [float(v) for v in seeded]
            best.box = FaceBox(best.committed_box, angle=ang)
            best.score = max(best.score, float(score))
            best.misses = 0
            best.angle = ang
            best.seeded = False
            best.clear_pending()
            return best
        return self._new_track(box, score, seeded=True, angle=ang)

    def reset(self):
        """장면 전환에서 이전 장면의 얼굴 트랙을 모두 버린다."""
        self.tracks.clear()

    def update(self, detections, alpha=1.0, flow_fn=None):
        """검출 결과를 트랙에 반영한다. detections=[(xyxy, score[, angle])].

        검출이 있으면 중심은 그 위치를 따른다(지연 0). 박스 크기만 한
        프레임에 ±15% 넘게 변하지 않도록 제한해, 옷/가방을 얼굴로 잘못
        잡은 큰 박스가 갑자기 모자이크를 키우는 것을 막는다. 박스 안정화는
        1패스 후처리(앞뒤 프레임 합집합)가 담당하고, 여기서는 검출이 끊긴
        프레임만 속도/광학 흐름으로 이어 붙인다.
        """
        used = set()
        dets = sorted(detections or [],
                      key=lambda item: float(item[1]), reverse=True)
        for det in dets:
            box, score = det[0], det[1]
            try:
                bw = abs(float(box[2]) - float(box[0]))
                bh = abs(float(box[3]) - float(box[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if bw < 8 or bh < 8 or not (0.45 <= bw / bh <= 1.55):
                continue
            angle = (float(det[2]) if len(det) > 2
                     else float(getattr(box, "angle", 0.0)))
            best, best_score = None, 0.0
            for track in self.tracks.values():
                if track.tid in used:
                    continue
                refs = [track.committed_box]
                if track.pending_box is not None:
                    refs.append(track.pending_box)
                for ref in refs:
                    match = self._match_score(box, track, ref=ref)
                    if match > best_score:
                        best, best_score = track, match
            if best is not None:
                used.add(best.tid)
                base = best.committed_box
                pred = ([base[i] + best.vel[i] * 0.5 for i in range(4)]
                        if best.misses == 0 else list(base))
                blend = [alpha * float(box[i]) + (1.0 - alpha) * pred[i]
                         for i in range(4)]
                new_ang = _blend_angle(best.angle, angle)
                new_box = (_clip_box(blend, 10 ** 9, 10 ** 9, angle=new_ang)
                           or FaceBox(box, angle=new_ang))
                raw = [float(v) for v in box]
                expected = self._pred_box(base, best.vel)
                if (not self.sticky
                        or self._is_small_change(raw, base)
                        or self._is_small_change(raw, expected)
                        or self._pending_matches(raw, best)):
                    self._commit(best, new_box, new_ang, score)
                else:
                    # 큰 변화는 후보로만 저장한다. 다음 프레임에 같은
                    # 변화가 확인되면 확정하고, 아니면 버린다. 확인 전
                    # 프레임은 committed와 pending의 합집합으로 가린다.
                    self._set_pending(best, raw, new_ang, score)
            else:
                self._new_track(box, score, angle=angle)

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
                moved = [old[i] + track.vel[i] * decay
                         for i in range(4)]
            new_committed = (_clip_box(moved, 10 ** 9, 10 ** 9,
                                       angle=track.angle) or old)
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
                union = track.union_box(track.committed_box,
                                        track.pending_box)
                track.box = self._smooth_size(track.committed_box, union,
                                              max_ratio=0.20)
            else:
                track.box = list(track.committed_box)
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
