"""개인정보 보호용 YOLO-World 개방형 검출.
얼굴, 차량번호판, 주소(간판·문서), 차량동승자 등을 탐지한다.

- 로컬 실행. 외부 전송 없음.
- 사람 마스크는 seg 모델이 맡고, 월드는 문구에 해당하는 박스만 찾는다.
- 키프레임·첫 화면 1회에만 ground()를 호출한다.
- 가중치 로드 실패 시 available()=False를 반환하고 폴백한다.
"""
import os

WORLD_WEIGHTS = os.path.join("weights", "yolov8s-world.pt")
WORLD_IMGSZ = 640
WORLD_CONF = 0.20

# 개인정보 타겟 → 월드 문구
PRIVACY_PHRASES = {
    "face": ["face", "person face", "human face"],
    "license_plate": ["license plate", "car plate", "number plate"],
    "address": ["address plate", "signboard", "door plate", "street sign"],
    "passenger": ["passenger", "person in car", "person inside vehicle"],
}

# 한국어 조각 → 월드 문구
OPEN_KEYWORDS = [
    ("얼굴", ["face", "person face"]),
    ("차량번호", ["license plate", "car plate"]),
    ("번호판", ["license plate", "number plate"]),
    ("주소", ["address", "signboard"]),
    ("간판", ["signboard"]),
    ("동승자", ["passenger"]),
    ("탑승자", ["passenger"]),
]


def _norm(s):
    return (s or "").replace(" ", "").replace(" ", "")


def open_phrases_for_raw(raw):
    """원문에서 열린 문구 추출. [(한국어, [영문...])]."""
    ntext = _norm(raw).replace("모자이크", "")
    out, seen = [], set()
    for ko, ens in sorted(OPEN_KEYWORDS, key=lambda x: -len(x[0])):
        if ko and _norm(ko) in ntext:
            for e in ens:
                if e not in seen:
                    seen.add(e)
                    out.append(e)
    return out


def needs_world(query_raw, targets=None):
    """자동 모드에서 월드를 쓸지."""
    return len(open_phrases_for_raw(query_raw)) > 0


def phrases_for_query(query_raw, targets=None, world_mode="auto"):
    """월드 set_classes에 넣을 문구 목록. 최대 8개."""
    ops = open_phrases_for_raw(query_raw)
    if world_mode == "world":
        extra = []
        for t in targets or []:
            for e in PRIVACY_PHRASES.get(t, []):
                if e not in ops and e not in extra:
                    extra.append(e)
        return (ops + extra)[:8]
    return ops[:8]


class WorldStore:
    """YOLO-World 보관함. 한 번만 로드."""

    def __init__(self, weights=WORLD_WEIGHTS):
        self.weights = weights
        self._model = None
        self._ok = None

    def available(self):
        if self._ok is not None:
            return self._ok
        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(self.weights)
            self._ok = True
        except Exception:
            self._model = None
            self._ok = False
        return self._ok

    def ground(self, frame_bgr, phrases, conf=WORLD_CONF, imgsz=WORLD_IMGSZ):
        """returns [{xyxy:[x1,y1,x2,y2], conf, phrase}]."""
        if not phrases or not self.available():
            return []
        try:
            self._model.set_classes(list(phrases))
            r = list(self._model.predict(source=frame_bgr, conf=conf,
                                         verbose=False, imgsz=imgsz))[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []
            xy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            names = r.names or {}
            out = []
            for i in range(len(xy)):
                x1, y1, x2, y2 = (int(v) for v in xy[i])
                out.append({"xyxy": [x1, y1, x2, y2],
                            "conf": round(float(cf[i]), 3),
                            "phrase": str(names.get(int(cl[i]), int(cl[i])))})
            return out
        except Exception:
            return []


_WORLD = None


def get_world():
    global _WORLD
    if _WORLD is None:
        _WORLD = WorldStore()
    return _WORLD


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    ba = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
    return inter / max(1.0, float(aa + ba - inter))


def match_box(person_xyxy, world_boxes, iou_thresh=0.12, expand=0.03):
    """사람 박스와 월드 박스 묶기.
    returns (hit, best_box|None, best_iou)."""
    px1, py1, px2, py2 = person_xyxy
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    ex1, ey1 = px1 - pw * expand, py1 - ph * expand
    ex2, ey2 = px2 + pw * expand, py2 + ph * expand
    best, best_iou = None, 0.0
    for wb in world_boxes or []:
        x1, y1, x2, y2 = wb["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        iou = _iou(person_xyxy, wb["xyxy"])
        inside = (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)
        if inside or iou >= iou_thresh:
            if wb["conf"] >= (best["conf"] if best else -1.0):
                best, best_iou = wb, iou
    if best is None:
        return False, None, 0.0
    return True, best, round(float(best_iou), 3)


FACE_ITEMS = {"face", "person face", "human face"}
