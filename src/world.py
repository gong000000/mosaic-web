"""YOLO-World 개방형 검출. 색·피부로 구분되지 않는 대상(안경·우산·안전모·모자)을 맡는다.

- 로컬 실행. 외부 전송 없음.
- 사람 마스크는 seg 모델이 맡고, 월드는 문구에 해당하는 박스만 찾는다.
- 키프레임·첫 화면 1회에만 ground()를 호출한다.
  파이프라인 키프레임 stride, 스트리밍 world_stride로 호출 빈도를 조절한다.
- 가중치 로드 실패 시 available()=False를 반환하고 휴리스틱으로 폴백한다.
"""
import os

WORLD_WEIGHTS = os.path.join("weights", "yolov8s-world.pt")
WORLD_IMGSZ = 640
WORLD_CONF = 0.20

# 색 타겟 → 월드 문구. world 모드 강제 때만 사용한다. 자동 모드 색상은 HSV가 맡는다.
COLOR_PHRASES = {
    "green shirt": ["green shirt"],
    "red shirt": ["red shirt"],
    "blue shirt": ["blue shirt"],
    "black shirt": ["black shirt"],
    "white sleeveless vest": ["white sleeveless vest", "white vest"],
    "white blazer": ["white blazer"],
    "beige plaid shirt": ["plaid shirt"],
    "striped sweater": ["striped sweater"],
    "dark navy uniform": ["navy uniform"],
}

# 한국어 조각 → 월드 문구. 긴 것부터 먼저 찾는다.
OPEN_KEYWORDS = [
    ("선글라스", ["sunglasses"]),
    ("안전모", ["safety helmet", "hard hat", "helmet"]),
    ("헬멧", ["safety helmet", "hard hat", "helmet"]),
    ("빨간모자", ["red helmet", "safety helmet"]),
    ("우산", ["umbrella"]),
    ("양산", ["umbrella"]),
    ("안경", ["glasses", "sunglasses"]),
    ("모자", ["hat", "cap"]),
    ("볼캡", ["cap", "hat"]),
    ("비니", ["beanie", "hat"]),
    ("마스크", ["face mask"]),
    ("목도리", ["scarf"]),
    ("스카프", ["scarf"]),
    ("장갑", ["gloves"]),
    ("백팩", ["backpack"]),
    ("배낭", ["backpack"]),
    ("가방", ["bag", "handbag"]),
    ("핸드백", ["handbag", "bag"]),
    ("넥타이", ["tie", "necktie"]),
    ("헤드폰", ["headphones"]),
    ("이어폰", ["earphones", "headphones"]),
]


def _norm(s):
    return (s or "").replace(" ", "").replace(" ", "")


def open_phrases_for_raw(raw):
    """원문에서 열린 문구 추출. [(한국어, [영문...])]. 순서 보장, 중복 제거."""
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
    """자동 모드에서 월드를 쓸지. 열린 말이 있으면 True. 색·피부만이면 False."""
    return len(open_phrases_for_raw(query_raw)) > 0


GREEN_KW = ("초록", "그린", "녹색", "연두", "청록")


def is_fallback_targets(query_raw, targets):
    """색 매칭 없이 초록 기본값만 돌아온 폴백인지."""
    if list(targets or []) != ["green shirt"]:
        return False
    raw = (query_raw or "").replace(" ", "")
    return not any(k in raw for k in GREEN_KW)


def phrases_for_query(query_raw, targets=None, world_mode="auto"):
    """월드 set_classes에 넣을 문구 목록. 최대 8개."""
    ops = open_phrases_for_raw(query_raw)
    if world_mode == "world":
        extra = []
        for t in targets or []:
            for e in COLOR_PHRASES.get(t, []):
                if e not in ops and e not in extra:
                    extra.append(e)
        return (ops + extra)[:8]
    return ops[:8]


class WorldStore:
    """YOLO-World 보관함. 한 번만 로드. 실패하면 available False."""

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
        """returns [{xyxy:[x1,y1,x2,y2], conf, phrase}]. 빈 입력이면 []."""
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


def match_person(person_xyxy, world_boxes, iou_thresh=0.12, expand=0.03,
                 face_xyxy=None, face_only=False):
    """사람 박스와 월드 박스 묶기. 작은 물건(안경)은 얼굴에서만 찾는다.
    returns (hit, best_box|None, best_iou)."""
    px1, py1, px2, py2 = person_xyxy
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    ex1, ey1 = px1 - pw * expand, py1 - ph * expand
    ex2, ey2 = px2 + pw * expand, py2 + ph * expand
    if face_only and face_xyxy is not None:
        fx1, fy1, fx2, fy2 = face_xyxy
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        ex1, ey1 = fx1 - fw * 0.02, fy1 - fh * 0.02
        ex2, ey2 = fx2 + fw * 0.02, fy2 + fh * 0.02
        box = [fx1, fy1, fx2, fy2]
        thresh = 0.10
    else:
        box = person_xyxy
        thresh = iou_thresh
    best, best_iou = None, 0.0
    for wb in world_boxes or []:
        x1, y1, x2, y2 = wb["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        iou = _iou(box, wb["xyxy"])
        inside = (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)
        if inside or iou >= thresh:
            if wb["conf"] >= (best["conf"] if best else -1.0):
                best, best_iou = wb, iou
    if best is None:
        return False, None, 0.0
    return True, best, round(float(best_iou), 3)


# 얼굴에서 찾는 소품. 몸통 매칭을 쓰면 근처 사람까지 딸려온다.
FACE_ITEMS = {"glasses", "sunglasses", "face mask"}
