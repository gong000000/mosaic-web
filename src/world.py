"""개인정보 보호용 YOLO-World 개방형 검출.
얼굴, 차량번호판, 주소(간판·문서), 차량동승자 등을 탐지한다.

- 로컬 실행. 외부 전송 없음.
- 사람 마스크는 seg 모델이 맡고, 월드는 문구에 해당하는 박스만 찾는다.
- VLM을 쓸 수 없을 때(키 없음/호출 실패) 폴백으로 호출된다.
- 가중치 로드 실패 시 available()=False를 반환하고 빈 결과를 준다.
"""
import os

WORLD_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "yolov8s-world.pt")
WORLD_IMGSZ = 640
WORLD_CONF = 0.20

# 개인정보 타겟 → 월드 문구
PRIVACY_PHRASES = {
    "face": ["face", "person face", "human face"],
    "license_plate": ["license plate", "car plate", "number plate"],
    "address": ["address plate", "signboard", "door plate", "street sign"],
    "passenger": ["passenger", "person in car", "person inside vehicle"],
    "name_tag": ["name tag", "badge", "employee badge"],
    "id_card": ["id card", "identity card", "document"],
    "tattoo": ["tattoo"],
    "phone_screen": ["cell phone screen", "smartphone screen"],
}

_PHRASE_TO_CAT = {}
for _cat, _phrases in PRIVACY_PHRASES.items():
    for _p in _phrases:
        _PHRASE_TO_CAT[_p.lower()] = _cat


def category_for_phrase(phrase):
    """월드가 돌려준 문구를 표준 카테고리로 되돌린다."""
    return _PHRASE_TO_CAT.get(str(phrase or "").lower())


def phrases_for_categories(cats, limit=10):
    """카테고리 목록 → 월드 문구 목록.
    카테고리마다 첫 문구를 먼저 넣어 모든 카테고리가 최소 1개씩 포함되게 한다.
    """
    cats = list(cats or [])
    out = []
    for cat in cats:
        ps = PRIVACY_PHRASES.get(cat, [])
        if ps and ps[0] not in out:
            out.append(ps[0])
    for cat in cats:
        for p in PRIVACY_PHRASES.get(cat, []):
            if p not in out:
                out.append(p)
    return out[:limit]


class WorldStore:
    """YOLO-World 보관함. 한 번만 로드."""

    def __init__(self, weights=WORLD_WEIGHTS):
        self.weights = weights
        self._model = None
        self._ok = None
        self._classes = None

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
        phrases = list(phrases or [])
        if not phrases or not self.available():
            return []
        try:
            # 같은 문구 조합이면 텍스트 임베딩을 다시 만들지 않는다.
            if tuple(phrases) != self._classes:
                self._model.set_classes(phrases)
                self._classes = tuple(phrases)
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
                            "phrase": str(names.get(int(cl[i]), int(cl[i]))),
                            "source": "world"})
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
            wc = float(wb.get("conf", 0.0) or 0.0)
            bc = float(best.get("conf", 0.0) or 0.0) if best else -1.0
            if wc >= bc:
                best, best_iou = wb, iou
    if best is None:
        return False, None, 0.0
    return True, best, round(float(best_iou), 3)
