"""옷 판정: HSV 휴리스틱. CLIP 부정문 실패를 피하기 위해 색상환 직접 계산."""
import cv2
import numpy as np
from .config import CANDS


def torso_roi(crop_bgr, mask_crop=None):
    h, w = crop_bgr.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.70)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    roi = crop_bgr[y1:y2, x1:x2].copy()
    m = None
    if mask_crop is not None:
        try:
            m = mask_crop[y1:y2, x1:x2]
            m = (m > 127).astype(np.uint8)
        except Exception:
            m = None
    if m is not None and m.sum() > 100:
        return roi, m
    h2, w2 = roi.shape[:2]
    return roi, np.ones((h2, w2), dtype=np.uint8)


def _skin_frac(bgr, fg):
    try:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    except Exception:
        return 0.0
    Y, Cr, Cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    skin = ((Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173) & (Y >= 60) & (fg > 0))
    n = float((fg > 0).sum())
    return float(skin.sum()) / n if n > 0 else 0.0


def face_roi(crop_bgr, mask_crop=None):
    """얼굴 영역. 상단 2개 창 중 피부가 많은 쪽. 피부색 판정용."""
    h, w = crop_bgr.shape[:2]
    best = None
    for top_frac, bot_frac in ((0.0, 0.32), (0.10, 0.42)):
        y1 = int(h * top_frac)
        y2 = max(y1 + 1, int(h * bot_frac))
        roi = crop_bgr[y1:y2].copy()
        m = None
        if mask_crop is not None:
            try:
                m = mask_crop[y1:y2]
                m = (m > 127).astype(np.uint8)
            except Exception:
                m = None
        if m is not None and m.sum() > 40:
            f = _skin_frac(roi, m)
            cand = (roi, m, True, f)
        else:
            cand = (roi, np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8), False, 0.0)
        if best is None or cand[3] > best[3]:
            best = cand
    roi, m, real, _ = best
    return roi, m, real


def skin_tone_scores(face_bgr, face_mask, from_real_mask=True):
    """YCrCb 피부 마스크+명도로 흰/검은 피부 점수.
    마스크 없는(배경 섞인) 영역은 0. V 60 미만(머리·그림자) 제외."""
    try:
        ycc = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    except Exception:
        return {"light skin": 0.0, "dark skin": 0.0, "yellow skin": 0.0}
    if not from_real_mask:
        return {"light skin": 0.0, "dark skin": 0.0, "yellow skin": 0.0}
    Y, Cr, Cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    fg = face_mask > 0
    if int(fg.sum()) < 40:
        return {"light skin": 0.0, "dark skin": 0.0, "yellow skin": 0.0}
    skin = ((Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173) & (Y >= 60) & fg)
    n = float(fg.sum())
    frac = float(skin.sum()) / n
    if frac < 0.15:
        return {"light skin": 0.0, "dark skin": 0.0, "yellow skin": 0.0}
    sv = Y[skin].astype(np.float32)
    top = sv[sv >= np.median(sv)]
    bright = float(top.mean()) / 255.0 if top.size > 0 else 0.0
    # 황인(올리브)은 중간 밝기 좁은 봉우리. 기존 흰/검정 판정을 안 뺏는 범위로.
    # 경계톤(p5류)은 조명에 따라 깜빡일 수 있어 엄격하게 좁힘.
    mid = 1.0 - min(1.0, abs(bright - 0.52) / 0.08)
    return {"light skin": frac * bright * bright,
            "dark skin": frac * (1.0 - bright) * (1.0 - bright),
            "yellow skin": frac * mid}


def person_scores(crop_bgr, mask_crop=None):
    """한 사람 전체 점수. 옷(상반신)+피부(얼굴) 합침."""
    roi, m2 = torso_roi(crop_bgr, mask_crop)
    out = scores_for_roi(roi, m2)
    froi, fmask, real = face_roi(crop_bgr, mask_crop)
    out.update(skin_tone_scores(froi, fmask, real))
    return out


def hsv_hist(roi_bgr, mask):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_corr(h1, h2):
    try:
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    except Exception:
        return 0.0


def _ratios(bgr, fg):
    """마스크 영역 색 비율. (ratios dict, night bool)
    옷 색은 피부 제외하고 잰다. 민소매 팔뚝이 빨강으로 잡히던 오탐 방지."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    n = float(fg.sum())
    night = float(V[fg].mean()) < 110
    s_blue = 20 if night else 30
    s_warm = 20 if night else 30
    try:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
        skin = ((Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173) & (Y >= 60))
        fg_c = (fg & (~skin))
        if int(fg_c.sum()) < 50:
            fg_c = fg
    except Exception:
        fg_c = fg
    nc = float(fg_c.sum())
    green = (((H >= 35) & (H <= 75) & (S > 50) & (V > 40)) & fg).sum() / n
    red1 = (((H <= 8) | (H >= 165)) & (S > 60) & (V > 50) & fg_c).sum() / n
    blue_bright = (((H >= 90) & (H <= 130) & (S > 60) & (V > 70) & fg)).sum() / n
    blue_dim = (((H >= 95) & (H <= 125) & (S > s_blue) & (S <= 60) & (V > 45)) & fg).sum() / n
    blue = max(float(blue_bright), 0.85 * float(blue_dim))
    black_lim_v = 95 if night else 75
    black_lim_s = 45 if night else 35
    if night:
        # 밤 검정은 두 통로 중 큰 값. V<45는 색상을 못 믿는 어둠(인간 눈에도 검정).
        # warm_dark는 나트륨등 주황빛 받은 검정. 둘 다 마스크 영역에서만 잰다.
        # 초록·파랑·보라(H 40~150 채도 있음)는 제외. 파랑 V>70과 맞춤.
        very_dark = ((V < 45) & fg).sum() / n
        warm_dark = ((V < 115) & (S < 60) & (((S < 40) | (H < 40) | (H > 150))) & fg).sum() / n
        black = max(float(very_dark), float(warm_dark))
    else:
        black = ((V < black_lim_v) & (S < black_lim_s) & fg).sum() / n
    white = (((S < 35) & (V > 180)) & fg).sum() / n
    warm_white = (((H >= 12) & (H <= 42) & (S >= s_warm) & (S < 110) & (V > 170)) & fg).sum() / n
    white = max(float(white), (1.0 if night else 0.9) * float(warm_white))
    # 틴트 먹은 흰옷(뮤직비디오 색보정). 밝고 채도만 살짝 있으면 흰으로 본다.
    # 밤에는 V>175가 거의 없어서 0. 낮에만 동작한다.
    cool_white = (((S >= 35) & (S < 80) & (V > 175)) & fg).sum() / n
    white = max(float(white), 0.85 * float(cool_white))
    bright = ((V > 150) & fg).sum() / n
    beige = (((S < 80) & (V > 120) & (V < 220)) & fg_c).sum() / n
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ex = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    ey = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_h = ((np.abs(ex) > 60) & fg_c).sum() / n
    edge_v = ((np.abs(ey) > 60) & fg_c).sum() / n
    navy = (((H >= 100) & (H <= 130) & (S > 40) & (V < 120)) & fg).sum() / n
    return ({"green": float(green), "red": float(red1), "blue": float(blue),
             "black": float(black), "white": float(white), "beige": float(beige),
             "navy": float(navy), "bright": float(bright),
             "edge_h": float(edge_h), "edge_v": float(edge_v)}, night)


def _combine(r, night=False):
    green, red1, blue = r["green"], r["red"], r["blue"]
    black, white, beige = r["black"], r["white"], r["beige"]
    edge_h, edge_v = r["edge_h"], r["edge_v"]
    # 밤에는 엣지가 노이즈로 부푼다. 무늬(베이지·줄무늬) 엣지 가중을 낮춘다.
    ew = 0.5 if night else 1.0
    # 줄무늬는 흰이 아님. 엣지 많으면 흰 감점.
    white_adj = white * (1.0 - min(0.7, edge_h * 1.5 + edge_v))
    # 흰은 측정한 밝은 픽셀이 있어야 한다. "색이 없으면 흰" 보너스는 밝을 때만.
    # 안 그러면 밤에 검은옷이 흰 1.0으로 뜬다.
    no_color = (1 - min(1.0, green * 3 + red1 * 3 + blue * 3)) * r["bright"]
    out = {
        "green shirt": green,
        "red shirt": red1,
        "blue shirt": blue,
        "black shirt": black * 0.9 + (1 - white) * 0.1,
        "white sleeveless vest": min(1.0, white_adj * 0.9 + no_color),
        "white blazer": min(1.0, white_adj) * 0.95,
        "beige plaid shirt": min(1.0, (edge_h + edge_v) * 1.5) * 0.5 * ew + beige * 0.5,
        "striped sweater": min(1.0, edge_h * 2.0) * 0.6 * ew + (black * white * 4) * 0.4 + edge_v * 0.2 * ew,
    }
    if "dark navy uniform" in CANDS:
        out["dark navy uniform"] = r["navy"]
    if max(green, red1, blue) > 0.18:
        out["white sleeveless vest"] *= 0.3
        out["white blazer"] *= 0.3
        out["striped sweater"] *= 0.5
        out["beige plaid shirt"] *= 0.5
    out["white sleeveless vest"] *= (1.0 - min(1.0, black))
    out["white blazer"] *= (1.0 - min(1.0, black))
    return out


def scores_for_roi(roi_bgr, mask):
    """전체+상중하 3밴드 중 후보별 최대값. 레이어드 옷(흰 조끼+분홍 티)에 강함.
    마스크 없는 변형은 쓰지 않는다. 배경 수프는 신호가 아니라 노이즈라서
    밤 네온 오탐·배경 검정 오탐의 원인이 된다."""
    fg = mask > 0
    if int(fg.sum()) < 50:
        return {c: 0.0 for c in CANDS}
    h, w = roi_bgr.shape[:2]
    variants = [(roi_bgr, fg, 1.0)]
    for i in range(3):
        y1, y2 = h * i // 3, h * (i + 1) // 3
        band_fg = fg[y1:y2]
        if int(band_fg.sum()) >= 30:
            variants.append((roi_bgr[y1:y2], band_fg, 1.0))
    best = {}
    for bgr, f, wt in variants:
        r, night_v = _ratios(bgr, f)
        for k, v in _combine(r, night_v).items():
            v *= wt
            if v > best.get(k, 0.0):
                best[k] = v
    # 검정은 전역 속성이라 밴드 최대값이 아니라 전체 값을 쓴다.
    # 머리칼·그림자 밴드 하나에 흰 패딩까지 검정이 되던 오탐 방지.
    try:
        r0, _ = _ratios(roi_bgr, fg)
        best["black shirt"] = _combine(r0)["black shirt"]
    except Exception:
        pass
    for c in CANDS:
        best.setdefault(c, 0.0)
    return best


class ClothingClassifier:
    """옷 판정기. 임계값을 객체에 묶는다. 파이프라인·미리보기 공용."""

    def __init__(self, margin=0.04, floor=0.10, abs_thresh=None):
        self.margin = margin
        self.floor = floor
        self.abs_thresh = dict(ABS_THRESH) if abs_thresh is None else dict(abs_thresh)

    def score(self, roi_bgr, mask):
        return scores_for_roi(roi_bgr, mask)

    def decide(self, scores, target):
        return _decide_impl(scores, target, self.margin, self.floor, self.abs_thresh)

    def judge(self, roi_bgr, mask, target):
        return self.decide(self.score(roi_bgr, mask), target)


ABS_THRESH = {
    "green shirt": 0.24, "red shirt": 0.30, "blue shirt": 0.30,
    "black shirt": 0.45,
    "white sleeveless vest": 0.55, "white blazer": 0.55,
}


SKIN_PAIR = ("light skin", "dark skin")
SKIN_TRIO = ("light skin", "dark skin", "yellow skin")


def _decide_impl(scores, target, margin, floor, abs_thresh):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return False, ranked
    if target in SKIN_TRIO:
        v = scores.get(target, 0.0)
        others = [scores.get(o, 0.0) for o in SKIN_TRIO if o != target]
        # 황인은 후발 타겟이라 엄격하게. 약한 중간톤(밤 어두운 얼굴) 오탐 방지.
        need = 0.20 if target == "yellow skin" else 0.08
        return (v >= need and (v - max(others)) >= 0.02), ranked
    ranked = [(k, v) for k, v in ranked if k not in SKIN_TRIO]
    if not ranked:
        return False, ranked
    if scores.get(target, 0.0) >= abs_thresh.get(target, 0.55):
        return True, ranked
    # 검정은 '신호 없음'이라 약한 1등이 노이즈인 경우가 많다. 상대룰 문턱을 높인다.
    need = 0.20 if target == "black shirt" else floor
    best_k, best_v = ranked[0]
    if best_v < need:
        return False, ranked
    second_v = ranked[1][1] if len(ranked) > 1 else 0.0
    return ((best_k == target) and ((best_v - second_v) >= margin)), ranked


def decide(scores, target, margin=0.04, floor=0.10):
    return _decide_impl(scores, target, margin, floor, ABS_THRESH)
