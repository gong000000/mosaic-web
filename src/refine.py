"""VLM이 찾은 영역 박스를 실제 내용에 맞춰 조인다.

VLM은 번호판 같은 작은 영역을 '조금 넓게' 잡는 경우가 많다. 넓게 잡은
사각형 전체를 모자이크하면 실제 글자 바깥까지 번져 보인다. 반대로 너무
줄이면 글자가 남아 프라이버시 목적을 놓친다.

그래서 '글자/경계가 몰려 있는 부분'만 남기되, 원래 박스의 일정 비율
이상은 반드시 유지한다. 조이기만 하고 늘리지는 않는다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# 조이기 대상. 사각형 형태가 명확한 작은 물체만 한다.
TIGHTEN_CATS = {"license_plate", "phone_screen"}

# 원래 박스 넓이의 최소 유지 비율. 이보다 줄어들면 원래 박스를 쓴다.
MIN_KEEP = {"license_plate": 0.60, "phone_screen": 0.55}

# 검색 영역을 원래 박스보다 얼마나 넓히는지.
SEARCH_PAD_RATIO = 0.30


def tighten_region_box(frame_bgr, xyxy, cat, margin=2):
    """영역 박스를 실제 정보가 몰린 부분에 맞춰 줄인다.

    원래 박스를 그대로 쓰지 못할 때(프레임 없음/카테고리 대상 아님)는
    원래 박스를 그대로 돌려준다.
    """
    if frame_bgr is None or cat not in TIGHTEN_CATS:
        return list(xyxy)
    try:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 8:
            return list(xyxy)

        # 검색 영역: 원래 박스를 30% 넓힌다.
        pad_x = int(bw * SEARCH_PAD_RATIO)
        pad_y = int(bh * SEARCH_PAD_RATIO)
        sx1 = max(0, x1 - pad_x)
        sy1 = max(0, y1 - pad_y)
        sx2 = min(w, x2 + pad_x)
        sy2 = min(h, y2 + pad_y)
        crop = frame_bgr[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return list(xyxy)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.add(cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy))
        thr = max(40, int(np.percentile(mag, 88)))
        binary = (mag >= thr).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(
            binary, 8)
        if count <= 1:
            return list(xyxy)

        ox = (x1 - sx1) + bw / 2.0
        oy = (y1 - sy1) + bh / 2.0
        norm = max(1.0, math.hypot(bw, bh))
        best = None
        for i in range(1, count):
            cx0, cy0, cw, ch, area = stats[i]
            if area < 0.01 * binary.size:
                continue
            ccx = cx0 + cw / 2.0
            ccy = cy0 + ch / 2.0
            dist = math.hypot(ccx - ox, ccy - oy) / norm
            if dist > 0.9:
                continue
            score = float(area) * (1.0 - dist)
            if best is None or score > best[0]:
                best = (score, int(cx0), int(cy0), int(cw), int(ch))
        if best is None:
            return list(xyxy)

        _score, bx, by, bw2, bh2 = best
        nx1 = max(x1, sx1 + bx)
        ny1 = max(y1, sy1 + by)
        nx2 = min(x2, sx1 + bx + bw2)
        ny2 = min(y2, sy1 + by + bh2)
        if nx2 - nx1 < 8 or ny2 - ny1 < 6:
            return list(xyxy)
        # 경계 글자(테두리)를 놓치지 않도록 아주 조금 넓힌다.
        nx1 = max(0, nx1 - int(margin))
        ny1 = max(0, ny1 - int(margin))
        nx2 = min(w, nx2 + int(margin))
        ny2 = min(h, ny2 + int(margin))

        orig_area = float(bw * bh)
        new_area = float((nx2 - nx1) * (ny2 - ny1))
        if new_area < orig_area * MIN_KEEP.get(cat, 0.6):
            return list(xyxy)
        return [nx1, ny1, nx2, ny2]
    except Exception:
        return list(xyxy)
