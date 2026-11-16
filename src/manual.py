"""사용자가 직접 지정하는 수동 영역(추가/제외).

영역 하나 = 시작/끝 시간 + 시작 박스 + 끝 박스. 박스는 프레임 크기에 대한
0~1 비율 좌표라 해상도가 달라도 쓸 수 있다. 시간 구간 안에서 두 박스를
선형 보간해 움직이는 대상도 따라간다.

- mode "add":     자동 검출과 무관하게 항상 가린다(누락 보완).
- mode "exclude": 그 영역은 모자이크를 되돌려 원본을 보인다(과검출 제거).
  프라이버시 위험이 있으므로 사용자가 명시적으로 만든 것만 적용한다.
"""
from __future__ import annotations

import json


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def normalize_regions(regions):
    """UI/세션에서 온 목록을 검증해 깨끗한 dict 목록으로 만든다."""
    out = []
    for r in regions or []:
        try:
            mode = r.get("mode", "add")
            if mode not in ("add", "exclude"):
                continue
            t0, t1 = float(r["t0"]), float(r["t1"])
            if t1 < t0:
                t0, t1 = t1, t0
            b0 = [_clamp01(v) for v in r["box0"]]
            b1 = [_clamp01(v) for v in (r.get("box1") or r["box0"])]
            for b in (b0, b1):
                if len(b) != 4 or b[2] - b[0] < 0.005 or b[3] - b[1] < 0.005:
                    raise ValueError("box")
            shape = r.get("shape", "rect")
            if shape not in ("rect", "ellipse"):
                shape = "rect"
            out.append({"mode": mode, "t0": t0, "t1": t1, "box0": b0,
                        "box1": b1, "shape": shape,
                        "label": str(r.get("label", ""))[:40]})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def regions_signature(regions):
    return json.dumps(normalize_regions(regions), sort_keys=True,
                      ensure_ascii=False)


def box_at(region, t):
    """시각 t(초)에서의 박스(비율). 구간 밖이면 None."""
    t0, t1 = region["t0"], region["t1"]
    if t < t0 - 1e-6 or t > t1 + 1e-6:
        return None
    if t1 - t0 < 1e-6:
        return list(region["box0"])
    a = (t - t0) / (t1 - t0)
    return [region["box0"][i] + (region["box1"][i] - region["box0"][i]) * a
            for i in range(4)]


def _px(box, w, h):
    return (max(0, int(round(box[0] * w))), max(0, int(round(box[1] * h))),
            min(w, int(round(box[2] * w))), min(h, int(round(box[3] * h))))


def active_boxes(regions, t, w, h, mode):
    """시각 t에 켜져 있는 해당 mode 영역의 (픽셀박스, shape) 목록."""
    out = []
    for r in regions:
        if r["mode"] != mode:
            continue
        b = box_at(r, t)
        if b is None:
            continue
        x1, y1, x2, y2 = _px(b, w, h)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append(((x1, y1, x2, y2), r["shape"]))
    return out


def apply_manual(frame, original, frame_idx, fps, regions, render_rect,
                 render_ellipse):
    """프레임에 수동 영역을 적용한다(모든 자동 렌더가 끝난 뒤 호출).

    1) add: 항상 가린다   2) exclude: 원본 픽셀로 되돌린다(추가보다 우선).
    render_rect/render_ellipse 는 (frame, x1, y1, x2, y2) 로 가리는 함수.
    original 은 자동 모자이크 이전의 프레임(exclude 에 필요, 없으면 None).
    """
    if not regions:
        return frame
    h, w = frame.shape[:2]
    t = frame_idx / float(fps if fps > 0 else 30.0)
    for (x1, y1, x2, y2), shape in active_boxes(regions, t, w, h, "add"):
        if shape == "ellipse":
            render_ellipse(frame, x1, y1, x2, y2)
        else:
            render_rect(frame, x1, y1, x2, y2)
    if original is not None:
        for (x1, y1, x2, y2), _shape in active_boxes(
                regions, t, w, h, "exclude"):
            frame[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    return frame
