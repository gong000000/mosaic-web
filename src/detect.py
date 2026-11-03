"""검출 후처리. 트래커/프리뷰/파이프라인 공통 기하 유틸."""

MIN_W = 24
MIN_H = 48


def big_enough(x1, y1, x2, y2):
    return (x2 - x1) >= MIN_W and (y2 - y1) >= MIN_H


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    ba = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / max(1.0, float(aa + ba - inter))


def match_box(person_xyxy, boxes, iou_thresh=0.12, expand=0.03):
    """사람 박스와 VLM 박스를 묶는다.
    VLM 박스 중심이 사람 박스(약간 확장) 안에 있거나 IoU가 충분하면 매칭.
    returns (hit, best_box|None, best_iou)."""
    px1, py1, px2, py2 = person_xyxy
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    ex1, ey1 = px1 - pw * expand, py1 - ph * expand
    ex2, ey2 = px2 + pw * expand, py2 + ph * expand
    best, best_iou = None, 0.0
    for box in boxes or []:
        x1, y1, x2, y2 = box["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        overlap = iou(person_xyxy, box["xyxy"])
        inside = (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)
        if inside or overlap >= iou_thresh:
            bc = float(box.get("conf", 0.0) or 0.0)
            best_conf = float(best.get("conf", 0.0) or 0.0) if best else -1.0
            if bc >= best_conf:
                best, best_iou = box, overlap
    if best is None:
        return False, None, 0.0
    return True, best, round(float(best_iou), 3)


def filter_contained(boxes, thresh=0.75):
    """큰 박스 안에 75% 이상 들어간 작은 박스 제외. 합쳐진 오탐 제거용.
    boxes: (N,4) array-like. keep 불리언 리스트 반환."""
    n = len(boxes)
    keep = [True] * n
    areas = [(int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1])) for b in boxes]
    for i in range(n):
        if areas[i] <= 0:
            keep[i] = False
            continue
        for j in range(n):
            if i == j or areas[j] <= areas[i]:
                continue
            ix1 = max(int(boxes[i][0]), int(boxes[j][0]))
            iy1 = max(int(boxes[i][1]), int(boxes[j][1]))
            ix2 = min(int(boxes[i][2]), int(boxes[j][2]))
            iy2 = min(int(boxes[i][3]), int(boxes[j][3]))
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / areas[i] >= thresh:
                keep[i] = False
                break
    return keep
