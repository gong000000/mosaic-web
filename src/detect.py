"""검출 후처리. 트래커/프리뷰/파이프라인 공통 기하 유틸."""
import cv2
import numpy as np

MIN_W = 24
MIN_H = 48

# 글자가 있어야 하는 배경 개인정보. VLM이 빈 벽/하늘을 큰 사각형으로
# 돌려주는 오탐을 렌더 단계에서 한 번 더 거른다.
TEXT_CATEGORIES = {"address", "id_card", "phone_screen"}


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


def filter_contained(boxes, thresh=0.80, min_area_ratio=0.35):
    """거의 같은 크기로 겹친 중복 검출만 제거한다.

    예전에는 큰 박스 안에 75%만 들어가도 지웠다. 앞사람이 화면을 크게
    덮는 장면에서는 뒤쪽 사람 박스가 전부 사라져 얼굴 모자이크가 빠졌다.
    중복 박스는 보통 크기가 비슷하므로, 면적이 큰 박스의 35% 이상일 때만
    포함 관계를 중복으로 본다. 뒤에 있는 작은 사람은 남긴다.
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
            if areas[i] < areas[j] * min_area_ratio:
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


def region_has_detail(frame, box, cat):
    """간판·문서 박스에 실제 글자/경계가 있는지 확인한다.

    VLM이 빈 벽이나 배경을 '주소·간판'으로 크게 잡는 경우가 있다.
    에지 밀도와 명암 분산이 모두 낮으면 아무것도 없는 사각형으로 보고
    목록/렌더에서 제외한다. 번호판처럼 작은 영역은 이 검사를 하지 않는다.
    """
    if cat not in TEXT_CATEGORIES or frame is None:
        return True
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 12 or y2 - y1 < 10:
            return False
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return False
        edges = cv2.Canny(gray, 45, 135)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        spread = float(gray.std())
        # 임계값을 너무 높이면 흐릿한 화면(야간 CCTV)에서 진짜 간판을
        # '빈 벽'으로 버린다. 오탐은 렌더 단계의 작은 여백과 겹침 제거로
        # 막을 수 있으므로 여기서는 실제 항목을 최대한 살린다.
        if edge_ratio >= 0.008:
            return True
        return spread >= 22.0 and edge_ratio >= 0.004
    except Exception:
        return True

