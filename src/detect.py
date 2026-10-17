"""검출 후처리. 트래커/프리뷰 공통. 작은 박스 기준·겹침 제거."""

MIN_W = 24
MIN_H = 48


def big_enough(x1, y1, x2, y2):
    return (x2 - x1) >= MIN_W and (y2 - y1) >= MIN_H


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
