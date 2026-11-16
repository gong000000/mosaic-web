"""번호판·간판 같은 VLM 박스를 앵커 사이에서 광학 흐름으로 따라가게 한다.

VLM 박스는 샘플 프레임에만 있어서, 움직이는 대상은 박스가 제자리에
남거나 선형 보간으로만 이동했다. 여기서는 앵커 박스 안의 특징점을
cv2.calcOpticalFlowPyrLK 로 추적해 프레임마다 박스를 이동시키고,
다음 앵커가 오면 그 위치로 맞춘다(드리프트가 앵커 간격 안에서만 쌓인다).
"""
import cv2
import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (max(1.0, a[2] - a[0]) * max(1.0, a[3] - a[1])
          + max(1.0, b[2] - b[0]) * max(1.0, b[3] - b[1]) - inter)
    return inter / max(1.0, ua)


def shift_box(prev_gray, cur_gray, box):
    """두 흑백 프레임 사이에서 박스 안 특징점의 중앙값 이동량만큼 박스를 옮긴다.

    얼굴 검출이 끊긴 프레임(옆모습, 모션 블러)에서 박스가 속도 감쇠로만
    움직이지 않고 실제 영상 움직임을 따라가게 한다. 실패하면 박스를 그대로 준다.
    """
    rf = RegionFlow()
    rf._prev, rf._cur = prev_gray, cur_gray
    try:
        return rf._shift([float(v) for v in box])
    except Exception:
        return [float(v) for v in box]


class RegionFlow:
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        20, 0.03))

    def __init__(self):
        self._prev = None      # 이전 프레임(흑백)
        self._cur = None
        self._state = {}       # cat -> (앵커 서명, [박스])

    def begin(self, frame_bgr):
        """프레임마다 한 번, 모자이크를 그리기 전의 원본 프레임으로 호출."""
        self._cur = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def end(self):
        self._prev, self._cur = self._cur, None

    def reset(self):
        """장면 전환 시 흐름 상태를 버린다."""
        self._prev = None
        self._state.clear()

    def _shift(self, box):
        prev, cur = self._prev, self._cur
        h, w = prev.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return list(box)
        roi = prev[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=80, qualityLevel=0.01,
                                      minDistance=3)
        if pts is None or len(pts) < 4:
            return list(box)
        pts = pts.reshape(-1, 2) + np.array([x1, y1], np.float32)
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **self.LK)
        if p1 is None:
            return list(box)
        # 역방향 검증으로 잘못 따라간 점을 거른다.
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **self.LK)
        ok = ((st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
              & (np.linalg.norm((p0r - p0).reshape(-1, 2), axis=1) < 1.5))
        if int(ok.sum()) < 4:
            return list(box)
        d = (p1.reshape(-1, 2) - p0.reshape(-1, 2))[ok]
        dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
        bw, bh = box[2] - box[0], box[3] - box[1]
        dx = max(-0.5 * bw, min(0.5 * bw, dx))
        dy = max(-0.5 * bh, min(0.5 * bh, dy))
        return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

    def track(self, cat, anchors):
        """anchors: 직전 앵커의 박스 dict 목록(xyxy). 이동된 사본을 돌려준다.

        모든 프레임에서 호출해야 한다(없으면 빈 목록).
        """
        sig = tuple(tuple(int(v) for v in a["xyxy"]) for a in anchors)
        st = self._state.get(cat)
        if st is None or st[0] != sig or self._prev is None \
                or self._cur is None:
            boxes = [[float(v) for v in a["xyxy"]] for a in anchors]
        else:
            boxes = [self._shift(b) for b in st[1]]
        self._state[cat] = (sig, boxes)
        return [dict(a, xyxy=[int(round(v)) for v in b])
                for a, b in zip(anchors, boxes)]

    @staticmethod
    def merge(flow_boxes, other_boxes, min_iou=0.3):
        """흐름 박스에 더해, 어느 것과도 겹치지 않는 보간 박스도 남긴다.

        흐름이 실패해 박스가 제자리에 남은 경우를 프라이버시 쪽으로 보완한다.
        """
        out = list(flow_boxes)
        for ob in other_boxes:
            if not any(_iou(fb["xyxy"], ob["xyxy"]) >= min_iou
                       for fb in flow_boxes):
                out.append(ob)
        return out
