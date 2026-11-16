"""모자이크 품질/보간 회귀 검사.

- 사람 마스크가 바깥으로 번지지 않는지
- 기울어진 얼굴 타원이 정상 동작하는지
- 긴 샘플 간격에서도 번호판 박스를 보간하고, 움직임만큼 넓히는지
- 얼굴 추적이 눈 각도를 따라가는지

실행: .\\.venv\\Scripts\\python.exe scripts\\qualitytest.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.faces import FaceTracker, expand_face_box
from src.pipeline import _interpolate_boxes, _sample_boxes
from src.render import feather_mask, refine_mask_alpha, render_face_ellipse


def test_mask_no_outward_bleed():
    small = np.zeros((20, 20), dtype=np.float32)
    small[5:15, 5:15] = 1.0
    alpha = refine_mask_alpha(small, 80, 80, erode=1)
    assert alpha is not None and alpha.shape == (80, 80), alpha
    assert alpha.dtype == np.uint8, alpha.dtype

    grown = cv2.resize(small, (80, 80), interpolation=cv2.INTER_LINEAR)
    core = grown > 0.5
    assert core.any(), "테스트 마스크가 비었음"
    outside = alpha[~core]
    assert outside.size and int(outside.max()) == 0, \
        f"마스크가 바깥으로 번짐: max={int(outside.max())}"

    hard = core.astype(np.uint8) * 255
    soft = feather_mask(hard)
    assert np.all(soft <= hard), "feather_mask가 마스크를 확장함"
    print("OK mask-inward-feather")


def test_rotated_face_ellipse():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    before = frame.copy()
    ok = render_face_ellipse(frame, 60, 50, 140, 130, angle=25.0)
    assert ok, "회전 얼굴 타원 렌더 실패"
    changed = np.any(frame != before, axis=2)
    assert changed.any(), "모자이크가 적용되지 않음"
    # 박스 모서리(0,0)는 얼굴 ROI 밖이라 그대로여야 한다.
    assert np.array_equal(frame[0, 0], before[0, 0]), "ROI 밖이 변경됨"

    expanded_flat = expand_face_box([60, 50, 140, 130], 200, 200, angle=0)
    expanded_tilt = expand_face_box([60, 50, 140, 130], 200, 200, angle=30)
    assert expanded_tilt[2] - expanded_tilt[0] > \
        expanded_flat[2] - expanded_flat[0], "기울어진 얼굴 여백이 부족함"
    print("OK rotated-face-ellipse")


def test_long_gap_interpolation():
    prev = [{"xyxy": [100, 100, 200, 140], "conf": 1.0}]
    nxt = [{"xyxy": [160, 100, 260, 140], "conf": 1.0}]
    mid = _interpolate_boxes(prev, nxt, 0.5)
    assert mid, "보간 결과 없음"
    x1, y1, x2, y2 = mid[0]["xyxy"]
    assert 170 <= (x1 + x2) // 2 <= 200, mid[0]["xyxy"]
    assert x2 - x1 > 100, f"움직임 여유가 없음: {mid[0]['xyxy']}"
    assert y2 - y1 >= 40, mid[0]["xyxy"]

    samples = {
        0: [{"tid": -1, "rel": None, "abs": [100, 100, 160, 130],
             "due": 0, "xyxy": [100, 100, 160, 130], "conf": 1.0}],
        6: [{"tid": -1, "rel": None, "abs": [200, 100, 260, 130],
             "due": 96, "xyxy": [200, 100, 260, 130], "conf": 1.0}],
    }
    long_mid = _sample_boxes(samples, 30, frame_idx=48, stride=15,
                             interpolate=True)
    assert long_mid and long_mid[0]["xyxy"] != samples[0][0]["xyxy"], \
        "긴 간격(48프레임) 보간이 안 됨"

    stale_samples = {
        0: [samples[0][0]],
        13: [{"tid": -1, "rel": None, "abs": [300, 100, 360, 130],
              "due": 200, "xyxy": [300, 100, 360, 130], "conf": 1.0}],
    }
    stale = _sample_boxes(stale_samples, 30, frame_idx=48, stride=15,
                          interpolate=True)
    assert stale == [samples[0][0]], "너무 먼 미래 박스를 끌고 옴"
    print("OK long-gap-interpolation")


def test_face_angle_tracking():
    tracker = FaceTracker()
    tracker.update([([80, 60, 120, 100], 0.95, 18.0)])
    tracker.update([([81, 60, 121, 100], 0.95, 22.0)])
    track = next(iter(tracker.tracks.values()))
    assert 18.0 < track.angle < 22.0, track.angle
    # 2-tuple(구형 호출)도 깨지지 않아야 한다.
    tracker.update([([82, 60, 122, 100], 0.95)])
    print("OK face-angle-tracking")


def main():
    test_mask_no_outward_bleed()
    test_rotated_face_ellipse()
    test_long_gap_interpolation()
    test_face_angle_tracking()
    print("QUALITY_TEST_PASS")


if __name__ == "__main__":
    main()
