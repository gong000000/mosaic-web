"""하이브리드 샘플러 검증.

합성 영상(장면 전환 3회 + 움직임 스침 1회)을 만들어
균등 샘플링이 놓치는 구간을 샘플러가 잡는지 확인한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\sampletest.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.sampling import plan_samples, scan_motion

FRAMES = 360
CUTS = (90, 180, 270)
EVENT = (205, 210)


def _make_video(path, frames=FRAMES, size=(320, 240), fps=30.0):
    """장면 전환 3회와 움직임 스침 1회(빈 영역 + 밝은 정사각형)를 넣은 영상."""
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    rng = np.random.default_rng(7)
    base = 30
    for i in range(frames):
        if i < 90:
            base = 30
        elif i < 180:
            base = 200
        elif i < 270:
            base = 120
        else:
            base = 60
        frame = np.full((h, w, 3), base, dtype=np.uint8)
        if EVENT[0] <= i <= EVENT[1]:
            frame[100:140, 120:200] = 255
        frame = frame + rng.integers(0, 6, size=(h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main():
    tmp = tempfile.mkdtemp(prefix="mosaic_sampling_")
    path = os.path.join(tmp, "synthetic.mp4")
    _make_video(path)

    total, motion = scan_motion(path, scan_max=360)
    assert total > 0, "영상 길이를 읽지 못함"
    assert motion, "움직임 변화량이 비어 있음"

    plan = plan_samples(path, base_count=4, max_count=10)
    idxs = plan["indices"]
    uniform = sorted({int(round(v)) for v in
                       np.linspace(0, total - 1, 4)})
    print(f"total={total} uniform={uniform}")
    print(f"samples={len(idxs)} indices={idxs}")
    print(f"cuts={plan['cuts']} peaks={plan['peaks']}")

    assert 0 in idxs and total - 1 in idxs, "처음/끝 프레임이 빠져 있음"
    assert len(idxs) <= 10, f"상한 초과: {len(idxs)}"
    assert len(idxs) > len(uniform), \
        f"샘플이 늘지 않음: {idxs}"

    # 균등 4장으로는 절대 안 잡히는 구간을 샘플러가 잡는지 확인한다.
    for cut in CUTS:
        assert cut not in uniform, f"테스트 설계 오류: {cut}는 균등 샘플에 있음"
        assert cut in idxs, f"장면 전환 {cut}프레임을 놓침: {idxs}"

    # 움직임 스침 구간도 잡는지 확인한다.
    assert any(p not in uniform for p in plan["peaks"]), \
        f"움직임 피크를 놓침 (peaks={plan['peaks']})"

    print("SAMPLING_TEST_PASS")
    print(f"균등 {len(uniform)}장 → 하이브리드 {len(idxs)}장 "
          f"(컷 {plan['cuts']}, 피크 {plan['peaks']})")


if __name__ == "__main__":
    main()
