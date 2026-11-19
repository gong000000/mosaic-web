"""CI용 합성 테스트 영상 생성.

실제 얼굴/차량 영상 없이 apptest의 업로드→분석→렌더→다운로드 흐름을
검사하기 위한 짧은 MP4를 만든다. VLM은 apptest에서 스텁으로 대체된다.

사용:
  python scripts/make_ci_video.py /tmp/mosaic_ci.mp4
"""
import argparse
import os

import cv2
import numpy as np


def make_video(path, seconds=3.0, fps=15, width=640, height=360):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                         float(fps), (int(width), int(height)))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter를 열 수 없습니다: {path}")
    n = max(1, int(float(seconds) * float(fps)))
    for i in range(n):
        frame = np.full((height, width, 3), (40, 45, 50), np.uint8)
        x = 80 + int((width - 240) * i / max(1, n - 1))
        cv2.rectangle(frame, (x, 60), (x + 160, 300), (60, 120, 180), -1)
        cv2.circle(frame, (x + 80, 120), 45, (120, 170, 220), -1)
        cv2.circle(frame, (x + 60, 105), 6, (0, 0, 0), -1)
        cv2.circle(frame, (x + 100, 105), 6, (0, 0, 0), -1)
        cv2.ellipse(frame, (x + 80, 145), (20, 10), 0, 0, 180,
                    (0, 0, 0), 2)
        vw.write(frame)
    vw.release()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"합성 영상 생성 실패: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    print(make_video(args.output, args.seconds, args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
