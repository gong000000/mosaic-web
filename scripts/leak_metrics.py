"""결과 영상의 얼굴 노출(프라이버시 누출) 대리 지표.

정답 라벨이 없으므로, YuNet 을 낮은 임계값(0.3)·면적 제한 없이 결과 영상에
다시 돌려 얼굴이 검출되면 그 프레임을 누출로 센다. 절대값보다 수정 전후
비교에 쓴다.

사용:  python scripts/leak_metrics.py 결과1.mp4 [결과2.mp4 ...]
"""
import os
import sys

import cv2


def count_leaks(path, step=1, threshold=0.3, min_area=180):
    """결과 영상에서 '사람 얼굴 크기'의 검출이 남은 프레임을 센다.

    min_area는 9×12px 같은 배경 무늬 오검출을 누출로 세지 않기 위한
    최소 박스 넓이다. 실제 원거리 얼굴(약 14×14px)은 포함한다.
    """
    model = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "weights",
        "face_detection_yunet_2023mar.onnx")
    det = None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {path}")
    total = leaked = 0
    prev = False
    flips = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        if det is None:
            h, w = frame.shape[:2]
            det = cv2.FaceDetectorYN.create(
                model, "", (w, h), float(threshold), 0.3, 5000)
        _ok, faces = det.detect(frame)
        hit = False
        for row in (faces if faces is not None else []):
            fw, fh = float(row[2]), float(row[3])
            score = float(row[-1])
            if score >= float(threshold) and fw * fh >= float(min_area):
                hit = True
                break
        total += 1
        leaked += int(hit)
        flips += int(hit != prev)
        prev = hit
    cap.release()
    return total, leaked, flips


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    print(f"{'file':40s} {'frames':>7s} {'leaked':>7s} {'leak%':>6s} {'flips':>6s}")
    for p in argv:
        t, l, f = count_leaks(p)
        print(f"{p[-40:]:40s} {t:7d} {l:7d} {100*l/max(1,t):6.1f} {f:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
