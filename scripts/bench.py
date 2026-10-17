"""속도 측정. 모델·해상도별 초당 프레임."""
import sys
import time

sys.path.insert(0, '.')
from src.models import get_model


def bench(model_name, imgsz, video, frames=30):
    import cv2
    m = get_model(model_name)
    cap = cv2.VideoCapture(video)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    t0 = time.time()
    n = 0
    for _ in range(frames):
        list(m.track(source=fr, persist=False, conf=0.20, classes=[0],
                     verbose=False, imgsz=imgsz, iou=0.5, max_det=20))
        n += 1
    el = time.time() - t0
    return n / el


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\gong0000\Downloads\face.mp4'
    for name, sz in [('yolo11n-seg.pt', 480), ('yolo11m-seg.pt', 640), ('yolo11x-seg.pt', 960)]:
        fps = bench(name, sz, video)
        print(f'{name}@{sz}: {fps:.1f} fps' if fps else f'{name}: FAIL')


if __name__ == "__main__":
    main()
