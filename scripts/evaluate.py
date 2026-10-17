"""첫 프레임에서 사람 수/점수 리포트만 출력한다. 전체 처리는 하지 않는다."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from src.models import get_model
from src.select import HeuristicSelector


def main(video, model_name="yolo11m-seg.pt", conf=0.20, target="green shirt"):
    if isinstance(target, str) and "," in target:
        target = [t.strip() for t in target.split(",") if t.strip()]
    targets = [target] if isinstance(target, str) else list(target)
    det = get_model(model_name)
    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    cap.release()
    if not ok:
        print("READ_FAIL")
        return 1
    pr = list(det.predict(source=first, conf=conf, classes=[0], verbose=False))[0]
    n = 0 if pr.boxes is None else len(pr.boxes)
    print(f"persons={n}")
    if n == 0:
        return 0
    boxes = pr.boxes.xyxy.cpu().numpy().astype(int)
    masks = None
    try:
        if pr.masks is not None:
            masks = pr.masks.data.cpu().numpy()
    except Exception:
        masks = None
    sel = HeuristicSelector(targets)
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(first.shape[1], x2), min(first.shape[0], y2)
        crop = first[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mc = None
        if masks is not None and bi < len(masks):
            m = cv2.resize(masks[bi], (first.shape[1], first.shape[0]))
            m = (m > 0.5).astype("uint8") * 255
            mc = m[y1:y2, x1:x2]
        is_t, detail = sel.judge(crop, mc)
        top = "/".join(f"{k}:{v:.2f}" for k, v in detail.get("top", [])[:2])
        print(f"person{bi+1} mosaic={is_t} {top}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="face.mp4")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--target", default="green shirt",
                    help="단일 또는 쉼표 여러 개. 예: 'green shirt,white sleeveless vest'")
    a = ap.parse_args()
    raise SystemExit(main(a.video, a.model, a.conf, a.target))
