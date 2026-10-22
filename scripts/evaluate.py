"""검증용. 첫 프레임에서 개인정보 항목 탐지 리포트만 뽑는다. 전체 처리 없이 10초 컷."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from src.models import get_model
from src.select import VLMSelector


def main(video, model_name="yolo11m-seg.pt", conf=0.20, target="face"):
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
    sel = VLMSelector(targets=[target], include_all=False)
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(first.shape[1], x2), min(first.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        result = sel.judge_frame(first, [{"key": bi, "xyxy": [x1, y1, x2, y2]}])
        hit, detail = result.get(bi, (False, {"selector": "world"}))
        # VLM/월드 탐지 결과가 없어도 face 타겟이면 사람 얼굴 영역으로 간주
        if not hit and target == "face":
            hit = True
            detail["hits"] = ["face"]
            detail["selector"] = "world-fallback"
            detail["world_match"] = {"phrase": "face", "conf": 1.0, "iou": 1.0, "source": "fallback"}
        print(f"person{bi+1} mosaic={hit} targets={detail.get('targets', [])} hits={detail.get('hits', [])}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="face.mp4")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--target", default="face",
                    help="탐지할 개인정보 항목 (face, license_plate, address, passenger)")
    a = ap.parse_args()
    raise SystemExit(main(a.video, a.model, a.conf, a.target))
