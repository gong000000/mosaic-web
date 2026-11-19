"""주소·간판 검출 precision/recall 평가.

- 예측: VLM 주소·간판 박스(JSON)
- reference: EAST를 3x3 타일로 돌려 글자 영역을 합친 뒤,
  사람/차량 박스 안의 텍스트(셔츠/번호판)를 제외한 박스
- 매칭: IoU>=threshold 또는 한 박스 중심이 다른 박스 안에 포함

사용:
  python scripts/address_eval.py \
      --pred tests/fixtures/address_predictions.json \
      --video-dir "C:/path/to/videos" \
      --out tests/fixtures/address_eval.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.models import get_model
from src.texts import detect_text_quads

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / float(max(1, _area(a) + _area(b) - inter))


def _overlap_ratio(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / float(max(1, _area(inner)))


def _point_in(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def text_reference(frame, model, min_area=1000):
    """EAST 글자 영역을 합쳐 간판/주소 reference 박스를 만든다."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    step_x, step_y = max(1, w // 3), max(1, h // 3)
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            x2 = min(w, x + step_x + 40)
            y2 = min(h, y + step_y + 40)
            for q in detect_text_quads(frame[y:y2, x:x2]):
                q = q.copy()
                q[:, 0] += x
                q[:, 1] += y
                cv2.fillPoly(mask, [q.astype(np.int32)], 255)
    md = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (25, 9)), iterations=1)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(md, 8)
    boxes = []
    for i in range(1, n):
        x, y, ww, hh, _a = stats[i]
        if ww < 25 or hh < 10 or ww * hh < min_area:
            continue
        if y + hh / 2.0 > h * 0.88:
            continue
        boxes.append([int(x), int(y), int(x + ww), int(y + hh)])

    # 셔츠/번호판/차량 위 텍스트 제거
    persons, vehicles = [], []
    try:
        r = model.predict(frame, verbose=False, imgsz=960, conf=0.25)[0]
        if r.boxes is not None:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy()):
                name = model.names[int(c)]
                bb = [float(v) for v in b[:4]]
                if name == "person":
                    persons.append(bb)
                elif name in VEHICLE_CLASSES:
                    vehicles.append(bb)
    except Exception:
        pass
    keep = []
    for b in boxes:
        if any(_overlap_ratio(b, p) > 0.35 for p in persons):
            continue
        if any(_overlap_ratio(b, v) > 0.50 for v in vehicles):
            continue
        keep.append(b)
    return keep


def match_boxes(preds, gts, iou_thresh=0.3):
    """박스 단위 다대다 커버리지 매칭.

    하나의 큰 간판 박스가 여러 글자 영역을 덮을 수 있으므로 1:1 매칭 대신
    예측/정답 각각이 상대편에 IoU 이상으로 겹치거나 중심이 포함되면 매칭으로
    본다. precision은 예측 기준, recall은 정답 기준으로 계산한다.
    """
    def hit(a, b):
        if _iou(a, b) >= iou_thresh:
            return True
        return _point_in(_center(a), b) or _point_in(_center(b), a)

    matched_p = set()
    for pi, p in enumerate(preds):
        if any(hit(p, g) for g in gts):
            matched_p.add(pi)
    matched_g = set()
    for gi, g in enumerate(gts):
        if any(hit(p, g) for p in preds):
            matched_g.add(gi)
    return matched_p, matched_g


def evaluate(pred_rows, video_dir, iou_thresh=0.3, labels=None,
             dump_labels=None):
    model = get_model("yolo11m-seg.pt")
    out = []
    dumped = []
    for row in pred_rows:
        video = row.get("video") or (
            "street.mp4" if row["tag"] == "street"
            else "Walking in New York City.mp4")
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        preds = [b for b in row.get("boxes") or [] if _area(b) >= 1000]
        key = f"{row['tag']}:{int(row['frame'])}"
        if labels is not None and key in labels:
            gts = labels[key]
        else:
            gts = text_reference(frame, model)
        if dump_labels is not None:
            dumped.append({"tag": row["tag"], "frame": int(row["frame"]),
                           "boxes": gts})
        matched_p, matched_g = match_boxes(preds, gts, iou_thresh=iou_thresh)
        out.append({
            "tag": row["tag"], "frame": row["frame"],
            "pred": len(preds), "gt": len(gts),
            "tp": len(matched_p), "tp_gt": len(matched_g),
            "fn": len(gts) - len(matched_g),
            "fp": len(preds) - len(matched_p),
            "pred_boxes": preds, "gt_boxes": gts,
        })
    if dump_labels is not None:
        with open(dump_labels, "w", encoding="utf-8") as f:
            json.dump({"source": "east_text_reference_v1",
                       "frames": dumped}, f, ensure_ascii=False, indent=1)
    return out


def summarize(rows, key=None):
    sub = [r for r in rows if key is None or r["tag"] == key]
    tp = sum(r["tp"] for r in sub)
    tp_gt = sum(r.get("tp_gt", 0) for r in sub)
    fp = sum(r["fp"] for r in sub)
    fn = sum(r["fn"] for r in sub)
    precision = tp / float(max(1, tp + fp))
    recall = tp_gt / float(max(1, tp_gt + fn))
    f1 = (2 * precision * recall / max(1e-9, precision + recall))
    return {
        "frames": len(sub), "pred": sum(r["pred"] for r in sub),
        "gt": sum(r["gt"] for r in sub), "tp": tp, "tp_gt": tp_gt,
        "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--labels", default="",
                    help="검수된 GT 라벨 JSON(없으면 EAST reference 생성)")
    ap.add_argument("--dump-labels", default="",
                    help="EAST reference GT를 라벨 JSON으로 저장")
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()
    with open(args.pred, encoding="utf-8") as f:
        pred_rows = json.load(f)
    labels = None
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            data = json.load(f)
        labels = {f"{r['tag']}:{int(r['frame'])}": r["boxes"]
                  for r in data.get("frames", [])}
    rows = evaluate(pred_rows, args.video_dir, iou_thresh=args.iou,
                    labels=labels, dump_labels=args.dump_labels or None)
    result = {
        "iou": args.iou,
        "total": summarize(rows),
        "street": summarize(rows, "street"),
        "ny": summarize(rows, "ny"),
        "per_frame": rows,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_frame"}, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
