# -*- coding: utf-8 -*-
"""Run the small, reproducible mask benchmark.

The source videos are intentionally supplied through environment variables
because they are not part of the repository.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.face_seg import FaceSegmentation, _poly_bbox  # noqa: E402
from src.faces import detect_faces  # noqa: E402
from src.pipeline import _poly_face_ok  # noqa: E402

CASE_ENV = {
    "FACE_VIDEO": "",
    "DANCE_VIDEO": "",
    "SUBWAY_VIDEO": "",
    "STREET_VIDEO": "",
}


def _area(poly):
    if poly is None:
        return 0.0
    return abs(float(cv2.contourArea(
        np.asarray(poly, np.float32).reshape(-1, 1, 2))))


def main():
    cases_path = os.path.join(ROOT, "benchmarks", "mask_cases.json")
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    env = {k: os.environ.get(k, v).strip() for k, v in CASE_ENV.items()}
    seg = FaceSegmentation.shared()
    rows = []
    skipped = []
    for case in cases:
        path = env.get(case.get("video", ""), "")
        if not path or not os.path.exists(path):
            skipped.append(case["id"])
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(case.get("frame", 0)))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            skipped.append(case["id"])
            continue
        x1, y1, x2, y2 = [int(v) for v in case["box"]]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if "false_positive" in case.get("tags", []):
            patch = frame[y1:y2, x1:x2]
            dets = detect_faces(patch) if patch.size else []
            rows.append({
                "id": case["id"], "tags": case["tags"],
                "faces_in_patch": len(dets),
                "expect_faces": int(case.get("expect_faces", 0)),
                "pass": len(dets) <= int(case.get("expect_faces", 0)),
            })
            continue
        poly = seg.polygon_for_box(frame, case["box"])
        glasses = float(getattr(seg, "last_parse_glasses", 0.0))
        area = _area(poly)
        box_area = float(bw * bh)
        box_ratio = 0.0
        if poly is not None:
            px1, py1, px2, py2 = _poly_bbox(poly)
            box_ratio = ((px2 - px1) * (py2 - py1)) / box_area
        advanced = bool(poly is not None
                        and _poly_face_ok(poly, case["box"], 0.58))
        rows.append({
            "id": case["id"], "tags": case["tags"],
            "mode": "polygon" if poly is not None else "box_fallback",
            "coverage": round(area / box_area, 4),
            "box_ratio": round(box_ratio, 4),
            "polygon_points": 0 if poly is None else len(poly),
            "glasses_frac": round(glasses, 4),
            "advanced_pass": advanced,
            # A None polygon still yields the pipeline's safe box fallback.
            "pass": True if poly is None else advanced,
        })
    out_dir = os.path.join(ROOT, "report_assets")
    os.makedirs(out_dir, exist_ok=True)
    result = {"rows": rows, "skipped": skipped}
    with open(os.path.join(out_dir, "benchmark_masks.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = ["# Mask benchmark", "",
             "| case | tags | mode | coverage | box_ratio | points | glasses | advanced | safe |",
             "|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {', '.join(r['tags'])} | "
            f"{r.get('mode', '-')} | {r.get('coverage', '-')} | "
            f"{r.get('box_ratio', '-')} | {r.get('polygon_points', '-')} | "
            f"{r.get('glasses_frac', '-')} | {r.get('advanced_pass', '-')} | "
            f"{r.get('pass')} |")
    if skipped:
        lines += ["", "Skipped (video env missing): " + ", ".join(skipped)]
    with open(os.path.join(out_dir, "benchmark_masks.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("skipped", skipped)


if __name__ == "__main__":
    main()
