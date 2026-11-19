"""GPU 전체 렌더 벤치마크.

L4/T4에서 yolo11m-seg + imgsz=960 전체 파이프라인의 렌더 시간/FPS/피크
VRAM을 측정한다. Modal 컨테이너 안에서 실행한다.

사용:
  python scripts/bench_gpu.py /data/_smoke/face.mp4 --targets face
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch

from src.pipeline import MosaicJob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", default="/data/outputs/_bench_gpu.mp4")
    ap.add_argument("--targets", default="face")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--area-mode", default="face_only")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    peak = [0.0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, timeout=5)
                vals = [int(x.strip()) for x in r.stdout.splitlines()
                        if x.strip().isdigit()]
                if vals:
                    peak[0] = max(peak[0], max(vals))
            except Exception:
                pass
            time.sleep(0.25)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.time()
    try:
        stats = MosaicJob(
            args.input, args.output,
            targets=[x.strip() for x in args.targets.split(",") if x.strip()],
            model_name=args.model, imgsz=args.imgsz,
            keyframe_stride=args.stride,
            conf=args.conf, area_mode=args.area_mode,
            mask_all_persons=True).run()
    finally:
        stop.set()
        th.join(timeout=2)
    elapsed = time.time() - t0
    cap = cv2.VideoCapture(args.output)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    try:
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        alloc, gpu = 0.0, "cpu"
    result = {
        "gpu": gpu,
        "elapsed_sec": round(elapsed, 2),
        "frames": frames,
        "fps": round(frames / max(0.01, elapsed), 2),
        "peak_vram_mb_nvidia_smi": int(peak[0]),
        "peak_torch_alloc_gb": round(alloc, 2),
        "regions": stats.get("_regions"),
        "faces": stats.get("_faces"),
    }
    print(json.dumps(result, ensure_ascii=False))
    if not args.keep:
        try:
            os.remove(args.output)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
