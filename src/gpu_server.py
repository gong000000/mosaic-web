"""Small FastAPI refiner service for a separate GPU worker.

Run locally::

    python -m src.gpu_server --host 127.0.0.1 --port 8600

The main Streamlit process can point at it with ``MOSAIC_GPU_WORKER``.  The
worker owns SAM 2.1, face parsing and OCR; if the worker is down the caller
falls back to the in-process implementations or the MediaPipe mesh.

This module intentionally has no Streamlit/job-queue dependency.  It can be
deployed as the ASGI app in ``modal_worker.py``.
"""
from __future__ import annotations

import base64
import hmac
import logging
import os
import time

import cv2
import numpy as np

# Never recurse if the worker process inherits the client environment variable.
os.environ["MOSAIC_GPU_WORKER"] = ""

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .face_seg import FaceSegmentation  # noqa: E402
from .ocr import document_text_boxes, plate_text_boxes  # noqa: E402

log = logging.getLogger("mosaic.gpu_server")
app = FastAPI(title="Mosaic GPU refiner", version="1.0")


class FaceRequest(BaseModel):
    image_b64: str
    boxes: list[list[float]]
    use_sam2: bool = True
    use_parse: bool = True



class OcrRequest(BaseModel):
    image_b64: str
    region: list[float] | None = None
    kind: str = "doc"


def _check_token(token):
    want = os.environ.get("MOSAIC_WORKER_TOKEN", "").strip()
    if not want:
        # Secure by default: an unconfigured worker is not usable.  Local
        # development can opt in explicitly with MOSAIC_WORKER_ALLOW_ANON=1.
        if os.environ.get("MOSAIC_WORKER_ALLOW_ANON", "0") == "1":
            return
        raise HTTPException(status_code=503,
                            detail="worker token is not configured")
    got = str(token or "").strip()
    if not got or not hmac.compare_digest(got, want):
        raise HTTPException(status_code=401, detail="invalid worker token")


def _decode(image_b64):
    try:
        raw = base64.b64decode(str(image_b64))
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decode failed")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid image: {exc}") from exc


@app.get("/health")
def health():
    # Health intentionally warms the models: the first dashboard/keepalive
    # probe should prove that SAM 2.1 and face parsing actually load, not just
    # that the FastAPI process is alive.
    seg = FaceSegmentation.shared()
    sam2 = bool(seg._sam_available())
    face_parse = bool(seg._parse_available())
    return {
        "ok": True,
        "sam2": sam2,
        "face_parse": face_parse,
        "device": getattr(seg, "_dev", "cpu"),
    }


@app.post("/v1/face-masks")
def face_masks(req: FaceRequest, x_worker_token: str = Header(default="")):
    _check_token(x_worker_token)
    frame = _decode(req.image_b64)
    t0 = time.time()
    seg = FaceSegmentation.shared()
    try:
        polys = seg.polygons_for_boxes(
            frame, req.boxes, use_parse=req.use_parse,
            use_sam2=req.use_sam2)
    except Exception:
        log.exception("worker batch face mask failed")
        polys = []
    if len(polys) != len(req.boxes):
        polys = []
        for box in req.boxes:
            try:
                polys.append(seg.polygon_for_box(
                    frame, box, use_parse=req.use_parse,
                    use_sam2=req.use_sam2))
            except Exception:
                log.exception("worker face mask failed box=%s", box)
                polys.append(None)
    out = []
    for poly in polys:
        out.append(None if poly is None else
                   [[int(v) for v in p] for p in np.asarray(poly)])
    return {"polygons": out,
            "ms": round((time.time() - t0) * 1000.0, 1),
            "backend": "sam2+parse"}


@app.post("/v1/ocr/lines")
def ocr_lines(req: OcrRequest, x_worker_token: str = Header(default="")):
    _check_token(x_worker_token)
    frame = _decode(req.image_b64)
    if req.kind == "plate":
        if not req.region:
            raise HTTPException(status_code=400, detail="plate region required")
        boxes = plate_text_boxes(frame, req.region)
    else:
        boxes = document_text_boxes(
            frame, req.region or [0, 0, frame.shape[1], frame.shape[0]])
    return {"boxes": [[int(v) for v in b] for b in boxes],
            "backend": "rapidocr"}


@app.get("/")
def root():
    return {"service": "mosaic-gpu-refiner", "endpoints":
            ["/health", "/v1/face-masks", "/v1/ocr/lines"]}


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
