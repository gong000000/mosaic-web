# -*- coding: utf-8 -*-
import os
import sys
import time

sys.path.insert(0, r"C:\Users\gong0000\Desktop\mosaic-web")
os.environ.setdefault("MOSAIC_FACE_MASK_MAX_GROW", "1.20")

from src.face_audit import audit_faces, fix_audited_faces  # noqa: E402
from src.pipeline import mosaic_video  # noqa: E402

ROOT = r"C:\Users\gong0000\Desktop\mosaic-web"
CASES = [
    ("face", r"C:\Users\gong0000\Downloads\face.mp4"),
    ("subway", r"C:\Users\gong0000\AppData\Local\Temp\opencode\subway_all_clip.mp4"),
    ("dance", r"C:\Users\gong0000\Downloads\A20 LITE(라이트) 'NEON' intro Dance Mirror mode 안무 거울모드 🎶 #shorts #a20lite #dance #kpop #dancechallenge.mp4"),
    ("street", r"C:\Users\gong0000\Downloads\street_preview.mp4"),
]

for tag, src in CASES:
    out = os.path.join(ROOT, "outputs", f"verify_{tag}_final.mp4")
    t0 = time.time()
    print(f"START {tag} {src}", flush=True)
    try:
        stats = mosaic_video(
            src, out, targets=["face"], area_mode="face_only",
            mask_all_persons=True, model_name="yolo11n-seg.pt",
            conf=0.20, imgsz=480, blocks=30, style="pixel",
            mux_audio=False, api_key="dummy", vlm_call_limit=0)
        render_s = time.time() - t0
        print(f"DONE {tag} render={render_s:.1f}s stats={stats}", flush=True)
        a0 = time.time()
        audit = audit_faces(src, out)
        print(f"AUDIT {tag} {time.time() - a0:.1f}s "
              f"checked={audit.get('checked')} suspicious={audit.get('suspicious')} "
              f"fixable={audit.get('fixable')}", flush=True)
        fixed = fix_audited_faces(out, audit.get("fixes") or [])
        post = audit_faces(src, out)
        print(f"FIX {tag} boxes={fixed} "
              f"post_checked={post.get('checked')} "
              f"post_suspicious={post.get('suspicious')}", flush=True)
    except Exception as exc:
        print(f"FAIL {tag} {exc!r}", flush=True)
        import traceback
        traceback.print_exc()
