"""렌더링/ffmpeg 유틸. 무거운 추론과 분리."""
import os
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg


def render_mosaic(roi, style="pixel", blocks=30, blur=61):
    """기본 모자이크 타일. 호환용으로 남김."""
    h, w = roi.shape[:2]
    if style == "solid":
        return np.zeros_like(roi)
    if style == "blur":
        k = int(blur) | 1
        k = max(3, k)
        b = cv2.GaussianBlur(roi, (k, k), 0)
        return cv2.GaussianBlur(b, (k, k), 0)
    if h < 4 or w < 4:
        return roi
    gw, gh = max(1, w // blocks), max(1, h // blocks)
    small = cv2.resize(roi, (max(1, w // gw), max(1, h // gh)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def feather_mask(mask_u8, guide_bgr=None):
    m = mask_u8.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=1)
    try:
        if guide_bgr is not None and hasattr(cv2, "ximgproc"):
            g = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
            m32 = m.astype(np.float32) / 255.0
            f = cv2.ximgproc.guidedFilter(g, m32, 8, 0.02)
            return (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return cv2.GaussianBlur(m, (21, 21), 0)
    except Exception:
        return cv2.GaussianBlur(m, (21, 21), 0)


def mux_audio_h264(video_noaudio, audio_src, output_path, preset="veryfast"):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", video_noaudio, "-i", audio_src,
           "-c:v", "libx264", "-preset", preset, "-crf", "20",
           "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:a", "aac", "-shortest", "-movflags", "+faststart", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def trim_video(input_path, output_path, start_sec=0.0, end_sec=None):
    if (not start_sec or start_sec <= 0) and (end_sec is None or end_sec <= 0):
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", input_path]
    if start_sec and start_sec > 0:
        cmd += ["-ss", str(float(start_sec))]
    if end_sec and end_sec > 0:
        cmd += ["-to", str(float(end_sec))]
    cmd += ["-c", "copy", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return output_path


def apply_area_limit(x1, y1, x2, y2, mode="full"):
    h = y2 - y1
    if mode == "face_only":
        return x1, y1, x2, y1 + int(h * 0.32)
    if mode == "upper":
        return x1, y1, x2, y1 + int(h * 0.65)
    return x1, y1, x2, y2


def cleanup_tmp(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


def render_person_strict(frame, x1, y1, x2, y2, mask_full_or_None, style="pixel", blocks=30, blur=61):
    """마스크 있을 때만 마스크 영역. 없으면 얼굴 영역(상단 35%)만 타원으로.
    렌더했으면 True."""
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    if mask_full_or_None is not None:
        try:
            mc = mask_full_or_None[y1c:y2c, x1c:x2c]
            if mc.size > 0 and mc.mean() >= 8:
                m = feather_mask(mc, frame[y1c:y2c, x1c:x2c])
                roi_f = frame[y1c:y2c, x1c:x2c]
                mos = render_mosaic(roi_f, style=style, blocks=blocks, blur=blur)
                mf = (m.astype(float) / 255.0)[..., None]
                if mf.mean() < 0.05:
                    raise ValueError("mask too small")
                frame[y1c:y2c, x1c:x2c] = (mos * mf + roi_f * (1 - mf)).astype(np.uint8)
                return True
        except Exception:
            pass
    h = y2c - y1c
    fy2 = y1c + int(h * 0.35)
    if fy2 - y1c < 15:
        return False
    w_ = x2c - x1c
    fb = np.zeros((fy2 - y1c, w_), dtype=np.uint8)
    cv2.ellipse(fb, (w_ // 2, (fy2 - y1c) // 2),
                (max(5, w_ // 2 - 2), max(5, (fy2 - y1c) // 2 - 2)),
                0, 0, 360, 255, -1)
    fb = cv2.GaussianBlur(fb, (15, 15), 0)
    roi_f = frame[y1c:fy2, x1c:x2c]
    mos = render_mosaic(roi_f, style=style, blocks=max(12, blocks // 2), blur=blur)
    mf = (fb.astype(float) / 255.0)[..., None]
    frame[y1c:fy2, x1c:x2c] = (mos * mf + roi_f * (1 - mf)).astype(np.uint8)
    return True
