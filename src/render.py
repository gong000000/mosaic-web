"""렌더링/ffmpeg 유틸. 무거운 추론과 분리."""
import os
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg


class FFmpegFrameWriter:
    """프레임을 ffmpeg 파이프로 바로 넘겨 H.264 MP4를 한 번에 만든다.

    기존 경로는 mp4v로 한 번 쓰고, 다시 H.264로 재인코딩하며 오디오를
    믹싱했다(영상 인코딩 2회). 여기서는 raw BGR 프레임을 ffmpeg stdin으로
    보내 원본 오디오와 함께 한 번만 인코딩한다.

    width/height를 0으로 넘기면 첫 프레임에서 실제 크기를 정한다.
    (미리보기 영상처럼 원본 크기를 알아야 할 때만 만들 수 있는 경우)
    """

    def __init__(self, output_path, width, height, fps,
                 audio_src=None, preset="veryfast", crf=20, pix_fmt="yuv420p"):
        self.output_path = output_path
        # yuv420p는 짝수 해상도가 필요하다. 1px 잘라서 맞춘다.
        if width and height:
            self.width = max(2, int(width) - (int(width) % 2))
            self.height = max(2, int(height) - (int(height) % 2))
        else:
            self.width = 0
            self.height = 0
        self.fps = float(fps or 30.0)
        self.audio_src = audio_src
        self.preset = preset
        self.crf = crf
        self.pix_fmt = pix_fmt
        self._proc = None
        self._closed = False
        self._pending = None

    def _start(self):
        """첫 프레임 크기가 정해진 뒤 ffmpeg를 띄운다."""
        if self._proc is not None:
            return
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # 브라우저(Edge/Chrome/Windows Media Foundation) 호환을 위해
        # H.264 Main 프로파일 + 고정 키프레임 + B프레임 없음 + yuv420p를
        # 명시한다. 옵션이 없으면 오디오는 재생돼도 영상이 검게 나오는
        # 증상이 있다.
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{self.width}x{self.height}",
               "-pix_fmt", "bgr24", "-r", f"{self.fps:.6f}", "-i", "-"]
        has_audio = bool(self.audio_src
                         and os.path.exists(self.audio_src))
        if has_audio:
            cmd += ["-i", self.audio_src, "-map", "0:v:0", "-map", "1:a?",
                    "-c:a", "aac"]
        cmd += ["-c:v", "libx264", "-preset", str(self.preset),
                "-crf", str(int(self.crf)), "-pix_fmt", "yuv420p",
                "-profile:v", "main", "-level", "3.1",
                "-g", "30", "-keyint_min", "30", "-bf", "0"]
        if has_audio:
            cmd += ["-shortest"]
        cmd += ["-movflags", "+faststart", self.output_path]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _set_size_from(self, frame):
        h, w = frame.shape[:2]
        self.width = max(2, int(w) - (int(w) % 2))
        self.height = max(2, int(h) - (int(h) % 2))
        self._start()

    @property
    def ok(self):
        return self._proc is not None

    def write(self, frame):
        if self._proc is None:
            if self.width == 0 or self.height == 0:
                self._set_size_from(frame)
                frame = self.resize_to(frame)
            else:
                self._start()
                frame = self.resize_to(frame)
        else:
            frame = self.resize_to(frame)
        try:
            self._proc.stdin.write(
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"ffmpeg 프레임 쓰기 실패: {e}")

    def resize_to(self, frame):
        """프레임을 인코더 해상도에 맞게 자른다/채운다."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if (w, h) == (self.width, self.height):
            return frame
        if (w >= self.width and h >= self.height):
            return frame[:self.height, :self.width]
        return cv2.resize(frame, (self.width, self.height),
                          interpolation=cv2.INTER_AREA)

    def close(self):
        if self._closed:
            return self.output_path
        self._closed = True
        proc = self._proc
        if proc is None:
            raise RuntimeError("ffmpeg writer가 열리지 않았습니다.")
        try:
            if proc.stdin:
                proc.stdin.close()
            code = proc.wait()
        except Exception as e:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg 종료 실패: {e}")
        if (code != 0 or not os.path.exists(self.output_path)
                or os.path.getsize(self.output_path) == 0):
            cleanup_tmp(self.output_path)
            raise RuntimeError("ffmpeg H.264 인코딩 실패")
        return self.output_path

    def abort(self):
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        cleanup_tmp(self.output_path)


def mosaic_block_size(w, h, blocks=30):
    """강도(blocks)에 따른 정사각 블록 크기(px). 클수록 더 강하게 가린다."""
    strength = max(1, int(blocks))
    div = max(2.0, 180.0 / strength)
    return max(2, int(round(min(w, h) / div)))


def render_mosaic(roi, style="pixel", blocks=30, blur=61):
    """기본 모자이크 타일."""
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
    # 가로로 긴 번호판도 세로 디테일이 남지 않도록 정사각 블록으로 만든다.
    bs = mosaic_block_size(w, h, blocks)
    sw, sh = max(1, w // bs), max(1, h // bs)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def feather_mask(mask_u8, guide_bgr=None):
    """마스크 경계를 부드럽게 만든다. 성능을 위해 커널을 작게 유지한다."""
    m = mask_u8.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.GaussianBlur(m, (9, 9), 0)


def mux_audio_h264(video_noaudio, audio_src, output_path, preset="veryfast"):
    """무음 영상에 원본 오디오를 입혀 H.264 MP4로 저장한다.
    실패하면 RuntimeError를 내서 호출부가 원본(무음)으로 대체하게 한다."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cleanup_tmp(output_path)
    cmd = [ff, "-y", "-i", video_noaudio, "-i", audio_src,
           "-c:v", "libx264", "-preset", preset, "-crf", "20",
           "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:a", "aac", "-shortest", "-movflags", "+faststart", output_path]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 오디오 믹싱 실패")
    return output_path


def normalize_video(input_path, output_path, max_width=1920):
    """브라우저/OpenCV 호환성을 위해 H.264 MP4로 다시 인코딩한다.

    MOV/AVI나 일부 코덱은 OpenCV가 못 읽거나 브라우저 미리보기가 안 될 수
    있다. 업로드 원본은 보존하고, 처리용 사본만 MP4로 만든다.
    """
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cleanup_tmp(output_path)
    vf = (f"scale=trunc(min({int(max_width)},iw)/2)*2:"
          "trunc(ih/2)*2")
    cmd = [ff, "-y", "-i", input_path, "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
           output_path]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 영상 변환 실패")
    return output_path


def trim_video(input_path, output_path, start_sec=0.0, end_sec=None):
    """구간 자르기. 스트림 복사(-c copy)는 키프레임 사이를 자르면
    영상 트랙이 통째로 빠질 수 있어서 재인코딩한다.
    실패하면 RuntimeError를 낸다(원본 전체를 처리하는 사고 방지)."""
    start = max(0.0, float(start_sec or 0.0))
    end = None if end_sec is None else float(end_sec)
    if end is not None and end <= start:
        end = None
    if start <= 0 and (end is None or end <= 0):
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cleanup_tmp(output_path)
    cmd = [ff, "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", input_path]
    if end is not None and end > 0:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            output_path]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 구간 자르기 실패")
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


def render_region(frame, x1, y1, x2, y2, style="pixel", blocks=30, blur=61, pad=6):
    """사람과 무관한 배경 영역(번호판·간판·문서 등)을 사각형으로 가린다."""
    x1c, y1c = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x2c, y2c = min(frame.shape[1], int(x2) + pad), min(frame.shape[0], int(y2) + pad)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return False
    roi = frame[y1c:y2c, x1c:x2c]
    mos = render_mosaic(roi, style=style, blocks=blocks, blur=blur)
    # 경계가 튀지 않도록 1px 테두리만 부드럽게 섞는다.
    try:
        m = np.ones(roi.shape[:2], dtype=np.float32)
        m[:1, :] = 0.6
        m[-1:, :] = 0.6
        m[:, :1] = 0.6
        m[:, -1:] = 0.6
        m = m[..., None]
        frame[y1c:y2c, x1c:x2c] = (mos * m + roi * (1 - m)).astype(np.uint8)
    except Exception:
        frame[y1c:y2c, x1c:x2c] = mos
    return True


def render_person_ellipse(frame, x1, y1, x2, y2, style="pixel",
                          blocks=30, blur=61, ratio=1.0):
    """사람 마스크가 없을 때 얼굴/상반신 영역을 타원으로 가린다."""
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    h = y2c - y1c
    fy2 = y1c + int(h * ratio)
    if fy2 - y1c < 15:
        return False
    w_ = x2c - x1c
    fb = np.zeros((fy2 - y1c, w_), dtype=np.uint8)
    cv2.ellipse(fb, (w_ // 2, (fy2 - y1c) // 2),
                (max(5, w_ // 2 - 2), max(5, (fy2 - y1c) // 2 - 2)),
                0, 0, 360, 255, -1)
    fb = cv2.GaussianBlur(fb, (9, 9), 0)
    roi_f = frame[y1c:fy2, x1c:x2c]
    mos = render_mosaic(roi_f, style=style, blocks=blocks, blur=blur)
    mf = (fb.astype(float) / 255.0)[..., None]
    frame[y1c:fy2, x1c:x2c] = (mos * mf + roi_f * (1 - mf)).astype(np.uint8)
    return True


def _area_ratio(area_mode):
    if area_mode == "face_only":
        return 0.32
    if area_mode == "upper":
        return 0.65
    return 1.0


def render_person_roi(frame, x1, y1, x2, y2, mask_roi, style="pixel",
                      blocks=30, blur=61, area_mode="full"):
    """ROI 크기에 맞춰 둔 마스크로 사람을 가린다.

    pipeline이 전체 프레임 크기로 마스크를 매번 resize하지 않고, 사람
    박스 영역만 잘라서 넘긴다. pixel 스타일은 경계 페더링을 생략해
    CPU 렌더 시간을 크게 줄인다.
    """
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    roi_f = frame[y1c:y2c, x1c:x2c]
    if mask_roi is not None:
        try:
            m = mask_roi
            if m.shape[:2] != roi_f.shape[:2]:
                m = cv2.resize(m, (roi_f.shape[1], roi_f.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.uint8:
                m = (m > 0.5).astype(np.uint8) * 255
            if m.mean() >= 8:
                if style != "pixel":
                    m = feather_mask(m, roi_f)
                mos = render_mosaic(roi_f, style=style, blocks=blocks,
                                    blur=blur)
                mf = (m.astype(float) / 255.0)[..., None]
                if mf.mean() >= 0.05:
                    frame[y1c:y2c, x1c:x2c] = (mos * mf
                                               + roi_f * (1 - mf)).astype(np.uint8)
                    return True
        except Exception:
            pass
    return render_person_ellipse(frame, x1c, y1c, x2c, y2c, style=style,
                                 blocks=blocks, blur=blur,
                                 ratio=_area_ratio(area_mode))


def render_person_strict(frame, x1, y1, x2, y2, mask_full_or_None, style="pixel", blocks=30, blur=61, area_mode="full"):
    """마스크 있을 때만 마스크 영역. 없으면 area_mode에 따라 얼굴/상반신/전체 영역을 가린다.
    렌더했으면 True."""
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    if mask_full_or_None is not None:
        try:
            mc = mask_full_or_None[y1c:y2c, x1c:x2c]
            if mc.size > 0 and mc.mean() >= 8:
                # pixel은 하드 경계가 자연스러워 페더링을 생략한다.
                m = mc if style == "pixel" else feather_mask(
                    mc, frame[y1c:y2c, x1c:x2c])
                roi_f = frame[y1c:y2c, x1c:x2c]
                mos = render_mosaic(roi_f, style=style, blocks=blocks, blur=blur)
                mf = (m.astype(float) / 255.0)[..., None]
                if mf.mean() < 0.05:
                    raise ValueError("mask too small")
                frame[y1c:y2c, x1c:x2c] = (mos * mf + roi_f * (1 - mf)).astype(np.uint8)
                return True
        except Exception:
            pass
    return render_person_ellipse(frame, x1c, y1c, x2c, y2c, style=style,
                                 blocks=blocks, blur=blur,
                                 ratio=_area_ratio(area_mode))
