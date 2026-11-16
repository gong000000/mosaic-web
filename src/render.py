"""렌더링/ffmpeg 유틸. 무거운 추론과 분리."""
import logging
import math
import os
import subprocess
import tempfile
import cv2
import numpy as np
import imageio_ffmpeg

log = logging.getLogger("mosaic.render")


def _run_ffmpeg(cmd, what):
    """ffmpeg 를 실행하고 실패 시 stderr 꼬리를 로그에 남긴다."""
    cmd = [cmd[0], "-hide_banner", "-loglevel", "error"] + list(cmd[1:])
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=False)
    if r.returncode != 0:
        tail = (r.stderr or b"")[-1500:].decode("utf-8", "replace")
        log.error("ffmpeg %s 실패 (code=%s): %s", what, r.returncode, tail)
    return r


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
        self._err = None

    def _start(self):
        """첫 프레임 크기가 정해진 뒤 ffmpeg를 띄운다."""
        if self._proc is not None:
            return
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{self.width}x{self.height}",
               "-pix_fmt", "bgr24", "-r", f"{self.fps:.6f}", "-i", "-"]
        has_audio = bool(self.audio_src
                         and os.path.exists(self.audio_src))
        if has_audio:
            cmd += ["-i", self.audio_src, "-map", "0:v:0", "-map", "1:a?",
                    "-c:a", "aac"]
        if self.height <= 720:
            level = "3.1"
        elif self.height <= 1080:
            level = "4.1"
        else:
            level = "4.2"
        gop = max(1, int(round(self.fps * 2)))
        cmd += ["-c:v", "libx264", "-preset", str(self.preset),
                "-crf", str(int(self.crf)), "-pix_fmt", "yuv420p",
                "-profile:v", "main", "-level", level,
                "-g", str(gop), "-keyint_min", str(gop), "-bf", "0"]
        cmd += ["-movflags", "+faststart", self.output_path]
        # stderr 를 파이프로 받으면 가득 찼을 때 교착되므로 임시 파일에 쓴다.
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self._err)

    def _set_size_from(self, frame):
        h, w = frame.shape[:2]
        self.width = max(2, int(w) - (int(w) % 2))
        self.height = max(2, int(h) - (int(h) % 2))
        self._start()

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
            buf = np.ascontiguousarray(frame, dtype=np.uint8)
            self._proc.stdin.write(memoryview(buf))
        except (BrokenPipeError, OSError) as e:
            self._log_stderr("프레임 쓰기 실패")
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

    def _log_stderr(self, what):
        try:
            err = getattr(self, "_err", None)
            if err is None:
                return
            err.seek(0, os.SEEK_END)
            size = err.tell()
            err.seek(max(0, size - 1500))
            tail = err.read().decode("utf-8", "replace")
            log.error("ffmpeg %s: %s", what, tail)
        except Exception:
            log.exception("ffmpeg stderr 읽기 실패")

    def _close_err(self):
        err = getattr(self, "_err", None)
        if err is not None:
            try:
                err.close()
            except Exception:
                pass
            self._err = None

    def close(self):
        if self._closed:
            return self.output_path
        self._closed = True
        proc = self._proc
        if proc is None:
            self._close_err()
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
            self._close_err()
            raise RuntimeError(f"ffmpeg 종료 실패: {e}")
        if (code != 0 or not os.path.exists(self.output_path)
                or os.path.getsize(self.output_path) == 0):
            self._log_stderr(f"인코딩 실패 code={code}")
            self._close_err()
            cleanup_tmp(self.output_path)
            raise RuntimeError("ffmpeg H.264 인코딩 실패")
        self._close_err()
        return self.output_path

    def abort(self):
        self._closed = True
        proc = self._proc
        if proc is None:
            self._close_err()
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
        self._close_err()
        cleanup_tmp(self.output_path)


def mosaic_block_size(w, h, blocks=30):
    """강도(blocks)에 따른 정사각 블록 크기(px).

    슬라이더 강도(18~44, 기본 30)가 클수록 블록이 커져(분할 수 감소)
    더 강하게 가려진다. 기본 강도(30)에서는 짧은 축을 약 5칸으로 나누어
    눈·코·입 특징점이 재검출되지 않게 하면서도 얼굴 윤곽은 유지한다.
    """
    strength = max(1, int(blocks))
    count = max(3.0, min(8.0, 156.0 / max(15.0, float(strength))))
    return max(4, int(round(min(w, h) / count)))


def render_mosaic(roi, style="pixel", blocks=30, blur=61, block_size=None):
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
    bs = (max(4, int(block_size)) if block_size
          else mosaic_block_size(w, h, blocks))
    sw, sh = max(1, w // bs), max(1, h // bs)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _stable_block_size(w, h, blocks):
    """얼굴 크기가 1~2px 흔들려도 블록 무늬가 바뀌지 않게 양자화한다."""
    q = max(16, int(round(min(int(w), int(h)) / 8.0)) * 8)
    bs = mosaic_block_size(q, q, blocks)
    return max(4, int(round(bs / 2.0)) * 2)


def feather_mask(mask_u8):
    """마스크 경계를 '안쪽으로만' 부드럽게 만든다.

    기존 MORPH_CLOSE(팽창)는 마스크를 사람 밖으로 넓혀 모자이크가
    배경까지 번지는 원인이었다. 바깥 픽셀은 절대 건드리지 않고,
    안쪽 1~2px만 페이드해 경계를 자연스럽게 만든다.
    """
    m = mask_u8.astype(np.uint8)
    if m.ndim != 2:
        m = m[..., 0]
    soft = cv2.GaussianBlur(m, (5, 5), 0)
    return np.minimum(soft, m)


def refine_mask_alpha(mask_prob, width, height, erode=1):
    """세그멘테이션 마스크(0~1 또는 0~255)를 ROI 알파로 변환한다.

    YOLO 마스크는 모델 해상도(예: 160×160)라서 그대로 늘리면 경계가
    계단처럼 튀고 사람 밖까지 덮는다. 선형 보간으로 경계를 부드럽게
    만들고 1px 침식해 '사람 밖 번짐'을 막는다. 침식이 얇은 팔다리를
    지워 버리면 원래 마스크를 유지한다. 바깥으로는 확장하지 않는다.
    """
    m = np.asarray(mask_prob, dtype=np.float32)
    if m.ndim != 2:
        m = m[..., 0]
    if m.size == 0 or width < 2 or height < 2:
        return None
    if float(m.max()) > 1.5:
        m = m / 255.0
    alpha = cv2.resize(m, (int(width), int(height)),
                       interpolation=cv2.INTER_LINEAR)
    core = (alpha > 0.5).astype(np.uint8) * 255
    if not core.any():
        return None
    if erode > 0:
        shrunk = cv2.erode(core, np.ones((3, 3), np.uint8),
                           iterations=int(erode))
        # 얇은 팔다리/손가락이 통째로 사라지지 않을 때만 침식한다.
        if shrunk.any() and shrunk.mean() >= core.mean() * 0.65:
            core = shrunk
    soft = cv2.GaussianBlur(core, (3, 3), 0)
    return np.minimum(soft, core)


def mux_audio_h264(video_noaudio, audio_src, output_path, preset="veryfast"):
    """무음 영상에 원본 오디오를 입혀 H.264 MP4로 저장한다.
    실패하면 RuntimeError를 내서 호출부가 원본(무음)으로 대체하게 한다."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cleanup_tmp(output_path)
    cmd = [ff, "-y", "-i", video_noaudio, "-i", audio_src,
           "-c:v", "libx264", "-preset", preset, "-crf", "20",
           "-pix_fmt", "yuv420p", "-profile:v", "main", "-bf", "0",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:a", "aac", "-shortest", "-movflags", "+faststart", output_path]
    r = _run_ffmpeg(cmd, "오디오 믹싱")
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 오디오 믹싱 실패")
    return output_path


def strip_audio(video_path):
    """영상 스트림은 복사하고 오디오만 제거한다(목소리·대화 속 개인정보 방지)."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = video_path + ".noaudio.tmp.mp4"
    cleanup_tmp(tmp)
    cmd = [ff, "-y", "-i", video_path, "-an", "-c:v", "copy",
           "-movflags", "+faststart", tmp]
    r = _run_ffmpeg(cmd, "오디오 제거")
    if (r.returncode != 0 or not os.path.exists(tmp)
            or os.path.getsize(tmp) == 0):
        cleanup_tmp(tmp)
        raise RuntimeError("오디오 제거 실패")
    os.replace(tmp, video_path)
    return video_path


def normalize_video(input_path, output_path, max_width=1920):
    """브라우저/OpenCV 호환성을 위해 H.264 MP4로 다시 인코딩한다."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cleanup_tmp(output_path)
    vf = (f"scale=trunc(min({int(max_width)},iw)/2)*2:"
          "trunc(ih/2)*2")
    cmd = [ff, "-y", "-i", input_path, "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-profile:v", "main", "-bf", "0",
           "-c:a", "aac", "-movflags", "+faststart",
           output_path]
    r = _run_ffmpeg(cmd, "영상 변환")
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 영상 변환 실패")
    return output_path


def trim_video(input_path, output_path, start_sec=0.0, end_sec=None):
    """구간 자르기. 스트림 복사(-c copy)는 키프레임 사이를 자르면
    영상 트랙이 통째로 빠질 수 있어서 재인코딩한다."""
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
            "-pix_fmt", "yuv420p", "-profile:v", "main", "-bf", "0",
            "-c:a", "aac", "-movflags", "+faststart",
            output_path]
    r = _run_ffmpeg(cmd, "구간 자르기")
    if (r.returncode != 0 or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        cleanup_tmp(output_path)
        raise RuntimeError("ffmpeg 구간 자르기 실패")
    return output_path


def _face_ratio(w, h):
    """사람 박스 종횡비에 따른 '얼굴만' 세로 비율."""
    w = max(1, int(w))
    h = max(0, int(h))
    aspect = h / float(w)
    if aspect >= 2.0:
        return 0.30
    if aspect >= 1.55:
        return 0.40
    return 0.62


def apply_area_limit(x1, y1, x2, y2, mode="full"):
    """사람 박스에서 가릴 세로 범위를 정한다."""
    h = max(0, int(y2) - int(y1))
    w = max(1, int(x2) - int(x1))
    if mode == "face_only":
        return x1, y1, x2, y1 + int(h * _face_ratio(w, h))
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


def region_pad_for(cat, w, h):
    """영역(번호판 등) 모자이크에 붙일 여백 px."""
    w = max(1, int(w))
    h = max(1, int(h))
    if cat == "license_plate":
        return int(max(2, min(12, 0.12 * min(w, h))))
    if cat == "phone_screen":
        return int(max(2, min(8, 0.05 * min(w, h))))
    return 4


def render_region(frame, x1, y1, x2, y2, style="pixel", blocks=30, blur=61, pad=6):
    """사람과 무관한 배경 영역(번호판·간판·문서 등)을 사각형으로 가린다."""
    x1c, y1c = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x2c, y2c = min(frame.shape[1], int(x2) + pad), min(frame.shape[0], int(y2) + pad)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return False
    roi = frame[y1c:y2c, x1c:x2c]
    mos = render_mosaic(roi, style=style, blocks=blocks, blur=blur)
    out = frame[y1c:y2c, x1c:x2c]
    try:
        alpha = 0.6
        out[1:-1, 1:-1] = mos[1:-1, 1:-1]
        out[:1] = (mos[:1] * alpha + out[:1] * (1 - alpha)).astype(np.uint8)
        out[-1:] = (mos[-1:] * alpha + out[-1:] * (1 - alpha)).astype(np.uint8)
        out[1:-1, :1] = (mos[1:-1, :1] * alpha
                         + out[1:-1, :1] * (1 - alpha)).astype(np.uint8)
        out[1:-1, -1:] = (mos[1:-1, -1:] * alpha
                          + out[1:-1, -1:] * (1 - alpha)).astype(np.uint8)
    except Exception:
        out[:] = mos
    return True


def _face_shape_points(w, h, angle=0.0):
    """얼굴 윤곽에 가까운 24점 폴리곤.

    이마는 넓고 평평하게, 광대는 가장 넓게, 턱으로 갈수록 좁아지는
    비율이다. 단순 타원보다 '사람 얼굴'로 읽히고, 박스가 얼굴보다 조금
    넓어도 주변 배경을 크게 덮지 않는다.
    """
    pts = np.array([
        (0.36, 0.00), (0.50, 0.00), (0.64, 0.00),
        (0.76, 0.04), (0.86, 0.12), (0.92, 0.24),
        (0.94, 0.38), (0.92, 0.50), (0.87, 0.62),
        (0.80, 0.73), (0.70, 0.83), (0.62, 0.89),
        (0.57, 0.915), (0.43, 0.915),
        (0.38, 0.89), (0.30, 0.83), (0.20, 0.73),
        (0.13, 0.62), (0.08, 0.50), (0.06, 0.38),
        (0.08, 0.24), (0.14, 0.12), (0.24, 0.04),
    ], dtype=np.float32)
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    if abs(float(angle or 0.0)) > 0.05:
        center = (float(w) / 2.0, float(h) / 2.0)
        mat = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        pts = cv2.transform(pts.reshape(1, -1, 2), mat).reshape(-1, 2)
    return np.round(pts).astype(np.int32)


def render_face_polygon(frame, polygon, style="pixel",
                        blocks=30, blur=61, block_size=None):
    """랜드마크 외곽 폴리곤(프레임 좌표)으로 모자이크를 적용한다.

    사람마다 다른 이마·광대·턱선을 그대로 쓴다. 실패하면 False를 돌려주고
    호출부는 고정 얼굴형으로 대체한다.
    """
    try:
        poly = np.asarray(polygon, np.int32).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    if len(poly) < 6:
        return False
    h0, w0 = frame.shape[:2]
    x1c = max(0, int(poly[:, 0].min()))
    y1c = max(0, int(poly[:, 1].min()))
    x2c = min(w0, int(poly[:, 0].max()) + 1)
    y2c = min(h0, int(poly[:, 1].max()) + 1)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return False
    w, h = x2c - x1c, y2c - y1c
    if style == "pixel":
        bs = (max(4, int(block_size)) if block_size
              else _stable_block_size(w, h, blocks))
        ox = (x1c // bs) * bs
        oy = (y1c // bs) * bs
        ex = min(w0, ox + int(math.ceil((x2c - ox) / float(bs))) * bs)
        ey = min(h0, oy + int(math.ceil((y2c - oy) / float(bs))) * bs)
    else:
        bs = 4
        ox, oy, ex, ey = x1c, y1c, x2c, y2c
    mask = np.zeros((ey - oy, ex - ox), dtype=np.uint8)
    cv2.fillPoly(mask, [poly - np.array([ox, oy], np.int32)],
                 255, lineType=cv2.LINE_AA)
    if not mask.any():
        return False
    mask = feather_mask(mask)
    roi = frame[oy:ey, ox:ex]
    mos = render_mosaic(roi, style=style, blocks=blocks, blur=blur,
                        block_size=bs)
    mf = (mask.astype(float) / 255.0)[..., None]
    frame[oy:ey, ox:ex] = (mos * mf + roi * (1 - mf)).astype(np.uint8)
    return True


def render_face_ellipse(frame, x1, y1, x2, y2, style="pixel",
                        blocks=30, blur=61, angle=0.0, block_size=None):
    """얼굴 박스에 얼굴형 폴리곤 마스크로 모자이크를 적용한다.

    함수 이름은 기존 호출부 호환을 위해 유지하지만, 실제로는 타원이 아니라
    이마·광대·턱이 있는 얼굴형으로 가린다. 목·어깨·손 등 얼굴 밖은
    건드리지 않는다.

    픽셀 모자이크는 블록 격자를 프레임 좌표에 고정하고 블록 크기를
    양자화한다. 얼굴이 1~2px 움직여도 무늬가 매 프레임 바뀌지 않아
    깜빡임/튐이 줄어든다.
    """
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 8 or y2c - y1c < 8:
        return False
    w, h = x2c - x1c, y2c - y1c
    pts = _face_shape_points(w, h, angle)
    if style == "pixel":
        bs = (max(4, int(block_size)) if block_size
              else _stable_block_size(w, h, blocks))
        ox = (x1c // bs) * bs
        oy = (y1c // bs) * bs
        ex = min(frame.shape[1], ox + int(math.ceil(
            (x2c - ox) / float(bs))) * bs)
        ey = min(frame.shape[0], oy + int(math.ceil(
            (y2c - oy) / float(bs))) * bs)
    else:
        bs = 4
        ox, oy, ex, ey = x1c, y1c, x2c, y2c
    shift = np.array([x1c - ox, y1c - oy], np.int32)
    mask = np.zeros((ey - oy, ex - ox), dtype=np.uint8)
    cv2.fillPoly(mask, [pts + shift], 255, lineType=cv2.LINE_AA)
    if not mask.any():
        return False
    mask = feather_mask(mask)
    roi = frame[oy:ey, ox:ex]
    mos = render_mosaic(roi, style=style, blocks=blocks, blur=blur,
                        block_size=bs)
    mf = (mask.astype(float) / 255.0)[..., None]
    frame[oy:ey, ox:ex] = (mos * mf + roi * (1 - mf)).astype(np.uint8)
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
    fb = feather_mask(fb)
    roi_f = frame[y1c:fy2, x1c:x2c]
    mos = render_mosaic(roi_f, style=style, blocks=blocks, blur=blur)
    mf = (fb.astype(float) / 255.0)[..., None]
    frame[y1c:fy2, x1c:x2c] = (mos * mf + roi_f * (1 - mf)).astype(np.uint8)
    return True


def _area_ratio(area_mode, w=None, h=None):
    if area_mode == "face_only":
        if w is not None and h is not None:
            return _face_ratio(w, h)
        return 0.32
    if area_mode == "upper":
        return 0.65
    return 1.0


def _mosaic_on_masked_rows(roi_f, m, style, blocks, blur):
    """마스크가 실제로 잡힌 행만 모아 모자이크를 만든다."""
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return False
    top = int(rows[0])
    bottom = int(rows[-1]) + 1
    src = roi_f[top:bottom]
    mos = render_mosaic(src, style=style, blocks=blocks, blur=blur)
    m_sub = m[top:bottom]
    mf = (m_sub.astype(float) / 255.0)[..., None]
    if mf.mean() < 0.05:
        return False
    roi_f[top:bottom] = (mos * mf + src * (1 - mf)).astype(np.uint8)
    return True


def render_person_roi(frame, x1, y1, x2, y2, mask_roi, style="pixel",
                      blocks=30, blur=61, area_mode="full"):
    """ROI 크기에 맞춰 둔 마스크로 사람을 가린다."""
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    roi_f = frame[y1c:y2c, x1c:x2c]
    if mask_roi is not None:
        try:
            m = mask_roi
            if m.shape[:2] != roi_f.shape[:2]:
                refined = refine_mask_alpha(
                    m, roi_f.shape[1], roi_f.shape[0])
                if refined is not None:
                    m = refined
                else:
                    m = cv2.resize(m, (roi_f.shape[1], roi_f.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.uint8:
                m = (m > 0.5).astype(np.uint8) * 255
            if m.mean() >= 8:
                if style != "pixel":
                    m = feather_mask(m)
                if area_mode != "full":
                    m = m.copy()
                    allowed_h = int(
                        (y2c - y1c) * _area_ratio(
                            area_mode, x2c - x1c, y2c - y1c))
                    if allowed_h < m.shape[0]:
                        m[allowed_h:, :] = 0
                if _mosaic_on_masked_rows(roi_f, m, style, blocks, blur):
                    return True
        except Exception:
            pass
    return render_person_ellipse(frame, x1c, y1c, x2c, y2c, style=style,
                                 blocks=blocks, blur=blur,
                                 ratio=_area_ratio(area_mode,
                                                   x2c - x1c, y2c - y1c))


def render_person_strict(frame, x1, y1, x2, y2, mask_full_or_None,
                         style="pixel", blocks=30, blur=61, area_mode="full"):
    """마스크가 있으면 마스크 영역만, 없으면 area_mode에 따라 가린다."""
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if x2c - x1c < 20 or y2c - y1c < 20:
        return False
    if mask_full_or_None is not None:
        try:
            mc = mask_full_or_None[y1c:y2c, x1c:x2c]
            if mc.size > 0 and mc.mean() >= 8:
                if mc.shape != (y2c - y1c, x2c - x1c):
                    refined = refine_mask_alpha(
                        mc, x2c - x1c, y2c - y1c)
                    if refined is not None:
                        mc = refined
                    else:
                        mc = cv2.resize(
                            mc, (x2c - x1c, y2c - y1c),
                            interpolation=cv2.INTER_NEAREST)
                if area_mode != "full":
                    mc = mc.copy()
                    allowed_h = int(
                        (y2c - y1c) * _area_ratio(
                            area_mode, x2c - x1c, y2c - y1c))
                    if allowed_h < mc.shape[0]:
                        mc[allowed_h:, :] = 0
                m = mc if style == "pixel" else feather_mask(mc)
                roi_f = frame[y1c:y2c, x1c:x2c]
                if _mosaic_on_masked_rows(roi_f, m, style, blocks, blur):
                    return True
        except Exception:
            pass
    return render_person_ellipse(frame, x1c, y1c, x2c, y2c, style=style,
                                 blocks=blocks, blur=blur,
                                 ratio=_area_ratio(area_mode,
                                                   x2c - x1c, y2c - y1c))
