"""상반신/전체 모드 2패스(검출 기록/재생)와 수동 영역 배선 점검. 스텁 모델 사용."""
import os
import sys
import tempfile
import types

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _B:
    def __init__(self, xyxy, ids):
        self.xyxy = _T(np.asarray(xyxy, np.float32).reshape(-1, 4))
        self.id = _T(np.asarray(ids, np.int64))

    def __len__(self):
        return len(self.xyxy.a)


class _TrackModel:
    """사람 1명이 오른쪽으로 움직이고, 5~7프레임은 검출이 끊기며 ID가 바뀐다."""
    calls = 0

    def track(self, source=None, **kw):
        type(self).calls += 1
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            x = 20 + 6 * i
            if 5 <= i <= 7:
                boxes = None
            else:
                tid = 1 if i < 5 else 2
                poly = np.array([[x, 40], [x + 40, 40], [x + 40, 180],
                                 [x, 180]], np.float32)
                boxes = (_B([[x, 40, x + 40, 180]], [tid]), [poly])
            r = types.SimpleNamespace(orig_img=frm)
            if boxes is None:
                r.boxes, r.masks = None, None
            else:
                r.boxes = boxes[0]
                r.masks = types.SimpleNamespace(xy=boxes[1])
            i += 1
            yield r
        cap.release()


def _make_video(d, n=16):
    src = os.path.join(d, "in.mp4")
    wr = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 220))
    rng = np.random.default_rng(1)
    for _ in range(n):
        wr.write((rng.random((220, 240, 3)) * 255).astype(np.uint8))
    wr.release()
    return src


def test_recorder_fills_gap_and_replays():
    import src.pipeline as pl
    d = tempfile.mkdtemp()
    src = _make_video(d)
    rec = pl._DetRecorder()
    for r in _TrackModel().track(source=src):
        rec.add(r)
    assert len(rec.frames) == 16
    added = rec.finalize()
    assert added >= 3
    assert all(len(f['b']) == 1 for f in rec.frames)
    got = list(rec.replay(src))
    assert len(got) == 16
    assert got[6].boxes is not None and len(got[6].boxes) == 1
    assert got[6].masks is not None and got[6].masks.xy[0] is None
    assert got[2].masks.xy[0] is not None


def test_manual_add_and_exclude():
    from src.manual import apply_manual, normalize_regions
    frame = np.full((100, 200, 3), 50, np.uint8)
    orig = frame.copy()
    regs = normalize_regions([
        {"mode": "add", "t0": 0, "t1": 1, "box0": [0.0, 0.0, 0.5, 0.5],
         "box1": [0.5, 0.0, 1.0, 0.5]},
        {"mode": "exclude", "t0": 0, "t1": 1, "box0": [0.0, 0.5, 1.0, 1.0]},
    ])
    assert len(regs) == 2

    def fill(f, x1, y1, x2, y2):
        f[y1:y2, x1:x2] = 255

    frame[:, :] = 50
    frame[60:80, 10:20] = 7
    orig[60:80, 10:20] = 99
    apply_manual(frame, orig, 5, 10.0, regs, fill, fill)
    assert (frame[10, 100] == 255).all()
    assert (frame[60:80, 10:20] == 99).all()
    f2 = np.full((100, 200, 3), 50, np.uint8)
    apply_manual(f2, f2.copy(), 20, 10.0, regs, fill, fill)
    assert (f2 == 50).all()


def test_run_replay_single_inference_and_manual_applied():
    import src.pipeline as pl
    orig_get = pl.get_model
    orig_thread = pl.get_thread_model
    pl.get_model = lambda name: _TrackModel()
    pl.get_thread_model = lambda name: _TrackModel()
    try:
        _TrackModel.calls = 0
        d = tempfile.mkdtemp()
        src = _make_video(d)
        out = os.path.join(d, "out.mp4")
        debug = os.path.join(d, "debug.mp4")
        regs = [{"mode": "add", "t0": 0.0, "t1": 1.5,
                 "box0": [0.8, 0.0, 1.0, 0.2], "box1": [0.8, 0.0, 1.0, 0.2]}]
        job = pl.MosaicJob(src, out, targets=["face", "passenger"],
                           area_mode="upper", mask_all_persons=True,
                           manual_regions=regs, debug_path=debug)
        job.run()
        assert _TrackModel.calls == 1
        assert os.path.getsize(debug) > 0
        cap = cv2.VideoCapture(out)
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert n == 16
    finally:
        pl.get_model = orig_get
        pl.get_thread_model = orig_thread
