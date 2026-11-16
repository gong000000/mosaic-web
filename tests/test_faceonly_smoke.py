"""얼굴만 2패스 경로 배선 점검(스텁 모델, 합성 영상). 검출 품질은 검증하지 않는다."""
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
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy):
        self.xyxy = _T(xyxy)

    def __len__(self):
        return len(self.xyxy.a)


class _FakePose:
    """프레임 0~ 끝까지 (60,40)-(120,200) 사람, 코/눈 키포인트 제공."""

    def predict(self, source=None, **kw):
        r = types.SimpleNamespace()
        r.boxes = _Boxes([[60, 40, 120, 200]])
        kp = np.zeros((1, 17, 2), np.float32)
        kp[0, 0] = (90, 70)
        kp[0, 1] = (84, 62)
        kp[0, 2] = (96, 62)
        c = np.zeros((1, 17), np.float32)
        c[0, :3] = 0.9
        r.keypoints = types.SimpleNamespace(xy=_T(kp), conf=_T(c))
        return [r]


def test_face_only_two_pass_runs(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(pl, "get_model", lambda name: _FakePose())

    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    wr = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 240))
    for i in range(20):
        wr.write(np.full((240, 200, 3), 100 + i, np.uint8))
    wr.release()
    out = os.path.join(d, "out.mp4")
    job = pl.MosaicJob(src, out, targets=["face"], area_mode="face_only",
                       mask_all_persons=True)
    job._run_face_only()
    cap = cv2.VideoCapture(out)
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 20
    cap.release()


def test_face_only_debug_video_is_separate(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(pl, "get_model", lambda name: _FakePose())
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    wr = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 240))
    for i in range(10):
        wr.write(np.full((240, 200, 3), 100 + i, np.uint8))
    wr.release()
    out = os.path.join(d, "out.mp4")
    dbg = os.path.join(d, "debug.mp4")
    job = pl.MosaicJob(src, out, targets=["face"], area_mode="face_only",
                       mask_all_persons=True, debug_path=dbg)
    job._run_face_only()
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert os.path.exists(dbg) and os.path.getsize(dbg) > 0
    out_cap = cv2.VideoCapture(out)
    dbg_cap = cv2.VideoCapture(dbg)
    out_n = dbg_n = 0
    while out_cap.read()[0]:
        out_n += 1
    while dbg_cap.read()[0]:
        dbg_n += 1
    out_cap.release()
    dbg_cap.release()
    assert out_n == dbg_n == 10


def test_pose_head_box_from_keypoints():
    import src.pipeline as pl
    heads = pl._detect_heads_pose(_FakePose(), np.zeros((240, 200, 3), np.uint8))
    assert len(heads) == 1
    x1, y1, x2, y2 = heads[0]
    assert x1 < 84 and x2 > 96 and y1 < 62 and y2 > 70
