# -*- coding: utf-8 -*-
"""Unit tests for the 2026-11-28 refinement stack.

These tests do not load SAM 2.1, face-parsing or RapidOCR.  They cover the
safety properties that matter when those models are present or absent:
One Euro never shrinks a box, ByteTrack keeps the FaceTracker interface, OCR
line boxes stay inside their plate, and the GPU client degrades to ``None``.
"""
import numpy as np

from src.bytetrack_adapter import (bytetrack_available, make_face_tracker)
from src.faces import FaceTracker
from src.gpu_client import remote_face_masks, remote_ocr_lines, worker_url
from src.ocr import _dedupe, _pad_line, project_text_boxes
from src.one_euro import OneEuroFilter, smooth_boxes_one_euro
from src.pipeline import _poly_covers_box, _poly_face_ok


def test_one_euro_scalar_converges():
    f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.01)
    vals = [f.filter(10.0, i / 30.0) for i in range(60)]
    assert abs(vals[-1] - 10.0) < 0.05
    assert all(np.isfinite(v) for v in vals)


def test_one_euro_boxes_never_shrink_below_raw():
    raw = [
        {1: [100.0, 100.0, 200.0, 200.0]},
        {1: [104.0, 97.0, 206.0, 203.0]},
        {1: [96.0, 102.0, 198.0, 199.0]},
        {1: [130.0, 90.0, 235.0, 205.0]},
    ]
    out = smooth_boxes_one_euro(raw, fps=30.0, min_cutoff=1.0)
    for src, dst in zip(raw, out):
        for tid, box in src.items():
            ob = dst[tid]
            assert ob[0] <= box[0] + 1e-6
            assert ob[1] <= box[1] + 1e-6
            assert ob[2] >= box[2] - 1e-6
            assert ob[3] >= box[3] - 1e-6
            assert all(np.isfinite(ob))


def test_one_euro_resets_across_scene_cut():
    raw = [{7: [10, 10, 60, 60]}, {7: [300, 300, 360, 360]}]
    out = smooth_boxes_one_euro(raw, cuts={1}, fps=30.0)
    assert out[1][7][0] <= 300 and out[1][7][1] <= 300
    assert out[1][7][2] >= 360 and out[1][7][3] >= 360


def test_bytetrack_factory_keeps_face_tracker_interface():
    tracker = make_face_tracker(force=False)
    assert isinstance(tracker, FaceTracker)
    tracker.update([([100, 100, 160, 170], 0.9)])
    assert tracker.active()
    tracker.reset()


def test_bytetrack_adapter_updates_when_available():
    if not bytetrack_available():
        return
    tracker = make_face_tracker(force=True, frame_rate=30.0)
    for i in range(4):
        box = [100 + i * 2, 100, 160 + i * 2, 170]
        tracker.update([(box, 0.9)])
        assert tracker.active()
    tracker.reset()


def test_ocr_pad_line_stays_in_frame_and_is_tight():
    box = _pad_line([100, 100, 200, 120], 640, 480)
    assert box is not None
    assert box[0] < 100 and box[2] > 200
    assert box[1] < 100 and box[3] > 120
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= 640 and box[3] <= 480
    # A plate-wide line must not be re-padded into a full-frame rectangle.
    assert (box[2] - box[0]) < 140


def test_ocr_projection_moves_lines_with_plate():
    ref = [([100, 200, 300, 240], [[120, 210, 280, 230]])]
    cur = [[120, 220, 320, 260]]
    out = project_text_boxes(ref, cur)
    assert len(out) == 1
    assert out[0][0] >= 120 and out[0][2] <= 320
    assert out[0][1] >= 220 and out[0][3] <= 260


def test_ocr_dedupe_keeps_larger_box():
    big = [0, 0, 100, 40]
    small = [10, 5, 90, 35]
    out = _dedupe([small, big], overlap=0.55)
    assert out == [big]


def test_poly_face_guard_rejects_tiny_mask():
    box = [100, 100, 200, 200]
    full = np.array([[98, 98], [202, 98], [202, 202], [98, 202]])
    tiny = np.array([[120, 120], [150, 120], [150, 150], [120, 150]])
    assert _poly_covers_box(full, box, 0.45)
    assert _poly_face_ok(full, box, 0.45)
    assert not _poly_covers_box(tiny, box, 0.45)
    assert not _poly_face_ok(tiny, box, 0.45)


def test_gpu_client_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("MOSAIC_GPU_WORKER", "")
    assert worker_url() == ""
    frame = np.zeros((32, 32, 3), np.uint8)
    assert remote_face_masks(frame, [[0, 0, 10, 10]]) is None
    assert remote_ocr_lines(frame, [0, 0, 10, 10], "doc") is None
