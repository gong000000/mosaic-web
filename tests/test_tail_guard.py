# -*- coding: utf-8 -*-
import numpy as np

from src.pipeline import _tail_face_guard


def test_tail_guard_adds_missing_face():
    frame = np.zeros((200, 200, 3), np.uint8)
    faces = {1: [10.0, 10.0, 50.0, 50.0]}
    det = lambda _frame: [([100, 100, 150, 150], 0.9)]
    out = _tail_face_guard(frame, faces, done=9, total=10, fps=30.0,
                           detector=det)
    assert len(out) == 2
    assert any(v == [100.0, 100.0, 150.0, 150.0] for v in out.values())


def test_tail_guard_does_not_duplicate_and_is_tail_only():
    frame = np.zeros((200, 200, 3), np.uint8)
    faces = {1: [10.0, 10.0, 50.0, 50.0]}
    det = lambda _frame: [([12, 12, 48, 48], 0.9)]
    early = _tail_face_guard(frame, faces, done=2, total=100, fps=30.0,
                             detector=det)
    assert early == faces
    late = _tail_face_guard(frame, faces, done=99, total=100, fps=30.0,
                            detector=det)
    assert len(late) == 1


def test_tail_guard_can_be_disabled(monkeypatch):
    frame = np.zeros((200, 200, 3), np.uint8)
    faces = {1: [10.0, 10.0, 50.0, 50.0]}
    monkeypatch.setenv("MOSAIC_FACE_TAIL_SAFETY", "0")
    out = _tail_face_guard(frame, faces, done=99, total=100, fps=30.0,
                           detector=lambda _f: [([100, 100, 150, 150], 0.9)])
    assert out == faces
