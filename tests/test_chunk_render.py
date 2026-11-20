import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from src.chunk_render import (_auto_chunk_seconds, _shift_manual,
                              _shift_priors, render_resumable)
from src.render_service import MosaicSettings


@dataclass
class FakePlan:
    targets: tuple = ("face",)
    region_priors: dict = field(default_factory=lambda: {"address": {
        "0": [[10, 10, 50, 30]], "30": [[12, 12, 52, 32]]}})
    person_priors: dict = field(default_factory=lambda: {"face": {
        "15": [[20, 20, 60, 60]]}})
    mask_all_persons: bool = True
    settings: MosaicSettings = field(default_factory=MosaicSettings)
    manual_regions: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    fail_at: int = -1

    def render(self, input_path, output_path, **kwargs):
        idx = len(self.calls)
        self.calls.append((input_path, kwargs.get("mux_audio")))
        if idx == self.fail_at:
            raise RuntimeError("simulated preemption")
        shutil.copyfile(input_path, output_path)
        return {"_faces": 2, "_regions": 1}


def _make_video(path, seconds=7):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ff, "-y",
        "-f", "lavfi", "-i",
        f"testsrc=size=160x120:rate=10:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


def test_auto_chunk_seconds_scales_with_resolution():
    assert _auto_chunk_seconds(3840, 2160) == 20.0
    assert _auto_chunk_seconds(2560, 1440) == 30.0
    assert _auto_chunk_seconds(1920, 1080) == 45.0
    assert _auto_chunk_seconds(1280, 720) == 60.0
    assert _auto_chunk_seconds(640, 360) == 90.0


def test_shift_priors_moves_region_and_filters_person():
    region = _shift_priors(
        {"address": {"0": [[1, 2, 3, 4]], "110": [[5, 6, 7, 8]],
                     "300": [[9, 9, 9, 9]]}},
        start_frame=100, end_frame=200, margin=10, region=True)
    assert "0" in region["address"]
    assert "10" in region["address"]
    assert "200" not in region["address"]

    person = _shift_priors(
        {"face": {"0": [[1, 2, 3, 4]], "110": [[5, 6, 7, 8]]}},
        start_frame=100, end_frame=200, margin=10, region=False)
    assert "0" not in person["face"]
    assert "10" in person["face"]


def test_shift_manual_region_times():
    rows = [{"mode": "add", "t0": 5.0, "t1": 20.0,
             "box0": [0, 0, .1, .1], "box1": [0, 0, .1, .1],
             "shape": "rect"}]
    local = _shift_manual(rows, start_sec=8.0, duration_sec=10.0)
    assert local and local[0]["t0"] == 0.0
    assert local[0]["t1"] == 10.0
    assert _shift_manual(rows, start_sec=30.0, duration_sec=10.0) == []


def test_face_audit_detects_and_fixes_unmasked(tmp_path, monkeypatch):
    from src import face_audit
    from src.faces import FaceBox

    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (160, 120))
    assert writer.isOpened()
    for _ in range(10):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[30:70, 50:90] = (0, 0, 255)
        writer.write(frame)
    writer.release()
    shutil.copyfile(src, out)

    monkeypatch.setattr(
        face_audit, "detect_faces_rotated",
        lambda frame, **kwargs: [(FaceBox([50, 30, 90, 70]), 0.95)])
    before = face_audit.audit_faces(str(src), str(out))
    assert before["checked"] == 10
    assert before["suspicious"] == 10
    assert before["fixable"] == 10

    fixed = face_audit.fix_audited_faces(
        str(out), before["fixes"], style="pixel", blocks=30, blur=61)
    assert fixed == 10
    after = face_audit.audit_faces(str(src), str(out))
    assert after["checked"] == 10
    assert after["suspicious"] == 0


def test_chunked_render_resumes_completed_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_CHUNK_SEC", "2")
    src = tmp_path / "in.mp4"
    _make_video(src)
    out = tmp_path / "out.mp4"

    first = FakePlan(fail_at=1)
    with pytest.raises(RuntimeError):
        render_resumable(
            first, str(src), str(out), model_name="yolo11n-seg.pt",
            imgsz=320, stride=15, conf=0.2, preset="veryfast")

    state_root = tmp_path / ".chunks"
    states = list(state_root.glob("*/state.json"))
    assert len(states) == 1
    with open(states[0], "r", encoding="utf-8") as f:
        state = json.load(f)
    total_chunks = len(state["segments"])
    assert total_chunks >= 3
    assert state["completed"] == [0]
    assert first.calls and all(call[1] is False for call in first.calls)

    second = FakePlan()
    stats = render_resumable(
        second, str(src), str(out), model_name="yolo11n-seg.pt",
        imgsz=320, stride=15, conf=0.2, preset="veryfast")

    assert os.path.exists(out)
    assert len(second.calls) == total_chunks - 1, "완료 구간을 다시 렌더함"
    assert all(call[1] is False for call in second.calls)
    assert stats["_faces"] == 2 * total_chunks
    cap = cv2.VideoCapture(str(out))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        cap.release()
    # 성공하면 구간 체크포인트가 정리된다.
    assert not list(state_root.glob("*/state.json"))
