"""run() 분할 뒤 프레임 한 장 렌더(_render_frame)를 합성 프레임/스텁 검출로 점검."""
import os
import sys
import types
from collections import defaultdict

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


class _Boxes:
    def __init__(self, box, tid):
        self.xyxy = _T(np.array([box], np.float32))
        self.id = _T(np.array([tid], np.int64))

    def __len__(self):
        return 1


def _result(frame, box, tid, poly=True):
    b = _Boxes(box, tid)
    x1, y1, x2, y2 = box
    xy = ([np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)]
          if poly else None)
    return types.SimpleNamespace(orig_img=frame, boxes=b,
                                 masks=types.SimpleNamespace(xy=xy)
                                 if poly else None)


def _setup(blanket=True):
    import src.pipeline as pl
    from src.flow import RegionFlow
    from src.faces import FaceTracker
    job = pl.MosaicJob("x.mp4", "y.mp4", targets=["face", "passenger"],
                       area_mode="upper", mask_all_persons=True)
    p = pl._Pass1State(W=200, H=160, fps=10.0, keyframe_stride=5,
                       person_box_samples=defaultdict(
                           lambda: defaultdict(list)),
                       region_samples=defaultdict(lambda: defaultdict(list)))
    plan = pl._Plan(person_cats={"face"}, region_cats=set(),
                    direct_cats=set(), mask_cats={"face"},
                    blanket_cats={"face"} if blanket else set(),
                    cat_on={}, global_on={}, mask_on={},
                    global_hit_targets={}, global_detail={},
                    needs_person=True, face_only_person=False)
    rs = pl._RenderState()
    rs.region_flow = RegionFlow()
    rs.face_trk = FaceTracker()
    return pl, job, p, plan, rs


def _noise():
    rng = np.random.default_rng(0)
    return (rng.random((160, 200, 3)) * 255).astype(np.uint8)


def test_render_frame_masks_person_and_keeps_outside():
    pl, job, p, plan, rs = _setup()
    src = _noise()
    r = _result(src, [60, 20, 120, 150], 1)
    out = job._render_frame(p, plan, rs, r)
    assert out is not r.orig_img and (r.orig_img == src).all()
    assert (out[:, :40] == src[:, :40]).all()
    assert (out[20:60, 60:120] != src[20:60, 60:120]).any()
    assert 1 in rs.rendered_tids and rs.track_frames[1] == 1


def test_render_frame_ghost_keeps_masking_after_track_lost():
    pl, job, p, plan, rs = _setup()
    src = _noise()
    job._render_frame(p, plan, rs, _result(src, [60, 20, 120, 150], 1))
    rs.done += 1
    empty = types.SimpleNamespace(orig_img=src, boxes=None, masks=None)
    out = job._render_frame(p, plan, rs, empty)
    assert (out[20:60, 60:120] != src[20:60, 60:120]).any()
    for _ in range(rs.MAX_MISS):
        rs.done += 1
        job._render_frame(p, plan, rs, empty)
    assert 1 not in rs.smooth


def test_render_frame_synthetic_box_without_polygon():
    pl, job, p, plan, rs = _setup()
    src = _noise()
    r = _result(src, [60, 20, 120, 150], 1, poly=False)
    out = job._render_frame(p, plan, rs, r)
    assert (out != src).any() and 1 in rs.rendered_tids
