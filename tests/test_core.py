"""의존성이 가벼운 핵심 로직 테스트 (GPU/YOLO/Streamlit 없이 실행)."""
import os
import sys
import tempfile
import threading
import types
import json
import time

import numpy as np  # noqa: E402
import cv2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.faces import (FaceTracker, expand_face_box, detect_faces,
                       filter_faces_by_person)  # noqa: E402


def _moving_dets(n, speed):
    return [([100 + speed * i, 100, 160 + speed * i, 170], 0.9)
            for i in range(n)]


def test_first_frame_visible():
    t = FaceTracker()
    t.update([([10, 10, 60, 70], 0.7)])
    assert len(t.active()) == 1, "첫 프레임부터 가려야 한다"


def test_tracker_follows_fast_motion_with_small_lag():
    t = FaceTracker()
    speed = 20
    err = []
    for i, det in enumerate(_moving_dets(30, speed)):
        t.update([det])
        tr = t.active()[0]
        err.append(abs(tr.box[0] - det[0][0]))
    assert max(err[5:]) < 6, err


def test_tracker_holds_big_jump_until_confirmed():
    t = FaceTracker()
    t.update([([100, 100, 160, 170], 0.9)])
    # 한 프레임만 30px 아래로 튄 검출: 바로 확정하지 않고 합집합으로 가린다.
    t.update([([100, 130, 160, 200], 0.9)])
    tr = t.active()[0]
    assert list(tr.committed_box) == [100, 100, 160, 170]
    assert list(tr.box) == [100, 100, 160, 200]
    assert tr.pending_box is not None
    # 다음 프레임에 같은 위치가 다시 오면 그때 확정한다.
    t.update([([100, 131, 160, 201], 0.9)])
    tr = t.active()[0]
    assert list(tr.committed_box) == [100, 131, 160, 201]
    assert tr.pending_box is None
    assert list(tr.box) == [100, 131, 160, 201]


def test_tracker_discards_single_frame_false_jump():
    t = FaceTracker()
    t.update([([100, 100, 160, 170], 0.9)])
    t.update([([100, 160, 160, 230], 0.9)])
    assert t.active()[0].pending_box is not None
    # 원래 위치로 돌아오면 후보를 버리고 기존 박스를 유지한다.
    t.update([([100, 101, 160, 171], 0.9)])
    tr = t.active()[0]
    assert tr.pending_box is None
    assert list(tr.committed_box) == [100, 101, 160, 171]


def test_plate_tracker_holds_big_jump_until_confirmed():
    from src.plates import PlateTracker
    t = PlateTracker(max_miss=5)
    t.update([[100, 100, 160, 130]])
    t.update([[100, 120, 160, 150]])
    tr = t.tracks[1]
    assert list(tr.committed_box) == [100, 100, 160, 130]
    assert list(tr.box) == [100, 100, 160, 150]
    t.update([[100, 121, 160, 151]])
    tr = t.tracks[1]
    assert tr.pending_box is None
    assert list(tr.committed_box) == [100, 121, 160, 151]


def test_dog_face_is_not_mosaicked():
    # CC0 Boxer-dog.jpg (Karen Arnold, Wikimedia Commons)
    path = os.path.join(ROOT, "tests", "fixtures", "boxer_dog.jpg")
    frame = cv2.imread(path)
    assert frame is not None, path
    assert detect_faces(frame) == [], "강아지 얼굴을 사람 얼굴로 검출함"


def test_hand_and_background_have_no_face():
    frame = np.full((480, 640, 3), (60, 70, 80), np.uint8)
    skin = (120, 160, 210)
    cv2.ellipse(frame, (320, 300), (80, 110), 0, 0, 360, skin, -1)
    for dx in (-60, -30, 0, 30, 60):
        cv2.line(frame, (320 + dx, 260), (320 + dx, 110),
                 skin, 24, cv2.LINE_AA)
    assert detect_faces(frame) == [], "손 모양을 얼굴로 검출함"

    rng = np.random.default_rng(7)
    bg = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    for i in range(0, 640, 40):
        cv2.line(bg, (i, 0), (i, 480), (255, 255, 255), 2)
    for j in range(0, 480, 40):
        cv2.line(bg, (0, j), (640, j), (255, 255, 255), 2)
    assert detect_faces(bg) == [], "배경 무늬를 얼굴로 검출함"


def test_person_gate_drops_ambiguous_face_outside_person():
    fake = [([400, 200, 500, 300], 0.55)]
    persons = [[0, 0, 200, 600]]
    out = filter_faces_by_person(
        fake, persons, frame_area=1280 * 720)
    assert out == [], "사람 밖의 애매한 얼굴 검출을 거르지 못함"


def test_east_refines_synthetic_address():
    from src.texts import _model_path, refine_box_with_text
    if not os.path.exists(_model_path()):
        return  # EAST 가중치는 CI에 포함되지 않는다.
    frame = np.full((400, 800, 3), 255, np.uint8)
    cv2.putText(frame, "123 MAIN STREET", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA)
    out = refine_box_with_text(frame, [20, 30, 700, 160])
    assert out is not None, "EAST가 합성 주소 글자를 찾지 못함"
    assert 20 <= out[0] and out[2] <= 700, out
    assert out[2] - out[0] < 680, f"박스가 정제되지 않음: {out}"


def test_tracker_extrapolates_on_miss():
    t = FaceTracker()
    for det in _moving_dets(10, 10):
        t.update([det])
    before = t.active()[0].box[0]
    t.update([])
    after = t.active()[0].box[0]
    assert after > before, "검출이 끊겨도 진행 방향으로 이동해야 한다"


def test_expand_grows_with_motion():
    a = expand_face_box([100, 100, 160, 170], 1000, 1000)
    b = expand_face_box([100, 100, 160, 170], 1000, 1000, motion=20)
    assert b[0] < a[0] and b[2] > a[2]


def test_poly_mask_matches_polygon():
    import importlib
    if "ultralytics" not in sys.modules:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            stub = types.ModuleType("ultralytics")
            stub.YOLO = object
            sys.modules["ultralytics"] = stub
    pl = importlib.import_module("src.pipeline")
    poly = np.array([[20, 20], [80, 20], [80, 90], [20, 90]], np.float32)
    m = pl._poly_roi_mask(poly, 10, 10, 100, 100, grow=1)
    assert m.shape == (90, 90)
    assert m[50 - 10, 50 - 10] == 255      # 안쪽
    assert m[2, 2] == 0                    # 밖
    assert m[(15 - 10), (50 - 10)] == 0    # 폴리곤 위쪽 바깥


def _fake_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = {}
    sys.modules.setdefault("streamlit", st)


def test_login_lock_persists_across_sessions():
    _fake_streamlit()
    tmp = tempfile.mkdtemp()
    old_data = os.environ.get("MOSAIC_DATA_DIR")
    os.environ["MOSAIC_DATA_DIR"] = tmp
    try:
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        import src.auth as auth
        importlib.reload(auth)
        auth.DB_PATH = os.path.join(tmp, "users.db")
        assert auth.register("alice_01", "password123")[0]
        for _ in range(auth.MAX_FAILS):
            auth.record_login_result("alice_01", False)
        assert auth.lock_remaining("alice_01") > 0
        auth.record_login_result("bob_01", True)
        assert auth.lock_remaining("bob_01") == 0
    finally:
        if old_data is None:
            os.environ.pop("MOSAIC_DATA_DIR", None)
        else:
            os.environ["MOSAIC_DATA_DIR"] = old_data


def test_invite_code():
    _fake_streamlit()
    import src.auth as auth
    os.environ["MOSAIC_INVITE_CODE"] = "s3cret"
    assert auth.invite_required()
    assert auth.check_invite("s3cret") and not auth.check_invite("x")
    del os.environ["MOSAIC_INVITE_CODE"]
    assert auth.check_invite("anything")


def test_region_flow_tracks_moving_texture():
    import cv2
    from src.flow import RegionFlow
    rng = np.random.default_rng(0)
    patch = (rng.random((40, 90)) * 255).astype(np.uint8)
    flow = RegionFlow()
    anchors = [{"xyxy": [100, 100, 190, 140]}]
    last_err = 0
    for i in range(30):
        bg = np.full((300, 400), 128, np.uint8)
        x = 100 + 4 * i
        bg[100:140, x:x + 90] = patch
        frame = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        flow.begin(frame)
        out = flow.track("plate", anchors)
        flow.end()
        last_err = abs(out[0]["xyxy"][0] - x)
    assert last_err <= 4, last_err


def test_backfill_and_smooth():
    from src.offline import backfill_and_smooth
    frames = [dict() for _ in range(20)]
    for i in range(8, 20):
        frames[i][1] = [100 + i, 50, 160 + i, 120]
    out = backfill_and_smooth(frames, cuts=[], back=5, win=2)
    assert 1 in out[3] and 1 in out[7], "첫 등장 전 프레임이 채워져야 한다"
    assert 1 not in out[2]
    out2 = backfill_and_smooth(frames, cuts=[6], back=5, win=2)
    assert 1 not in out2[5] and 1 in out2[6]
    for i in range(8, 20):
        b, r = out[i][1], frames[i][1]
        assert b[0] <= r[0] and b[2] >= r[2]


def test_union_smoothing_leads_fast_motion():
    from src.offline import backfill_and_smooth
    frames = [{1: [10, 10, 20, 20]}, {1: [20, 10, 30, 20]},
              {1: [40, 10, 50, 20]}]
    out = backfill_and_smooth(frames, [], back=0, win=1)
    assert out[1][1][0] <= 10 and out[1][1][2] >= 50


def test_head_fallback_is_head_sized_not_full_width():
    from src.pipeline import _head_fallback_box, _faces_in_person
    box = [50, 50, 450, 250]
    hb = _head_fallback_box(box)
    assert (hb[2] - hb[0]) < 0.5 * (box[2] - box[0])
    assert _faces_in_person([[100, 100, 150, 160]], box)
    assert not _faces_in_person([[900, 900, 950, 960]], box)


def test_bridge_gaps_fills_id_switch():
    from src.offline import bridge_gaps
    fr = [{1: [0, 0, 20, 20]}, {1: [10, 0, 30, 20]}, {}, {},
          {2: [40, 0, 60, 20]}, {2: [50, 0, 70, 20]}]
    out = bridge_gaps(fr)
    assert 1 in out[2] and 1 in out[3]
    assert out[2][1][0] < out[3][1][0] < 40
    out2 = bridge_gaps(fr, cuts=[3])
    assert 1 not in out2[2]


def test_tracker_follows_flow_on_miss():
    import cv2
    from src.faces import FaceTracker
    from src.flow import shift_box
    rng = np.random.default_rng(0)
    tex = (rng.random((200, 300)) * 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (5, 5), 0)
    big = np.zeros((300, 500), np.uint8)
    big[50:250, 50:350] = tex
    g0 = big.copy()
    g1 = np.roll(big, 7, axis=1)
    tr = FaceTracker()
    tr.update([([100, 100, 200, 200], 0.9)])
    tr.update([], flow_fn=lambda b: shift_box(g0, g1, b))
    t = list(tr.tracks.values())[0]
    assert 5 <= t.box[0] - 100 <= 9


def test_mosaic_strength_increases_block_size():
    from src.render import mosaic_block_size
    weak = mosaic_block_size(100, 100, 18)
    normal = mosaic_block_size(100, 100, 30)
    strong = mosaic_block_size(100, 100, 44)
    assert weak < normal < strong, (weak, normal, strong)
    # 기본 강도에서 얼굴 특징점이 다시 검출되지 않을 만큼 큼직해야 한다.
    assert normal >= 18


def test_nearest_track_values_fills_interpolated_frames():
    from src.offline import nearest_track_values
    frames = [{1: 12.0}, {}, {}, {1: 24.0}, {}]
    out = nearest_track_values(frames, 5)
    assert out[0][1] == 12.0 and out[1][1] == 12.0
    assert out[2][1] in (12.0, 24.0)
    assert out[3][1] == 24.0 and out[4][1] == 24.0
    cut = nearest_track_values(frames, 5, cuts=[2])
    # 컷 이전 프레임은 이전 각도를 유지하고, 컷 경계를 넘는 보간은 하지 않는다.
    assert cut[1][1] == 12.0
    # 컷 프레임부터는 새 장면의 다음 관측값을 쓴다(이전 값을 끌고 오지 않음).
    assert cut[2][1] == 24.0
    assert cut[3][1] == 24.0 and cut[4][1] == 24.0


def test_history_is_per_user_and_recovers_orphan_files():
    import tempfile
    from src.history import JobHistory
    d = tempfile.mkdtemp()
    h = JobHistory(directory=d, max_rows=2)
    for i in range(3):
        h.add("a.mp4", "p", "face", False, f"alice--a_{i}.mp4", 1.0,
              user="alice")
    h.add("b.mp4", "p", "face", False, "bob--b_0.mp4", 1.0, user="bob")
    open(os.path.join(d, "alice--orphan.mp4"), "wb").close()
    rows = h.all_rows("alice")
    names = {r["file"] for r in rows}
    assert "alice--orphan.mp4" in names
    assert all(n.startswith("alice--") for n in names)
    assert len([r for r in h.all_rows("alice")
                if r["file"].startswith("alice--a_")]) == 2
    assert {r["file"] for r in h.all_rows("bob")} == {"bob--b_0.mp4"}


def test_history_migrates_legacy_json_once():
    from src.history import JobHistory
    d = tempfile.mkdtemp()
    legacy = [
        {"at": "01-01 00:00", "user": "alice", "file": "alice--old.mp4",
         "src": "old.mp4", "target": "face", "sec": 1.0},
        {"at": "01-01 00:01", "user": "bob", "file": "bob--old.mp4",
         "src": "old.mp4", "target": "face", "sec": 2.0},
    ]
    with open(os.path.join(d, "jobs.json"), "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False)
    h = JobHistory(directory=d, max_rows=5)
    assert {r["file"] for r in h.all_rows()} == {
        "alice--old.mp4", "bob--old.mp4"}
    assert os.path.exists(os.path.join(d, "jobs.json.migrated"))
    h2 = JobHistory(directory=d, max_rows=5)
    assert len(h2.all_rows()) == 2, "마이그레이션 후 중복 삽입됨"


def test_render_gate_reports_waiters():
    from src.render_queue import RenderGate
    g = RenderGate()
    g.enter()
    assert g.waiting() == 0
    g.enter()
    assert g.waiting() == 1
    g.leave_waiting()
    assert g.waiting() == 0
    with g:
        pass


def test_retention_purges_old_and_quota_keeps_db():
    from src.retention import RetentionPolicy
    d = tempfile.mkdtemp()
    for name, content in (("jobs.db", b"db"), ("jobs.db-wal", b"wal")):
        with open(os.path.join(d, name), "wb") as f:
            f.write(content)
    now = time.time()
    old = os.path.join(d, "old.mp4")
    mid = os.path.join(d, "mid.mp4")
    new = os.path.join(d, "new.mp4")
    for p, size in ((old, 10), (mid, 20), (new, 30)):
        with open(p, "wb") as f:
            f.write(b"x" * size)
    os.utime(old, (now - 25 * 3600, now - 25 * 3600))
    os.utime(mid, (now - 60, now - 60))
    os.utime(new, (now, now))
    p = RetentionPolicy(d, max_hours=24, max_files=1, max_bytes=0)
    removed = p.enforce()
    assert removed == 2, removed
    assert os.path.exists(old) is False and os.path.exists(mid) is False
    assert os.path.exists(new) is True
    assert os.path.exists(os.path.join(d, "jobs.db")) is True
    assert os.path.exists(os.path.join(d, "jobs.db-wal")) is True


def test_address_eval_matching_and_summary():
    from scripts.address_eval import match_boxes, summarize
    preds = [[0, 0, 100, 50]]
    gts = [[0, 0, 100, 50], [200, 200, 250, 250]]
    matched_p, matched_g = match_boxes(preds, gts, iou_thresh=0.3)
    assert matched_p == {0}
    assert matched_g == {0}
    rows = [{"tag": "x", "pred": 1, "gt": 2, "tp": 1, "tp_gt": 1,
             "fp": 0, "fn": 1}]
    s = summarize(rows)
    assert s["precision"] == 1.0, s
    assert s["recall"] == 0.5, s


def test_render_queue_persists_and_claims_one_job():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"), max_parallel=1)
    items = [{"category": "face", "type": "person",
              "samples": {"0": [[1, 2, 3, 4]]},
              "thumb": np.zeros((2, 2, 3), np.uint8)}]
    job1 = q.enqueue(
        user="alice", input_path=src, output_path=os.path.join(d, "out1.mp4"),
        items=items, selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q", src_name="in.mp4",
        preset_name="얼굴만")
    job2 = q.enqueue(
        user="bob", input_path=src, output_path=os.path.join(d, "out2.mp4"),
        items=items, selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q", src_name="in.mp4",
        preset_name="얼굴만")
    assert q.position(job2) == 2
    assert q.position(job1, "alice") == 1
    assert q.position(job2, "bob") == 1, "다른 사용자의 작업이 순번에 섞임"
    assert len(q.active_jobs("alice")) == 1
    assert q.active_jobs("alice")[0]["status"] == "queued"
    claimed = q.claim()
    assert claimed and claimed["id"] == job1
    assert "thumb" not in claimed["payload"]
    assert q.claim() is None, "동시에 두 작업을 잡으면 안 된다"
    q.finish(job1, "done", summary="1개 항목")
    assert q.get(job1)["status"] == "done"
    assert q.get(job2)["status"] == "queued"
    q.recover_running()
    assert q.get(job1)["status"] == "done"


def test_queue_runs_different_users_in_parallel_but_one_per_user():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"), max_parallel=2)

    def add(user):
        return q.enqueue(
            user=user, input_path=src,
            output_path=os.path.join(d, f"{user}.mp4"), items=[],
            selected_cats=["face"], settings=MosaicSettings(),
            model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
            preset="veryfast", query_raw="q")

    alice1 = add("alice")
    bob1 = add("bob")
    alice2 = add("alice")

    assert q.claim()["id"] == alice1
    # 다른 계정 작업은 같은 계정의 running 작업에 막히지 않는다.
    assert q.claim()["id"] == bob1
    # 동시 실행 상한(2)에 도달했고, alice의 두 번째 작업은 자기 차례를 기다린다.
    assert q.claim() is None
    assert q.position(alice2, "alice") == 2

    q.finish(alice1, "done")
    # alice1이 끝나면 alice2가 들어갈 수 있다. bob은 아직 running이다.
    assert q.claim()["id"] == alice2
    assert q.position(alice2, "alice") == 1


def test_queue_worker_lease_prevents_two_processes():
    from src.jobqueue import RenderQueue
    d = tempfile.mkdtemp()
    q1 = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                     input_dir=os.path.join(d, "inputs"))
    q2 = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                     input_dir=os.path.join(d, "inputs"))
    assert q1.acquire_worker("one", ttl=60) is True
    assert q2.acquire_worker("two", ttl=60) is False
    assert q1.heartbeat_worker("one") is True
    assert q2.heartbeat_worker("two") is False
    # heartbeat가 만료되면 다른 프로세스가 이어받을 수 있다.
    assert q2.acquire_worker("two", ttl=0) is True
    assert q1.heartbeat_worker("one") is False


def test_queue_detects_stale_running_job():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    job_id = q.enqueue(
        user="alice", input_path=src,
        output_path=os.path.join(d, "out.mp4"), items=[],
        selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q")
    job = q.claim()
    assert job["id"] == job_id
    assert q.stale_running(60) is None
    with q._db() as c:
        c.execute("UPDATE render_jobs SET updated=? WHERE id=?",
                  (time.time() - 120, job_id))
    stale = q.stale_running(60)
    assert stale and stale["id"] == job_id


def test_queue_five_accounts_and_per_account_order():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"), max_parallel=5)

    def add(user):
        return q.enqueue(
            user=user, input_path=src,
            output_path=os.path.join(d, f"{user}.mp4"), items=[],
            selected_cats=["face"], settings=MosaicSettings(),
            model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
            preset="veryfast", query_raw="q")

    ids = {u: add(u) for u in ("u1", "u2", "u3", "u4", "u5")}
    claimed = [q.claim() for _ in range(5)]
    assert {c["user"] for c in claimed} == {"u1", "u2", "u3", "u4", "u5"}
    assert q.claim() is None, "5개 계정 상한을 넘어 실행됨"

    u1_second = add("u1")
    u6 = add("u6")
    assert q.position(u1_second, "u1") == 2
    assert q.position(u6, "u6") == 1, "다른 계정 작업이 내 순번에 섞임"

    q.finish(ids["u1"], "done")
    nxt = q.claim()
    assert nxt["id"] == u1_second, "같은 계정의 다음 작업이 먼저 시작돼야 함"
    assert q.claim() is None
    q.finish(u1_second, "done")
    nxt = q.claim()
    assert nxt["id"] == u6


def test_queue_default_parallel_is_three(monkeypatch):
    from src.jobqueue import RenderQueue
    monkeypatch.delenv("MOSAIC_MAX_PARALLEL", raising=False)
    d = tempfile.mkdtemp()
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    assert q.max_parallel == 3


def test_queue_status_snapshot_and_audit():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"), max_parallel=3)
    job_id = q.enqueue(
        user="alice", input_path=src,
        output_path=os.path.join(d, "out.mp4"), items=[],
        selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q")
    snap = q.status_snapshot()
    assert snap["counts"]["queued"] == 1
    assert snap["max_parallel"] == 3
    assert snap["active"][0]["id"] == job_id
    q.set_audit(job_id, {"checked": 10, "suspicious": 0, "fixable": 0})
    import json
    assert json.loads(q.get(job_id)["audit"])["checked"] == 10


def test_thread_local_models_are_isolated(monkeypatch):
    from src import models
    seen = {}
    monkeypatch.setattr(models.ModelStore, "get",
                        lambda self, name: self)

    def worker(key):
        seen[key] = (models.get_thread_model("x"),
                     models.get_thread_model("x"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen[0][0] is seen[0][1]
    assert seen[1][0] is seen[1][1]
    assert seen[0][0] is not seen[1][0], "스레드가 같은 YOLO 모델을 공유함"


def test_queue_worker_completes_job_without_model():
    from unittest.mock import patch
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings

    class FakePlan:
        @classmethod
        def from_items(cls, *args, **kwargs):
            return cls()

        def render(self, input_path, output_path, **kwargs):
            with open(output_path, "wb") as f:
                f.write(b"x" * 2048)
            return {"_faces": 1, "_regions": 0}

    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    job_id = q.enqueue(
        user="alice", input_path=src, output_path=os.path.join(d, "out.mp4"),
        items=[], selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q", src_name="in.mp4",
        preset_name="얼굴만", person_total=1)
    job = q.claim()
    with patch("src.jobqueue.RenderPlan", FakePlan), \
            patch("src.jobqueue.H.add", lambda **kwargs: None):
        q.run_job(job)
    row = q.get(job_id)
    assert row["status"] == "done", row
    assert os.path.exists(row["output_path"])
    assert not os.path.exists(job["input_path"]), "입력 임시 파일이 남음"


def test_east_text_regions_returns_list():
    from src.texts import detect_text_regions
    frame = np.zeros((120, 160, 3), np.uint8)
    out = detect_text_regions(frame)
    assert isinstance(out, list)


def test_render_queue_cancel_queued_and_running():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))

    def add():
        return q.enqueue(
            user="alice", input_path=src,
            output_path=os.path.join(d, "out.mp4"), items=[],
            selected_cats=["face"], settings=MosaicSettings(),
            model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
            preset="veryfast", query_raw="q")

    queued = add()
    queued_input = q.get(queued)["input_path"]
    assert os.path.exists(queued_input)
    assert q.cancel(queued, "alice") is True
    assert q.get(queued)["status"] == "cancelled"
    assert not os.path.exists(queued_input), "취소된 queued 입력 파일이 남음"
    assert q.claim() is None

    running = add()
    job = q.claim()
    assert job["id"] == running
    assert q.cancel(running, "alice") is True
    assert q.is_cancel_requested(running) is True
    q.finish(running, "cancelled", error="사용자 취소")
    assert q.get(running)["status"] == "cancelled"


def test_failed_queue_job_cleans_input_and_result():
    from src.jobqueue import RenderQueue, RenderCancelled
    from src.render_service import MosaicSettings

    class CancelPlan:
        @classmethod
        def from_items(cls, *args, **kwargs):
            return cls()

        def render(self, input_path, output_path, **kwargs):
            raise RenderCancelled()

    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    job_id = q.enqueue(
        user="alice", input_path=src,
        output_path=os.path.join(d, "out.mp4"), items=[],
        selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q")
    job = q.claim()
    from unittest.mock import patch
    with patch("src.jobqueue.RenderPlan", CancelPlan), \
            patch("src.jobqueue.H.add", lambda **kwargs: None):
        q.run_job(job)
    assert q.get(job_id)["status"] == "cancelled"
    assert not os.path.exists(job["input_path"])


def test_failed_queue_job_can_retry():
    from unittest.mock import patch
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings

    class FailPlan:
        @classmethod
        def from_items(cls, *args, **kwargs):
            return cls()

        def render(self, input_path, output_path, **kwargs):
            raise RuntimeError("boom")

    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    job_id = q.enqueue(
        user="alice", input_path=src,
        output_path=os.path.join(d, "out.mp4"), items=[],
        selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q")
    job = q.claim()
    with patch("src.jobqueue.RenderPlan", FailPlan), \
            patch("src.jobqueue.H.add", lambda **kwargs: None):
        q.run_job(job)
    failed = q.get(job_id)
    assert failed["status"] == "failed"
    assert os.path.exists(failed["input_path"]), "실패 입력이 재시도용으로 남지 않음"
    new_id = q.retry(job_id, "alice")
    assert new_id and new_id != job_id
    assert q.get(new_id)["status"] == "queued"
    assert q.get(job_id)["status"] == "retried"
    assert q.get(new_id)["input_path"] == failed["input_path"]


def test_ui_groups_and_signboard_aliases():
    from src.config import CANDS, UI_GROUPS, canon_category
    assert "passenger" not in CANDS
    assert "signboard" in CANDS
    assert canon_category("간판") == "signboard"
    assert canon_category("store sign") == "signboard"
    assert canon_category("택배 송장") == "id_card"
    assert set(UI_GROUPS) == {"얼굴", "번호판", "텍스트·문서", "간판"}
    assert "signboard" in UI_GROUPS["간판"]
    assert "name_tag" in UI_GROUPS["텍스트·문서"]


def test_queue_recovers_stale_running_on_worker_start():
    from src.jobqueue import RenderQueue
    from src.render_service import MosaicSettings
    d = tempfile.mkdtemp()
    src = os.path.join(d, "in.mp4")
    with open(src, "wb") as f:
        f.write(b"video")
    q = RenderQueue(db_path=os.path.join(d, "jobs.db"),
                    input_dir=os.path.join(d, "inputs"))
    job_id = q.enqueue(
        user="alice", input_path=src,
        output_path=os.path.join(d, "out.mp4"), items=[],
        selected_cats=["face"], settings=MosaicSettings(),
        model_name="yolo11n-seg.pt", imgsz=640, stride=15, conf=0.2,
        preset="veryfast", query_raw="q")
    assert q.claim()["id"] == job_id
    q.recover_running()
    assert q.get(job_id)["status"] == "queued"
    assert os.path.exists(q.get(job_id)["input_path"])


def test_face_rotation_maps_boxes_back():
    from src.faces import FaceBox, _map_box_back
    W, H = 640, 480
    orig = (100, 50, 200, 150)
    rotated = {
        90: ([H - 1 - orig[3], orig[0], H - 1 - orig[1], orig[2]], 90),
        180: ([W - 1 - orig[2], H - 1 - orig[3],
               W - 1 - orig[0], H - 1 - orig[1]], 180),
        270: ([orig[1], W - 1 - orig[2],
               orig[3], W - 1 - orig[0]], 270),
    }
    for rot, (box, angle) in rotated.items():
        mapped = _map_box_back(FaceBox(box, angle=angle), rot, W, H)
        assert mapped is not None
        assert [int(round(float(v))) for v in mapped[:4]] == list(orig), rot


def test_merge_face_dets_keeps_higher_score():
    from src.faces import FaceBox, _merge_face_dets
    a = (FaceBox([0, 0, 40, 40]), 0.6)
    b = (FaceBox([1, 1, 41, 41]), 0.9)
    merged = _merge_face_dets([a], [b])
    assert len(merged) == 1
    assert merged[0][1] == 0.9


def test_poly_covers_box_detects_partial_face_mask():
    from src.pipeline import _poly_covers_box
    box = [0, 0, 100, 100]
    half = np.array([[0, 0], [50, 0], [50, 100], [0, 100]], np.int32)
    full = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.int32)
    assert not _poly_covers_box(half, box, 0.8)
    assert _poly_covers_box(full, box, 0.8)


def test_face_mesh_falls_back_without_model():
    from src.face_mesh import FaceMeshRefiner
    ref = FaceMeshRefiner(model_path="/nonexistent/model.task")
    assert not ref.available()
    assert ref.polygon_for_box(np.zeros((100, 100, 3), np.uint8),
                               [10, 10, 50, 50]) is None


def test_face_polygon_render_matches_shape():
    from src.render import render_face_polygon
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    before = frame.copy()
    # 왼쪽이 좁고 오른쪽이 넓은 비대칭 얼굴형.
    poly = np.array([[80, 20], [130, 30], [140, 100], [110, 170],
                     [70, 170], [50, 100]], np.int32)
    assert render_face_polygon(frame, poly, style="pixel", blocks=30)
    changed = np.any(frame != before, axis=2)
    assert changed.any()
    # 바깥(모서리)은 그대로여야 한다.
    assert np.array_equal(frame[0, 0], before[0, 0])


def test_plate_detector_disabled_returns_empty():
    os.environ["MOSAIC_PLATE_DET"] = "0"
    try:
        from src.plates import detect_plates
        out = detect_plates(np.zeros((100, 100, 3), np.uint8))
        assert out == []
    finally:
        os.environ.pop("MOSAIC_PLATE_DET", None)


def test_sample_boxes_backfills_before_first_anchor():
    from src.pipeline import _sample_boxes
    samples = {5: [{"xyxy": [10, 10, 50, 30], "due": 100}]}
    # 첫 앵커(100) 이전 90프레임은 다음 앵커로 미리 가린다.
    got = _sample_boxes(samples, 0, frame_idx=90, stride=15,
                        interpolate=True)
    assert got and got[0]["xyxy"] == [10, 10, 50, 30]
    # 너무 이전(70프레임 전)은 가리지 않는다.
    assert _sample_boxes(samples, 0, frame_idx=0, stride=15,
                         interpolate=True) == []


def test_plate_filter_by_vehicles():
    from src.plates import filter_by_vehicles
    plates = [[100, 100, 180, 130], [500, 500, 560, 520]]
    vehicles = [[50, 50, 300, 300]]
    kept = filter_by_vehicles(plates, vehicles)
    assert kept == [[100, 100, 180, 130]]
    # 차량 검출 실패 시 모두 살린다(노출 방지).
    assert filter_by_vehicles(plates, []) == plates
    assert filter_by_vehicles([], vehicles) == []


def test_texts_disabled_returns_none():
    import numpy as np
    os.environ["MOSAIC_TEXT_DET"] = "0"
    try:
        from src.texts import refine_box_with_text
        assert refine_box_with_text(
            np.zeros((100, 100, 3), np.uint8), [10, 10, 50, 50]) is None
    finally:
        os.environ.pop("MOSAIC_TEXT_DET", None)


def test_small_face_crops_empty_without_persons():
    import numpy as np
    from src.faces import detect_faces_in_person_crops
    frame = np.zeros((200, 200, 3), np.uint8)
    assert detect_faces_in_person_crops(frame, []) == []
