"""MosaicJob.run() 동작 불변 검증용 골든 비교 스크립트 (스텁 모델/VLM, 합성 영상).

사용법 (경로 하드코딩 없음, 저장소 위치는 --root 로 받는다):
  python scripts/golden_run.py --root <저장소> --out <결과.json>
  python scripts/golden_run.py --compare <기준.json> <비교.json>

시나리오마다 다음을 해시로 남긴다.
  - 작성기(writer)에 넘어간 모든 출력 프레임의 SHA-256 (인코딩 영향 없음)
  - stats dict 전체, progress_fn 호출 열, model.track 호출 인자와 횟수,
    예외 발생 시 예외 종류/메시지
실제 ultralytics / GPU / Gemini / 실제 영상은 쓰지 않는다. 그 경로는 검증하지 못한다.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import types

import cv2
import numpy as np

W, H, N, FPS = 320, 240, 40, 10
ROOT_FOR_NORM = [""]


def _norm(o):
    """키를 문자열로 바꿔 정렬 가능한 JSON 값으로 만든다."""
    if isinstance(o, dict):
        return {str(k): _norm(v) for k, v in sorted(
            o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_norm(v) for v in o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


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
        self.id = None if ids is None else _T(np.asarray(ids, np.int64))

    def __len__(self):
        return len(self.xyxy.a)


def _rect_poly(x1, y1, x2, y2):
    # 사람처럼 보이도록 약간 둥근 다각형
    cx = (x1 + x2) / 2.0
    return np.array([[cx, y1], [x2, y1 + 25], [x2, y2], [x1, y2],
                     [x1, y1 + 25]], np.float32)


class StubModel:
    """사람 A(id 1→3, 14~16 검출 끊김), B(id 2, 10~19 마스크 없음),
    C(A 안에 포함된 작은 중복 박스), D(id 없음, 25~30)."""

    def __init__(self, log):
        self.log = log

    def track(self, source=None, **kw):
        if os.environ.get("GOLDEN_TRACK_FAIL") == "1":
            raise RuntimeError("tracker crashed")
        self.log.append(sorted((k, repr(v).replace(ROOT_FOR_NORM[0], '<ROOT>'))
                               for k, v in kw.items()))
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, frm = cap.read()
            if not ok or os.environ.get("GOLDEN_EMPTY") == "1":
                break
            boxes, ids, polys = [], [], []
            if not (14 <= i <= 16):
                ax = 20 + 5 * i
                a = [ax, 40, ax + 60, 190]
                boxes.append(a)
                ids.append(1 if i < 14 else 3)
                polys.append(_rect_poly(*a))
                c = [ax + 15, 60, ax + 45, 120]   # A 안에 포함 → filter_contained
                boxes.append(c)
                ids.append(7)
                polys.append(_rect_poly(*c))
            bx = 200 - 3 * i
            b = [bx, 50, bx + 55, 200]
            boxes.append(b)
            ids.append(2)
            # i==5: 점 2개짜리 퇴화 폴리곤(빈 마스크 경로)
            polys.append(_rect_poly(*b)[:2] if i == 5 else _rect_poly(*b))
            if 25 <= i <= 30:
                d = [10, 60, 70, 200]
                boxes.append(d)
                ids.append(-1)
                polys.append(_rect_poly(*d))
            r = types.SimpleNamespace(orig_img=frm)
            idarr = np.array(ids)
            if any(v < 0 for v in ids):
                r.boxes = _B(boxes, None)       # id 없는 프레임
            else:
                r.boxes = _B(boxes, idarr)
            if 10 <= i <= 19:
                r.masks = None
            elif i == 8:
                class _BadMasks:   # .xy 접근 시 예외
                    @property
                    def xy(self):
                        raise RuntimeError("bad masks")
                r.masks = _BadMasks()
            elif i == 22:          # 박스 수와 다른 마스크 수
                r.masks = types.SimpleNamespace(xy=polys[:-1])
            else:
                r.masks = types.SimpleNamespace(xy=polys)
            i += 1
            yield r
        cap.release()


class RecWriter:
    """FFmpegFrameWriter 대체: 프레임을 그대로 해시한다."""
    instances = []

    def __init__(self, output_path, width, height, fps, audio_src=None,
                 preset="veryfast", **kw):
        if os.environ.get("GOLDEN_WRITER_FAIL") == "1":
            raise OSError("ffmpeg unavailable")
        self.h = hashlib.sha256()
        self.n = 0
        self.closed = self.aborted = False
        self.meta = (os.path.basename(output_path), width, height, round(fps, 3),
                     preset)
        RecWriter.instances.append(self)

    def write(self, frame):
        self.h.update(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


VLM_BOXES = [
    {"xyxy": [20, 40, 80, 80], "conf": 0.9, "category": "face",
     "phrase": "face", "region_type": "person"},
    {"xyxy": [30, 100, 60, 120], "conf": 0.8, "category": "name_tag",
     "phrase": "name tag", "region_type": "person"},
    {"xyxy": [240, 170, 300, 200], "conf": 0.9, "category": "license_plate",
     "phrase": "plate", "region_type": "region"},
    {"xyxy": [10, 200, 100, 230], "conf": 0.7, "category": "address",
     "phrase": "sign", "region_type": "region"},
]


def make_video(path):
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(3)
    base = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    for i in range(N):
        f = np.roll(base, 2 * i, axis=1).copy()
        if i >= 30:
            f = 255 - f          # 장면 전환(얼굴 트랙 reset 경로)
        wr.write(f)
    wr.release()


SCENARIOS = [
    # (이름, MosaicJob 인자, 환경변수, 두 번 실행(캐시 적중 확인) 여부)
    ("face_passenger_upper", dict(targets=["face", "passenger"],
                                  area_mode="upper", mask_all_persons=True),
     {}, False),
    ("face_passenger_faceonlyscope", dict(targets=["face", "passenger"],
                                          area_mode="face_only",
                                          mask_all_persons=True,
                                          person_priors={"name_tag": {}}),
     {}, False),
    ("face_plate_full", dict(targets=["face", "license_plate"],
                             area_mode="full", api_key="k",
                             keyframe_stride=5), {}, False),
    ("face_plate_nametag_vlm", dict(targets=["face", "name_tag", "license_plate",
                                     "address"], area_mode="upper",
                                    api_key="k", keyframe_stride=5,
                                    query_raw="x"), {}, False),
    ("invert_full", dict(targets=["face"], invert=True, api_key="k",
                         keyframe_stride=5), {}, False),
    ("invert_upper_style_blur", dict(targets=["face"], invert=True,
                                     api_key="k", area_mode="upper",
                                     keyframe_stride=5, style="blur"), {}, False),
    ("area_face_only_vlm", dict(targets=["face", "license_plate"],
                                area_mode="face_only", api_key="k",
                                keyframe_stride=5), {}, False),
    ("replay_off", dict(targets=["face", "passenger"], area_mode="upper",
                        mask_all_persons=True), {"MOSAIC_REPLAY": "0"}, False),
    ("replay_off_vlm", dict(targets=["face", "license_plate"],
                            api_key="k", keyframe_stride=5),
     {"MOSAIC_REPLAY": "0"}, False),
    ("cache_miss_then_hit", dict(targets=["face", "license_plate"],
                                 api_key="k", keyframe_stride=5,
                                 use_cache=True), {}, True),
    ("fast_person_only_replay_off", dict(targets=["face"], area_mode="upper",
                                         mask_all_persons=True),
     {"MOSAIC_REPLAY": "0"}, False),
    ("plate_only_no_person", dict(targets=["license_plate"], api_key="k",
                                  keyframe_stride=5), {}, False),
    ("plate_only_vlm_limit0", dict(targets=["license_plate"], api_key="k",
                                   keyframe_stride=5, vlm_call_limit=0),
     {}, False),
    ("vlm_limit2_priors", dict(targets=["face", "name_tag", "license_plate"],
                               api_key="k", keyframe_stride=5, vlm_call_limit=2,
                               region_priors={"license_plate": {
                                   "3": [[240, 170, 300, 200]]}},
                               person_priors={"name_tag": {
                                   "5": [[30, 100, 60, 120]]}}), {}, False),
    ("ultra_invert", dict(targets=["face"], invert=True, ultra=True,
                          api_key="k", keyframe_stride=5), {}, False),
    ("manual_regions", dict(targets=["face", "passenger"], area_mode="upper",
                            mask_all_persons=True,
                            manual_regions=[
                                {"mode": "add", "t0": 0.0, "t1": 1.5,
                                 "box0": [0.1, 0.1, 0.3, 0.3],
                                 "box1": [0.5, 0.1, 0.7, 0.3]},
                                {"mode": "exclude", "t0": 1.0, "t1": 3.0,
                                 "box0": [0.0, 0.8, 1.0, 1.0]}]), {}, False),
    ("vlm_error_generic", dict(targets=["face", "license_plate"], api_key="k",
                               keyframe_stride=5), {"GOLDEN_VLM": "error"},
     False),
    ("vlm_error_quota", dict(targets=["face", "license_plate"], api_key="k",
                             keyframe_stride=5), {"GOLDEN_VLM": "quota"}, False),
    ("writer_fallback_videowriter", dict(targets=["face", "license_plate"],
                                         api_key="k", keyframe_stride=5),
     {"GOLDEN_WRITER_FAIL": "1"}, False),
    ("cache_version_mismatch", dict(targets=["face", "license_plate"],
                                    api_key="k", keyframe_stride=5,
                                    use_cache=True, _cache_pre="v1"), {}, False),
    ("cache_corrupt", dict(targets=["face", "license_plate"], api_key="k",
                           keyframe_stride=5, use_cache=True,
                           _cache_pre="junk"), {}, False),
    ("empty_stream_abort", dict(targets=["face", "name_tag"], api_key="k",
                                keyframe_stride=5,
                                person_priors={"name_tag": {
                                    "5": [[30, 100, 60, 120]]}}),
     {"GOLDEN_EMPTY": "1"}, False),
    ("track_fail_in_record_pass", dict(targets=["face", "passenger"],
                                       area_mode="upper",
                                       mask_all_persons=True),
     {"GOLDEN_TRACK_FAIL": "1"}, False),
    ("ultra_cache_hit", dict(targets=["face", "license_plate"], api_key="k",
                             keyframe_stride=5, use_cache=True, ultra=True),
     {}, True),
    ("no_vlm_key_error", dict(targets=["face", "license_plate"],
                              keyframe_stride=5), {}, False),
]


def run_scenarios(root, workdir):
    sys.path.insert(0, root)
    ROOT_FOR_NORM[0] = root
    if "ultralytics" not in sys.modules:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            stub = types.ModuleType("ultralytics")
            stub.YOLO = object
            sys.modules["ultralytics"] = stub
    import src.pipeline as pl
    from src.select import VLMSelector

    pl.FFmpegFrameWriter = RecWriter
    def fake_detect(self, frame):
        mode = os.environ.get("GOLDEN_VLM")
        if mode == "error":
            raise RuntimeError("temporary failure 503 unavailable")
        if mode == "quota":
            raise RuntimeError("quota exceeded: daily limit reached "
                               "RESOURCE_EXHAUSTED per day")
        return [dict(b) for b in VLM_BOXES]

    VLMSelector.detect_boxes = fake_detect
    track_log = []
    fcount = [0]

    def fake_faces(frame, *a, **k):
        # 사람 A 의 얼굴: 2~12, 22~35 프레임에서만 검출(13~21 은 끊김)
        i = fcount[0]
        fcount[0] += 1
        if i == 20:
            raise RuntimeError("face detector failure")
        if 2 <= i <= 12 or 22 <= i <= 35:
            ax = 20 + 5 * i
            return [([ax + 10, 42, ax + 50, 90], 0.9)]
        return []

    pl.detect_faces = fake_faces
    pl.get_model = lambda name: StubModel(track_log)

    src = os.path.join(workdir, "in.mp4")
    if not os.path.exists(src):
        make_video(src)

    results = {}
    for name, kw, env, twice in SCENARIOS:
        old_env = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            runs = []
            for k in range(2 if twice else 1):
                RecWriter.instances.clear()
                del track_log[:]
                out = os.path.join(workdir, f"{name}_{k}.mp4")
                args = dict(kw)
                pre = args.pop("_cache_pre", None)
                if args.get("use_cache"):
                    args["cache_path"] = os.path.join(
                        workdir, "cache_%s.pkl" % name)
                    if k == 0 and os.path.exists(args["cache_path"]):
                        os.remove(args["cache_path"])
                    if pre == "v1":
                        import pickle
                        with open(args["cache_path"], "wb") as cf:
                            pickle.dump({"version": 1}, cf)
                    elif pre == "junk":
                        with open(args["cache_path"], "wb") as cf:
                            cf.write(b"not a pickle")
                prog = []
                rec = {}
                fcount[0] = 0
                try:
                    job = pl.MosaicJob(src, out, **args)
                    stats = job.run(
                        lambda p, i, t, e: prog.append(
                            (round(p, 6), i, t)))
                    rec["stats"] = _norm(stats)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"
                if os.path.exists(out):
                    hh = hashlib.sha256()
                    cp = cv2.VideoCapture(out)
                    nfr = 0
                    while True:
                        okf, fr = cp.read()
                        if not okf:
                            break
                        hh.update(fr.tobytes())
                        nfr += 1
                    cp.release()
                    rec["decoded_output"] = [nfr, hh.hexdigest()]
                rec["progress"] = prog
                rec["track_calls"] = list(track_log)
                rec["writers"] = [
                    {"meta": list(w.meta), "frames": w.n,
                     "sha": w.h.hexdigest(), "closed": w.closed,
                     "aborted": w.aborted} for w in RecWriter.instances]
                runs.append(rec)
            results[name] = runs
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return results


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    bad = 0
    for name in sorted(set(a) | set(b)):
        if a.get(name) == b.get(name):
            print(f"  SAME  {name}")
        else:
            bad += 1
            print(f"  DIFF  {name}")
            for k, (ra, rb) in enumerate(zip(a.get(name, []), b.get(name, []))):
                for key in sorted(set(ra) | set(rb)):
                    if ra.get(key) != rb.get(key):
                        print(f"        run{k}.{key}")
    print("결과:", "모두 동일" if not bad else f"{bad}개 시나리오 다름")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="저장소 루트(기본: 이 스크립트의 상위 폴더)")
    ap.add_argument("--out", help="결과 JSON 경로")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--workdir", help="합성 영상/출력 임시 폴더(기본: 새 임시 폴더)")
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    root = os.path.abspath(a.root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    work = a.workdir or tempfile.mkdtemp(prefix="golden_")
    os.makedirs(work, exist_ok=True)
    res = run_scenarios(root, work)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    for name, runs in res.items():
        for k, r in enumerate(runs):
            if "error" in r:
                print(f"{name}[{k}]: ERROR {r['error']}")
            else:
                ws = r["writers"]
                print(f"{name}[{k}]: frames={[w['frames'] for w in ws]} "
                      f"track_calls={len(r['track_calls'])} "
                      f"cats={r['stats'].get('_cats')}")


if __name__ == "__main__":
    main()
