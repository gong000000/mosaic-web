"""2패스 파이프라인. 1패스는 점수 수집, 투표로 확정. 단일 임계값 없음."""
import cv2
import numpy as np
import time
from collections import defaultdict
from .models import get_model


def _reset_tracker(model):
    """pass 사이에 추적기 상태 초기화. 같은 파일 2회 추적 시 ID 어긋남 방지."""
    try:
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            delattr(pred, "trackers")
    except Exception:
        pass
from .detect import filter_contained, big_enough
from .render import render_person_strict, mux_audio_h264, apply_area_limit, cleanup_tmp
from .select import HybridSelector


class MosaicJob:
    """모자이크 작업 1건. 설정+모델+분류기를 객체에 묶는다."""

    def __init__(self, input_path, output_path, target="green shirt", invert=False, conf=0.15, margin=0.04, model_name="yolo11m-seg.pt", imgsz=960, blocks=30, style="pixel", blur=61, ultra=True, area_mode="full", ref_hists=None, ref_bank=None, ref_thresh=0.82, ref_margin=0.05, encode_preset="veryfast", keyframe_stride=3, query_raw="", world_mode="auto", world_conf=None, world_imgsz=None):
        self.input_path = input_path
        self.output_path = output_path
        self.target = target
        self.invert = invert
        self.conf = conf
        self.margin = margin
        self.model_name = model_name
        self.imgsz = imgsz
        self.blocks = blocks
        self.style = style
        self.blur = blur
        self.ultra = ultra
        self.area_mode = area_mode
        self.ref_hists = ref_hists
        self.ref_bank = ref_bank
        self.ref_thresh = ref_thresh
        self.ref_margin = ref_margin
        self.encode_preset = encode_preset
        self.keyframe_stride = keyframe_stride
        self.query_raw = query_raw or (target if isinstance(target, str) else "")
        self.world_mode = world_mode
        self.world_conf = world_conf
        self.world_imgsz = world_imgsz
        from .models import get_model as _gm
        from .clothing import ClothingClassifier as _CC
        self.model = _gm(model_name)
        self.classifier = _CC(margin=margin)

    def run(self, progress_fn=None):
        input_path, output_path = self.input_path, self.output_path
        target = self.target
        invert = self.invert
        conf = self.conf
        model_name = self.model_name
        imgsz = self.imgsz
        blocks = self.blocks
        style = self.style
        blur = self.blur
        ultra = self.ultra
        area_mode = self.area_mode
        ref_hists = self.ref_hists
        ref_bank = self.ref_bank
        if ref_bank is None and ref_hists:
            ref_bank = [(h, True) for h in ref_hists]
        ref_thresh = self.ref_thresh
        ref_margin = self.ref_margin
        encode_preset = self.encode_preset
        keyframe_stride = self.keyframe_stride
        model = self.model
        """returns: {tid: {"mosaic": bool, "frames": int, "thumb": np.ndarray|None}}"""
        model = get_model(model_name)
        _reset_tracker(model)
        targets = [target] if isinstance(target, str) else [t for t in target]
        if not targets:
            from .world import needs_world as _needs_world
            if not _needs_world(self.query_raw):
                targets = ["green shirt"]
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        if ultra:
            imgsz = max(imgsz, 960)
            conf = min(conf, 0.18)

        use_ref = ref_bank is not None and len(ref_bank) > 0
        selector = HybridSelector(targets, query_raw=self.query_raw, margin=self.margin,
                                  ref_bank=ref_bank, ref_thresh=ref_thresh,
                                  ref_margin=ref_margin, classifier=self.classifier,
                                  world_mode=self.world_mode, world_conf=self.world_conf,
                                  world_imgsz=self.world_imgsz, world_stride=1)

        # ---- 1패스: 키프레임에서 트랙별 찬반 수집 (월드는 키프레임 1회만).
        # vid_stride로 추적기 안에서 건너뛴다. 렌더(2패스)는 전프레임 그대로라 품질 동일.
        votes = defaultdict(list)
        first_thumb = {}
        results1 = model.track(source=input_path, stream=True, persist=True,
                               tracker="configs/bytetrack_stable.yaml", conf=conf, classes=[0],
                               verbose=False, imgsz=imgsz, iou=0.5, max_det=30,
                               vid_stride=max(1, int(keyframe_stride)))
        for r in results1:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes)
            masks_full = None
            if r.masks is not None:
                try:
                    masks_full = r.masks.data.cpu().numpy()
                except Exception:
                    masks_full = None
            keep = filter_contained(boxes)
            boxes = boxes[keep]
            ids = ids[keep]
            if masks_full is not None:
                masks_full = masks_full[np.array(keep)]
            items = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                tid = int(ids[i])
                x1c, y1c = max(0, int(x1)), max(0, int(y1))
                x2c, y2c = min(r.orig_img.shape[1], int(x2)), min(r.orig_img.shape[0], int(y2))
                if not big_enough(x1c, y1c, x2c, y2c):
                    continue
                mc = None
                if masks_full is not None and i < len(masks_full):
                    try:
                        m = cv2.resize(masks_full[i], (r.orig_img.shape[1], r.orig_img.shape[0]))
                        m = (m > 0.5).astype(np.uint8) * 255
                        mc = m[y1c:y2c, x1c:x2c]
                    except Exception:
                        mc = None
                crop = r.orig_img[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue
                items.append({"key": (tid, len(items)), "tid": tid,
                              "xyxy": [x1c, y1c, x2c, y2c], "crop": crop, "mc": mc})
            judged = selector.judge_frame(r.orig_img, items)
            for it in items:
                _tid = it["tid"]
                _hit, _detail = judged.get(it["key"], (False, {"selector": "heuristic"}))
                votes[_tid].append({"hit": bool(_hit), "targets": _detail.get("hits", []),
                                    "detail": _detail})
                if _tid not in first_thumb:
                    try:
                        _crop = it["crop"]
                        _th = cv2.resize(_crop, (96, max(1, int(96 * _crop.shape[0] / max(1, _crop.shape[1])))),
                                       interpolation=cv2.INTER_AREA)
                        first_thumb[_tid] = _th
                    except Exception:
                        first_thumb[_tid] = None

        # ---- 전역 확정: 승자 판정이 전체 키프레임의 20% 이상이면 확정.
        # 최고점 단일 기준은 쓰지 않는다. 경계 점수는 분율로 살린다. ----
        global_on = {}
        global_hit_targets = {}
        global_detail = {}
        for tid, lst in votes.items():
            if not lst:
                global_on[tid] = False
                global_hit_targets[tid] = []
                global_detail[tid] = {}
            else:
                frac = sum(1 for v in lst if v["hit"]) / len(lst)
                global_on[tid] = frac >= 0.2
                seen_t, ht = set(), []
                for v in lst:
                    for t in v["targets"]:
                        if t not in seen_t:
                            seen_t.add(t)
                            ht.append(t)
                global_hit_targets[tid] = ht
                samp = next((v["detail"] for v in lst if v["hit"]), (lst[0].get("detail", {}) if lst else {}))
                global_detail[tid] = {"frac": round(float(frac), 3), "sample": samp}

        _reset_tracker(model)

        # ---- 2패스: 확정 라벨로 렌더만 ----
        tmp_noaudio = output_path + ".noaudio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
        if not out.isOpened():
            raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")

        smooth = {}
        hist_mask = {}
        hist_miss = defaultdict(int)
        ALPHA = 0.55
        MAX_MISS = 12

        t0 = time.time()
        results2 = model.track(source=input_path, stream=True, persist=True,
                               tracker="configs/bytetrack_stable.yaml", conf=conf, classes=[0],
                               verbose=False, imgsz=imgsz, iou=0.5, max_det=30)
        done = 0
        for r in results2:
            frame = r.orig_img.copy()
            seen = set()
            keep_union = None
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes)
                masks_full = None
                if r.masks is not None:
                    try:
                        masks_full = r.masks.data.cpu().numpy()
                    except Exception:
                        masks_full = None
                keep = filter_contained(boxes)
                for _d, _k in zip(ids, keep):
                    if not _k:
                        seen.add(int(_d))
                boxes = boxes[keep]
                ids = ids[keep]
                if masks_full is not None:
                    masks_full = masks_full[np.array(keep)]
                items = []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    tid = int(ids[i])
                    seen.add(tid)
                    if tid in smooth:
                        px1, py1, px2, py2 = smooth[tid]
                        x1 = int(ALPHA * x1 + (1 - ALPHA) * px1)
                        y1 = int(ALPHA * y1 + (1 - ALPHA) * py1)
                        x2 = int(ALPHA * x2 + (1 - ALPHA) * px2)
                        y2 = int(ALPHA * y2 + (1 - ALPHA) * py2)
                    smooth[tid] = (x1, y1, x2, y2)
                    hist_miss[tid] = 0
                    x1, y1, x2, y2 = apply_area_limit(int(x1), int(y1), int(x2), int(y2), area_mode)
                    now_on = bool(global_on.get(tid, False))
                    should = (not now_on) if invert else now_on
                    mf = None
                    if masks_full is not None and i < len(masks_full):
                        try:
                            mfull = cv2.resize(masks_full[i], (frame.shape[1], frame.shape[0]))
                            mfull = (mfull > 0.5).astype(np.uint8) * 255
                            if area_mode != "full":
                                tmp = np.zeros_like(mfull)
                                tmp[max(0, y1):y2, max(0, x1):x2] = 255
                                mfull = cv2.bitwise_and(mfull, mfull, mask=tmp)
                            hist_mask[tid] = mfull
                            mf = mfull
                        except Exception:
                            mf = None
                    else:
                        mf = hist_mask.get(tid)
                    items.append((tid, x1, y1, x2, y2, should, mf))
                # 유지 판정 마스크 합집합. 타겟 렌더에서 빼서 이웃 번짐 방지.
                keep_union = None
                for (_tid, _x1, _y1, _x2, _y2, _should, _mf) in items:
                    if _should or _mf is None:
                        continue
                    try:
                        b = (_mf > 127).astype(np.uint8) * 255
                        b = cv2.dilate(b, np.ones((5, 5), np.uint8), iterations=1)
                        keep_union = b if keep_union is None else cv2.bitwise_or(keep_union, b)
                    except Exception:
                        pass
                for (tid, x1, y1, x2, y2, should, mf) in items:
                    if not should:
                        continue
                    if mf is not None and keep_union is not None:
                        try:
                            mf = cv2.bitwise_and(mf, mf, mask=cv2.bitwise_not(keep_union))
                        except Exception:
                            pass
                    render_person_strict(frame, x1, y1, x2, y2, mf, style, blocks, blur)

            for tid in list(smooth.keys()):
                if tid in seen:
                    continue
                hist_miss[tid] += 1
                if hist_miss[tid] > MAX_MISS:
                    continue
                if not invert and not global_on.get(tid, False):
                    continue
                if invert and global_on.get(tid, False):
                    continue
                x1, y1, x2, y2 = smooth[tid]
                x1, y1, x2, y2 = apply_area_limit(int(x1), int(y1), int(x2), int(y2), area_mode)
                _mf = hist_mask.get(tid)
                if _mf is not None and keep_union is not None:
                    try:
                        _mf = cv2.bitwise_and(_mf, _mf, mask=cv2.bitwise_not(keep_union))
                    except Exception:
                        pass
                render_person_strict(frame, x1, y1, x2, y2, _mf, style, blocks, blur)

            out.write(frame)
            done += 1
            if progress_fn and total > 0:
                try:
                    progress_fn(0.5 + 0.5 * min(1.0, done / max(1, total)), done, total, time.time() - t0)
                except Exception:
                    pass

        out.release()
        try:
            mux_audio_h264(tmp_noaudio, input_path, output_path, preset=encode_preset)
        except Exception:
            import shutil
            shutil.copy(tmp_noaudio, output_path)
        finally:
            cleanup_tmp(tmp_noaudio)

        stats = {}
        for tid, lst in votes.items():
            samp_sel = global_detail.get(tid, {}).get("sample", {}).get("selector",
                        ("heuristic-ref" if use_ref else "heuristic"))
            on = bool(global_on.get(tid, False))
            stats[tid] = {
                "mosaic": (not on) if invert else on,
                "frames": len(lst),
                "thumb": first_thumb.get(tid),
                "selector": samp_sel,
                "query_raw": self.query_raw,
                "targets": list(targets),
                "hit_targets": global_hit_targets.get(tid, []),
                "frac": global_detail.get(tid, {}).get("frac", 0.0),
                "sample": global_detail.get(tid, {}).get("sample", {}),
            }
        return stats


def mosaic_video(input_path, output_path, target="green shirt", invert=False, conf=0.15, margin=0.04, model_name="yolo11m-seg.pt", imgsz=960, blocks=30, style="pixel", blur=61, ultra=True, area_mode="full", ref_hists=None, ref_bank=None, ref_thresh=0.82, ref_margin=0.05, encode_preset="veryfast", keyframe_stride=3, progress_fn=None, query_raw="", world_mode="auto", world_conf=None, world_imgsz=None):
    """호환 래퍼. 새 코드는 MosaicJob을 직접 쓴다."""
    job = MosaicJob(input_path, output_path, target=target, invert=invert, conf=conf, margin=margin, model_name=model_name, imgsz=imgsz, blocks=blocks, style=style, blur=blur, ultra=ultra, area_mode=area_mode, ref_hists=ref_hists, ref_bank=ref_bank, ref_thresh=ref_thresh, ref_margin=ref_margin, encode_preset=encode_preset, keyframe_stride=keyframe_stride, query_raw=query_raw, world_mode=world_mode, world_conf=world_conf, world_imgsz=world_imgsz)
    return job.run(progress_fn)
