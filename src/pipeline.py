"""2패스 파이프라인 (개인정보 보호용).
1패스는 VLM/월드 개인정보 탐지 결과 수집, 중앙값으로 확정.
2패스는 확정 라벨로 렌더만.
"""
import cv2
import numpy as np
import time
from collections import defaultdict
from .models import get_model


def _reset_tracker(model):
    try:
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            delattr(pred, "trackers")
    except Exception:
        pass


from .detect import filter_contained, big_enough
from .render import render_person_strict, mux_audio_h264, apply_area_limit, cleanup_tmp
from .select import VLMSelector, WorldSelector


class MosaicJob:
    """개인정보 보호 모자이크 작업 1건."""

    def __init__(self, input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, query_raw="",
                 api_key=None, vlm_model="gpt-4o-mini", use_vlm=False,
                 world_mode="auto"):
        self.input_path = input_path
        self.output_path = output_path
        self.targets = [targets] if isinstance(targets, str) else (list(targets) if targets else ["face"])
        if not self.targets:
            self.targets = ["face"]
        self.invert = invert
        self.conf = conf
        self.model_name = model_name
        self.imgsz = imgsz
        self.blocks = blocks
        self.style = style
        self.blur = blur
        self.ultra = ultra
        self.area_mode = area_mode
        self.encode_preset = encode_preset
        self.keyframe_stride = keyframe_stride
        self.query_raw = query_raw or ""
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.use_vlm = use_vlm
        self.world_mode = world_mode
        self.model = get_model(model_name)

    def run(self, progress_fn=None):
        input_path, output_path = self.input_path, self.output_path
        targets = self.targets
        invert = self.invert
        conf = self.conf
        model_name = self.model_name
        imgsz = self.imgsz
        blocks = self.blocks
        style = self.style
        blur = self.blur
        ultra = self.ultra
        area_mode = self.area_mode
        encode_preset = self.encode_preset
        keyframe_stride = self.keyframe_stride
        query_raw = self.query_raw
        api_key = self.api_key
        vlm_model = self.vlm_model
        use_vlm = self.use_vlm
        world_mode = self.world_mode

        model = get_model(model_name)
        _reset_tracker(model)

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        if ultra:
            imgsz = max(imgsz, 960)
            conf = min(conf, 0.18)

        if use_vlm and api_key:
            selector = VLMSelector(query_raw=query_raw, targets=targets,
                                   api_key=api_key, vlm_model=vlm_model,
                                   frame_stride=max(1, int(keyframe_stride)))
        else:
            from .select import WorldSelector
            selector = WorldSelector(targets=targets, query_raw=query_raw,
                                     conf=conf, imgsz=imgsz)

        # ---- 1패스: 키프레임에서 트랙별 개인정보 탐지 수집 ----
        votes = defaultdict(list)
        first_thumb = {}
        results1 = model.track(source=input_path, stream=True, persist=True,
                               tracker="configs/bytetrack_stable.yaml",
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30,
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

            judged = selector.select_first_frame(r.orig_img, items)
            for it in items:
                tid = it["tid"]
                hit, detail = judged.get(it["key"], (False, {"selector": "world"}))
                # VLM/월드 탐지 결과가 없어도 face 타겟이면 사람 얼굴 영역으로 간주
                if not hit and "face" in targets:
                    hit = True
                    detail["hits"] = ["face"]
                    detail["selector"] = "world-fallback"
                    detail["world_match"] = {"phrase": "face", "conf": 1.0, "iou": 1.0, "source": "fallback"}
                votes[tid].append({"hit": bool(hit), "targets": detail.get("hits", []),
                                   "detail": detail})
                if tid not in first_thumb:
                    try:
                        crop = it["crop"]
                        th = cv2.resize(crop, (96, max(1, int(96 * crop.shape[0] / max(1, crop.shape[1])))),
                                        interpolation=cv2.INTER_AREA)
                        first_thumb[tid] = th
                    except Exception:
                        first_thumb[tid] = None

        # ---- 전역 확정 ----
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
                samp = next((v["detail"] for v in lst if v["hit"]),
                            (lst[0].get("detail", {}) if lst else {}))
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
                               tracker="configs/bytetrack_stable.yaml",
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30)
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
                    render_person_strict(frame, x1, y1, x2, y2, mf,
                                         style=style, blocks=blocks, blur=blur)

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
                render_person_strict(frame, x1, y1, x2, y2, _mf,
                                     style=style, blocks=blocks, blur=blur)

            out.write(frame)
            done += 1
            if progress_fn and total > 0:
                try:
                    progress_fn(0.5 + 0.5 * min(1.0, done / max(1, total)),
                                done, total, time.time() - t0)
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
            samp_sel = global_detail.get(tid, {}).get("sample", {}).get("selector", "world")
            stats[tid] = {
                "mosaic": bool(global_on.get(tid, False)),
                "frames": len(lst),
                "thumb": first_thumb.get(tid),
                "selector": samp_sel,
                "query_raw": query_raw,
                "targets": list(targets),
                "hit_targets": global_hit_targets.get(tid, []),
                "frac": global_detail.get(tid, {}).get("frac", 0.0),
                "sample": global_detail.get(tid, {}).get("sample", {}),
            }
        return stats


def mosaic_video(input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, progress_fn=None,
                 query_raw="", api_key=None, vlm_model="gpt-4o-mini",
                 use_vlm=False, world_mode="auto"):
    """호환 래퍼."""
    job = MosaicJob(input_path, output_path, targets=targets, invert=invert,
                    conf=conf, model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur, ultra=ultra,
                    area_mode=area_mode, encode_preset=encode_preset,
                    keyframe_stride=keyframe_stride, query_raw=query_raw,
                    api_key=api_key, vlm_model=vlm_model,
                    use_vlm=use_vlm, world_mode=world_mode)
    return job.run(progress_fn)
