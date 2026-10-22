"""2패스 파이프라인 (개인정보 보호용).
1패스는 모든 프레임을 추적해 트랙 ID를 고정하고, 키프레임에서만 VLM/월드
개인정보 탐지를 수행해 (사람 트랙, 카테고리)별로 수집한다.
2패스는 같은 추적 시퀀스로 선택된 카테고리만 렌더링한다.
사람 항목은 사람 마스크/얼굴 영역, 배경 항목(번호판·간판 등)은 사각형 영역으로 가린다.
"""
import cv2
import math
import numpy as np
import os
import shutil
import time
from collections import defaultdict

from .config import (CATEGORY_TYPE, ROOT_DIR, VLM_MODEL, canon_category,
                     category_type)
from .models import get_model
from .detect import filter_contained, big_enough
from .render import (render_person_strict, render_region, mux_audio_h264,
                     apply_area_limit, cleanup_tmp)
from .select import VLMSelector
from .world import get_world, phrases_for_categories

_TRACKER_CFG = os.path.join(ROOT_DIR, "configs", "bytetrack_stable.yaml")


def _reset_tracker(model):
    try:
        pred = getattr(model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            delattr(pred, "trackers")
    except Exception:
        pass


def _cat_type(cat, rtype=None):
    """표준 카테고리는 설정을 따르고, 자유 형식은 VLM이 준 region_type을 쓴다."""
    if cat in CATEGORY_TYPE:
        return CATEGORY_TYPE[cat]
    if rtype in ("person", "region"):
        return rtype
    return category_type(cat)


def _dedupe_boxes(boxes, thresh=0.55):
    out = []
    for b in boxes:
        xy = b["xyxy"]
        dup = False
        for o in out:
            ox = o["xyxy"]
            ix1, iy1 = max(xy[0], ox[0]), max(xy[1], ox[1])
            ix2, iy2 = min(xy[2], ox[2]), min(xy[3], ox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            aa = max(1, xy[2] - xy[0]) * max(1, xy[3] - xy[1])
            ba = max(1, ox[2] - ox[0]) * max(1, ox[3] - ox[1])
            if inter / max(1.0, aa + ba - inter) >= thresh:
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def _sample_boxes(samples, sample_idx):
    """샘플 인덱스 이하에서 가장 가까운 VLM/월드 박스 묶음을 돌려준다.
    첫 샘플보다 이전 프레임에는 아직 탐지가 없었으므로 빈 목록을 준다."""
    if not samples:
        return []
    pick = None
    for k in sorted(samples):
        if k <= sample_idx:
            pick = k
        else:
            break
    return samples[pick] if pick is not None else []


class MosaicJob:
    """개인정보 보호 모자이크 작업 1건."""

    def __init__(self, input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, query_raw="",
                 api_key=None, vlm_model=VLM_MODEL):
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
        self.keyframe_stride = max(1, int(keyframe_stride))
        self.query_raw = query_raw or ""
        self.api_key = api_key
        self.vlm_model = vlm_model

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

        model = get_model(model_name)
        _reset_tracker(model)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, first = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
        H, W = first.shape[:2]
        del first

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        if ultra:
            imgsz = max(imgsz, 960)
            conf = min(conf, 0.18)

        selector = VLMSelector(query_raw=query_raw, targets=targets,
                               api_key=api_key, vlm_model=vlm_model,
                               include_all=False)

        # ---- 1패스: 모든 프레임 추적 + 키프레임 VLM/월드 탐지 ----
        # 1패스와 2패스가 같은 프레임 시퀀스를 추적해야 트랙 ID가 일치한다.
        votes = defaultdict(lambda: defaultdict(list))
        region_samples = defaultdict(lambda: defaultdict(list))
        first_thumb = {}
        seen_tids = set()
        observed_types = {}
        person_cats_seen = {c for c in targets
                            if CATEGORY_TYPE.get(c) == "person"}
        all_detail = defaultdict(dict)

        t0 = time.time()
        results1 = model.track(source=input_path, stream=True, persist=True,
                               tracker=_TRACKER_CFG,
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30)

        frame_idx = -1
        for r in results1:
            frame_idx += 1
            if progress_fn and total > 0 and frame_idx % 5 == 0:
                try:
                    progress_fn(0.5 * min(1.0, (frame_idx + 1) / total),
                                frame_idx + 1, total, time.time() - t0)
                except Exception:
                    pass

            # 얼굴 타겟은 키프레임 사이에만 보이는 트랙도 놓치지 않도록 ID를 모아둔다.
            if "face" in targets and r.boxes is not None and r.boxes.id is not None:
                try:
                    seen_tids.update(int(t) for t in
                                     r.boxes.id.cpu().numpy().astype(int))
                except Exception:
                    pass

            if frame_idx % keyframe_stride != 0:
                continue
            sample_idx = frame_idx // keyframe_stride

            frame = r.orig_img
            boxes = np.zeros((0, 4), dtype=int)
            ids = np.zeros((0,), dtype=int)
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                ids = (r.boxes.id.cpu().numpy().astype(int)
                       if r.boxes.id is not None else [-1] * len(boxes))

            keep = filter_contained(boxes) if len(boxes) else []
            cand = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if not keep[i]:
                    continue
                tid = int(ids[i])
                x1c, y1c = max(0, int(x1)), max(0, int(y1))
                x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                if not big_enough(x1c, y1c, x2c, y2c):
                    continue
                cand.append({"key": (tid, len(cand)), "tid": tid,
                             "xyxy": [x1c, y1c, x2c, y2c]})
                if tid not in first_thumb:
                    try:
                        crop = frame[y1c:y2c, x1c:x2c]
                        th = cv2.resize(
                            crop,
                            (96, max(1, int(96 * crop.shape[0] / max(1, crop.shape[1])))),
                            interpolation=cv2.INTER_AREA)
                        first_thumb[tid] = th
                    except Exception:
                        first_thumb[tid] = None

            # 사람이 없어도 배경 개인정보를 찾아야 하므로 항상 호출한다.
            world_boxes = selector.detect_boxes(frame)
            judged = selector.match_items(cand, world_boxes)
            hits_by_key = {}
            detail_by_key = {}
            for it in cand:
                _hit, _detail = judged.get(it["key"], (False, {}))
                hits_by_key[it["key"]] = set(_detail.get("hits", []))
                detail_by_key[it["key"]] = _detail

            # 사람별 카테고리 투표. 선택된 카테고리만 반영한다.
            for it in cand:
                tid = it["tid"]
                # match_items가 이미 배경(region) 박스를 걸러냈으므로
                # 여기서는 선택된 카테고리만 남기면 된다.
                hits = {c for c in hits_by_key.get(it["key"], set())
                        if c in targets}
                # 같은 키프레임에서 처음 발견된 자유 형식 카테고리도 즉시 투표한다.
                person_cats_seen.update(hits)
                cats_here = set(person_cats_seen)
                for cat in cats_here:
                    votes[tid][cat].append(cat in hits)
                for cat in hits:
                    all_detail[tid].setdefault(cat, detail_by_key.get(it["key"], {}))

            # 배경 영역 후보 저장
            for b in world_boxes:
                cat = canon_category(b.get("category") or b.get("phrase"))
                if cat not in targets:
                    continue
                rtype = _cat_type(cat, b.get("region_type"))
                observed_types.setdefault(cat, rtype)
                if rtype == "region":
                    region_samples[cat][sample_idx].append(b)

        _reset_tracker(model)

        # 키프레임 사이에만 보인 사람도 얼굴 타겟이면 가린다.
        if "face" in targets:
            for tid in seen_tids:
                votes[tid]["face"].append(True)

        # ---- 전역 확정 ----
        # 사람 카테고리 / 배경 카테고리 분리. 선택 해제한 항목은 targets에 없다.
        person_cats = {c for c in targets
                       if c in person_cats_seen
                       or _cat_type(c, observed_types.get(c)) == "person"}
        region_cats = {c for c in targets if c not in person_cats}
        # 분석에서 한 번이라도 보인 배경 카테고리는 2패스에서 영상 전체를 다시 찾는다.
        region_cats |= {c for c in region_samples
                        if c in targets and c not in person_cats}

        cat_on = {}
        global_on = {}
        global_hit_targets = {}
        global_detail = {}
        for tid, by_cat in votes.items():
            cat_on[tid] = {}
            for cat, lst in by_cat.items():
                if not lst:
                    continue
                frac = sum(1 for v in lst if v) / len(lst)
                cat_on[tid][cat] = frac >= 0.2
            on_cats = [c for c in targets
                       if cat_on[tid].get(c) and c in person_cats]
            global_on[tid] = bool(on_cats)
            global_hit_targets[tid] = on_cats
            fracs = [sum(1 for v in lst if v) / len(lst)
                     for lst in by_cat.values() if lst]
            frac_any = max(fracs) if fracs else 0.0
            samp = {}
            for c in on_cats or list(cat_on[tid].keys()):
                if tid in all_detail and c in all_detail[tid]:
                    samp = all_detail[tid][c]
                    break
            global_detail[tid] = {"frac": round(float(frac_any), 3),
                                  "cat_on": dict(cat_on[tid]),
                                  "sample": samp}

        # ---- 2패스: 선택된 카테고리만 렌더 ----
        tmp_noaudio = output_path + ".noaudio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
        if not out.isOpened():
            cleanup_tmp(tmp_noaudio)
            raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")

        smooth = {}
        hist_mask = {}
        hist_miss = defaultdict(int)
        rendered_tids = set()
        ALPHA = 0.55
        MAX_MISS = 12

        # 배경 영역: YOLO-World를 키프레임마다 돌려 박스를 갱신하고,
        # VLM이 찾은 박스는 샘플 구간 동안 유지한다.
        region_phrases = phrases_for_categories(region_cats)
        region_world = get_world() if region_phrases else None
        region_live = {}
        region_stride = max(1, keyframe_stride)
        rendered_region_cats = set()

        results2 = model.track(source=input_path, stream=True, persist=True,
                               tracker=_TRACKER_CFG,
                               conf=conf, classes=[0], verbose=False,
                               imgsz=imgsz, iou=0.5, max_det=30)
        done = 0
        for r in results2:
            frame = r.orig_img.copy()

            if region_world is not None and done % region_stride == 0:
                try:
                    wb = region_world.ground(frame, region_phrases,
                                            conf=0.20, imgsz=640)
                    cur = defaultdict(list)
                    for b in wb:
                        cat = canon_category(b.get("category") or b.get("phrase"))
                        if cat in region_cats:
                            cur[cat].append(b)
                    region_live = dict(cur)
                except Exception:
                    region_live = {}

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
                    if render_person_strict(frame, x1, y1, x2, y2, mf,
                                            style=style, blocks=blocks, blur=blur):
                        rendered_tids.add(tid)

            # 트랙이 잠깐 사라진 프레임도 같은 카테고리면 계속 가린다.
            for tid in list(smooth.keys()):
                if tid in seen:
                    continue
                hist_miss[tid] += 1
                if hist_miss[tid] > MAX_MISS:
                    smooth.pop(tid, None)
                    hist_mask.pop(tid, None)
                    hist_miss.pop(tid, None)
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
                if render_person_strict(frame, x1, y1, x2, y2, _mf,
                                        style=style, blocks=blocks, blur=blur):
                    rendered_tids.add(tid)

            # 배경 영역 렌더
            sidx = done // keyframe_stride
            for cat in region_cats:
                boxes = region_live.get(cat) or _sample_boxes(
                    region_samples.get(cat), sidx)
                boxes = _dedupe_boxes(boxes)
                for b in boxes:
                    x1, y1, x2, y2 = b["xyxy"]
                    if render_region(frame, x1, y1, x2, y2,
                                     style=style, blocks=blocks, blur=blur,
                                     pad=8):
                        rendered_region_cats.add(cat)

            out.write(frame)
            done += 1
            if progress_fn and total > 0:
                try:
                    progress_fn(0.5 + 0.5 * min(1.0, done / max(1, total)),
                                done, total, time.time() - t0)
                except Exception:
                    pass

        out.release()
        if done == 0:
            cleanup_tmp(tmp_noaudio)
            raise RuntimeError("처리된 프레임이 없습니다.")
        try:
            mux_audio_h264(tmp_noaudio, input_path, output_path, preset=encode_preset)
        except Exception:
            cleanup_tmp(output_path)
            shutil.copy(tmp_noaudio, output_path)
        finally:
            cleanup_tmp(tmp_noaudio)

        stats = {}
        for tid in votes:
            _by_cat = votes[tid]
            stats[tid] = {
                "mosaic": tid in rendered_tids,
                "frames": sum(len(v) for v in _by_cat.values())
                if isinstance(_by_cat, dict) else 0,
                "thumb": first_thumb.get(tid),
                "selector": global_detail.get(tid, {}).get("sample", {}).get("selector", "world"),
                "query_raw": query_raw,
                "targets": list(targets),
                "hit_targets": global_hit_targets.get(tid, []),
                "cat_on": {k: v for k, v in cat_on.get(tid, {}).items() if v},
                "frac": global_detail.get(tid, {}).get("frac", 0.0),
                "sample": global_detail.get(tid, {}).get("sample", {}),
            }
        stats["_regions"] = len(rendered_region_cats)
        return stats


def mosaic_video(input_path, output_path, targets=None, invert=False,
                 conf=0.15, model_name="yolo11m-seg.pt", imgsz=960,
                 blocks=30, style="pixel", blur=61, ultra=True,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=3, progress_fn=None,
                 query_raw="", api_key=None, vlm_model=VLM_MODEL):
    """호환 래퍼."""
    job = MosaicJob(input_path, output_path, targets=targets, invert=invert,
                    conf=conf, model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur, ultra=ultra,
                    area_mode=area_mode, encode_preset=encode_preset,
                    keyframe_stride=keyframe_stride, query_raw=query_raw,
                    api_key=api_key, vlm_model=vlm_model)
    return job.run(progress_fn)
