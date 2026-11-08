"""2패스 파이프라인 (개인정보 보호용).
1패스는 모든 프레임을 추적해 트랙 ID를 고정하고, 키프레임에서만 VLM
개인정보 탐지를 수행해 (사람 트랙, 카테고리)별로 수집한다.
2패스는 같은 추적 시퀀스로 선택된 카테고리만 렌더링한다.
사람 항목은 사람 마스크/얼굴 영역, 배경 항목(번호판·간판 등)은 VLM이 찾은
사각형 영역으로 가린다. 명찰처럼 사람에 붙은 항목은 VLM 박스를 사람 기준
상대 좌표로 저장해 '가리는 범위'와 무관하게 직접 가린다.
체크리스트에서 VLM이 찾은 배경 박스(region_priors)를 처리 단계의 초기값으로
넘겨, 처리 중 VLM이 같은 프레임에서 놓쳐도 체크리스트와 결과가 어긋나지
않게 한다.
"""
import cv2
import math
import numpy as np
import os
import pickle
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from .config import (CATEGORY_TYPE, ROOT_DIR, VLM_MODEL, canon_category,
                     category_type, plausible_box)
from .models import get_model
from .detect import filter_contained, big_enough, iou, region_has_detail
from .faces import (FaceTracker, detect_faces, expand_face_box,
                    scene_signature, is_scene_cut)
from .render import (render_person_strict, render_person_roi, render_region,
                     render_face_ellipse, mux_audio_h264, apply_area_limit,
                     cleanup_tmp, FFmpegFrameWriter)
from .select import VLMSelector
from .vlm_budget import VLMBudget, classify_vlm_error

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


def _prior_coords(boxes):
    """체크리스트의 [x1,y1,x2,y2] 목록을 안전한 정수 박스로 바꾼다."""
    out = []
    for xyxy in boxes or []:
        if len(xyxy) != 4:
            continue
        try:
            coords = [int(float(v)) for v in xyxy]
        except (TypeError, ValueError):
            continue
        if coords[2] - coords[0] < 4 or coords[3] - coords[1] < 4:
            continue
        out.append(coords)
    return out


def _sample_boxes(samples, sample_idx, frame_idx=None, stride=1,
                  interpolate=False):
    """렌더 프레임에 쓸 VLM 박스 묶음을 돌려준다.

    체크리스트 priors는 실제 프레임(``due``)에 흩어져 있고 샘플 버킷
    인덱스와 정확히 맞지 않는다. 버킷 인덱스로만 고르면 다음 priors보다
    이른 프레임이 빈 목록이 되어 번호판이 깜빡인다. 그래서 이 함수는
    ``due`` 기준으로 이전 앵커와 다음 앵커를 고르고, 그 사이를 보간한다.
    due가 없는 1패스 검출은 키프레임 인덱스(stride 배수)라 앵커 프레임을
    버킷 위치로 간주한다.
    """
    if not samples:
        return []
    keys = sorted(samples)
    if frame_idx is None:
        pick = None
        for k in keys:
            if k <= sample_idx:
                pick = k
            else:
                break
        if pick is None:
            return []
        return list(samples[pick])

    stride = max(1, int(stride))
    anchors = []
    for k in keys:
        fallback = int(k) * stride
        for entry in samples[k]:
            due = entry.get("due", fallback)
            try:
                due = int(due)
            except (TypeError, ValueError):
                due = fallback
            anchors.append((due, int(k), entry))
    if not anchors:
        return []

    due_keys = sorted({due for due, _k, _e in anchors})
    prev_due = None
    nxt_due = None
    for due in due_keys:
        if due <= int(frame_idx):
            prev_due = due
        else:
            nxt_due = due
            break
    if prev_due is None:
        return []
    prev = [entry for due, _k, entry in anchors if due == prev_due]
    if not interpolate or nxt_due is None:
        return prev
    gap = nxt_due - prev_due
    if gap <= 0 or gap > 5 * stride:
        return prev
    t = (int(frame_idx) - prev_due) / float(gap)
    if t <= 0.0:
        return prev
    nxt = [entry for due, _k, entry in anchors if due == nxt_due]
    return _interpolate_boxes(prev, nxt, min(1.0, t))


def _interpolate_boxes(prev_boxes, next_boxes, t):
    """앞뒤 키프레임 박스를 크기/거리로 짝지어 선형 보간한다.

    짝을 못 찾은 이전 박스는 그대로 유지하고, 다음 키프레임에만 있는
    새 박스는 아직 이 프레임에 없을 수 있으므로 추가하지 않는다.
    """
    if not prev_boxes or not next_boxes:
        return prev_boxes
    used = set()
    out = []
    for p in prev_boxes:
        pxy = p["xyxy"]
        pw = max(1.0, pxy[2] - pxy[0])
        ph = max(1.0, pxy[3] - pxy[1])
        pcx, pcy = (pxy[0] + pxy[2]) / 2.0, (pxy[1] + pxy[3]) / 2.0
        best = None
        best_score = None
        for j, n in enumerate(next_boxes):
            if j in used:
                continue
            nxy = n["xyxy"]
            nw = max(1.0, nxy[2] - nxy[0])
            nh = max(1.0, nxy[3] - nxy[1])
            ratio = (nw * nh) / max(1.0, pw * ph)
            if ratio < 0.25 or ratio > 4.0:
                continue
            ncx, ncy = (nxy[0] + nxy[2]) / 2.0, (nxy[1] + nxy[3]) / 2.0
            scale = max(pw, ph, nw, nh)
            dist = math.hypot(ncx - pcx, ncy - pcy) / max(1.0, scale)
            if dist > 1.5:
                continue
            if best_score is None or dist < best_score:
                best, best_score = j, dist
        if best is None:
            out.append(p)
            continue
        used.add(best)
        nxy = next_boxes[best]["xyxy"]
        xy = [int(round(pxy[i] + (nxy[i] - pxy[i]) * t))
              for i in range(4)]
        merged = dict(p)
        merged["xyxy"] = xy
        out.append(merged)
    return out


def _relative_box(inner, outer):
    """사람 박스 기준 상대 좌표(0~1)로 바꾼다. 사람이 움직여도 같이 따라간다."""
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    ow, oh = max(1.0, ox2 - ox1), max(1.0, oy2 - oy1)
    x1, y1, x2, y2 = [float(v) for v in inner]

    def clamp(v):
        return max(0.0, min(1.0, v))

    return [clamp((x1 - ox1) / ow), clamp((y1 - oy1) / oh),
            clamp((x2 - ox1) / ow), clamp((y2 - oy1) / oh)]


def _absolute_box(rel, outer, width, height):
    ox1, oy1, ox2, oy2 = outer
    ow, oh = max(1, ox2 - ox1), max(1, oy2 - oy1)
    x1 = int(round(ox1 + rel[0] * ow))
    y1 = int(round(oy1 + rel[1] * oh))
    x2 = int(round(ox1 + rel[2] * ow))
    y2 = int(round(oy1 + rel[3] * oh))
    return [max(0, x1), max(0, y1),
            min(width, x2), min(height, y2)]


def _store_person_box(samples, cat, sample_idx, box, persons, due=0):
    """사람에 붙은 VLM 박스를 트랙 ID와 상대 좌표로 저장한다.
    사람이 움직여도 명찰/얼굴 박스가 같이 따라가도록 하기 위함이다.
    due는 이 박스가 실제로 발견된 프레임 번호다. 그 이전 프레임에는
    렌더하지 않는다."""
    xyxy = [int(round(float(v))) for v in box.get("xyxy", [])]
    if len(xyxy) != 4 or xyxy[2] - xyxy[0] < 4 or xyxy[3] - xyxy[1] < 4:
        return
    cx, cy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
    best = None
    best_area = None
    for p in persons:
        px1, py1, px2, py2 = p["xyxy"]
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        if (px1 - pw * 0.03 <= cx <= px2 + pw * 0.03
                and py1 - ph * 0.03 <= cy <= py2 + ph * 0.03):
            area = pw * ph
            if best is None or area < best_area:
                best, best_area = p, area
    if best is None:
        # 중심이 사람 박스 밖이면 겹침이 충분한 사람을 쓴다.
        best_iou = 0.12
        for p in persons:
            overlap = iou(p["xyxy"], xyxy)
            if overlap >= best_iou:
                best, best_iou = p, overlap
    tid = int(best["tid"]) if best is not None else -1
    # 얼굴은 사람 몸 안에 붙어 있어야 한다. VLM이 좌표를 잘못 해석해
    # 사람 몸 전체나 배경을 얼굴로 준 경우는 버린다(얼굴은 blanket 마스크가
    # 따로 적용되므로 놓쳐도 안전하다). 명찰은 사람이 검출되지 않은
    # 프레임에서도 절대 좌표 fallback으로 가릴 수 있게 남긴다.
    if best is None and cat == "face":
        return
    if best is not None and cat in ("face", "name_tag"):
        px1, py1, px2, py2 = best["xyxy"]
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        iw, ih = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        if iw > pw * 1.15 or ih > ph * 0.95:
            return
        max_frac = 0.55 if cat == "face" else 0.35
        if (iw * ih) > (pw * ph) * max_frac:
            return
    rel = _relative_box(xyxy, best["xyxy"]) if best is not None else None
    entries = samples[cat][sample_idx]
    for e in entries:
        if e["tid"] == tid and iou(e["abs"], xyxy) >= 0.55:
            return
    entries.append({"tid": tid, "rel": rel, "abs": xyxy, "due": int(due)})


def _nearest_rel_boxes(samples, sample_idx, tid, frame_idx=None):
    """해당 트랙에서 가장 최근에 찾은 사람 부착 박스의 상대 좌표."""
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        rels = [e["rel"] for e in samples[k]
                if e["tid"] == tid and e.get("rel")
                and (frame_idx is None
                     or int(e.get("due", 0) or 0) <= int(frame_idx))]
        if rels:
            return rels
    return []


def _missing_abs_boxes(samples, sample_idx, active_tids, max_age=2,
                       frame_idx=None):
    """현재 2패스에서 보이지 않는 트랙(tid=-1 포함)의 절대 박스.

    1패스/2패스 사이에 트랙 ID가 바뀌면 상대좌표 박스를 그릴 대상이
    없어진다. 이때 최근 키프레임의 절대 박스를 fallback으로 그려
    명찰·얼굴이 통째로 빠지는 것을 막는다. 오래된 박스가 배경을 계속
    가리지 않도록 max_age 키프레임까지만 사용한다.
    """
    if not samples:
        return []
    for k in sorted(samples, reverse=True):
        if k > sample_idx:
            continue
        if sample_idx - k > max_age:
            return []
        out = []
        for e in samples[k]:
            tid = int(e.get("tid", -1))
            if frame_idx is not None and int(e.get("due", 0) or 0) > frame_idx:
                continue
            if tid < 0 or tid not in active_tids:
                if e.get("abs"):
                    out.append((tid, e["abs"]))
        return out
    return []


def _render_rel_boxes(frame, raw_box, rels, style, blocks, blur,
                      area_mode="full", clip_area=True,
                      clip_to_person=True):
    """사람 박스 기준 상대 좌표를 현재 프레임 좌표로 바꿔 가린다.

    명찰처럼 작은 부착물은 ``clip_area=False``로 '가리는 범위'와 무관하게
    그대로 가린다. 얼굴·동승자·자유형 사람 항목은 사람 박스의 허용 범위
    안으로 잘라, '얼굴만' 설정이 전체 사람 박스로 번지지 않게 한다.
    clip_to_person은 얼굴/명찰이 사람 몸 밖으로 튀는 것을 막는다.
    """
    if not rels:
        return False
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = [int(v) for v in raw_box]
    ax1, ay1, ax2, ay2 = apply_area_limit(*raw_box, area_mode)
    rendered = False
    for rel in rels:
        bx = _absolute_box(rel, raw_box, w, h)
        if clip_to_person:
            bx = [max(bx[0], rx1), max(bx[1], ry1),
                  min(bx[2], rx2), min(bx[3], ry2)]
        if clip_area and area_mode != "full":
            bx = [max(bx[0], ax1), max(bx[1], ay1),
                  min(bx[2], ax2), min(bx[3], ay2)]
        if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
            continue
        if render_region(frame, *bx, style=style, blocks=blocks,
                         blur=blur, pad=4):
            rendered = True
    return rendered


class MosaicJob:
    """개인정보 보호 모자이크 작업 1건."""

    def __init__(self, input_path, output_path, targets=None, invert=False,
                 conf=0.20, model_name="yolo11n-seg.pt", imgsz=480,
                 blocks=30, style="pixel", blur=61, ultra=False,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=15, query_raw="",
                 api_key=None, vlm_model=VLM_MODEL, vlm_base_url=None,
                 region_priors=None, person_priors=None,
                 cache_path=None, use_cache=False,
                 vlm_call_limit=None, mask_all_persons=False):
        self.input_path = input_path
        self.output_path = output_path
        self.targets = ([targets] if isinstance(targets, str)
                        else (list(targets) if targets else ["face"]))
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
        self.vlm_base_url = vlm_base_url
        self.region_priors = region_priors or {}
        self.person_priors = person_priors or {}
        self.cache_path = cache_path
        self.use_cache = bool(use_cache)
        # 처리 단계 VLM 호출 상한(영상 1개 기준). None이면 키프레임마다 호출.
        self.vlm_call_limit = (None if vlm_call_limit is None
                               else max(0, int(vlm_call_limit)))
        # 얼굴/동승자처럼 사람 전체에 적용해도 되는 항목은 트랙마다
        # VLM 재탐지 없이 바로 가린다(누락 방지 + 속도).
        self.mask_all_persons = bool(mask_all_persons)

    def _face_only_mode(self):
        """얼굴만/얼굴+배경 항목을 로컬 얼굴 추적만으로 렌더할 수 있는지."""
        if (self.invert or self.area_mode != "face_only"
                or not self.mask_all_persons or "face" not in self.targets):
            return False
        return all(c == "face" or _cat_type(c) != "person"
                   for c in self.targets)

    def _run_face_only(self, progress_fn=None):
        """YuNet 얼굴 검출 + 짧은 추적으로 '얼굴만' 영상을 렌더한다.

        VLM 얼굴 박스는 키프레임 초기값으로만 쓰고, 모든 프레임의 실제
        얼굴은 로컬 검출기로 찾는다. 얼굴이 잠깐 가려져도 추적을 유지해
        모자이크가 사라졌다 나타나는 깜빡임을 줄인다.
        """
        input_path, output_path = self.input_path, self.output_path
        targets = list(self.targets)
        style, blocks, blur = self.style, self.blocks, self.blur
        area_mode = self.area_mode
        stride = max(1, int(self.keyframe_stride))
        preset = self.encode_preset

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
        H, W = first.shape[:2]
        del first
        cap.release()

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        # 배경 항목(번호판·간판 등)은 체크리스트 박스를 그대로 쓴다.
        region_samples = defaultdict(lambda: defaultdict(list))
        for cat, by_frame in (self.region_priors or {}).items():
            if cat not in targets or _cat_type(cat) != "region":
                continue
            for fidx, boxes in (by_frame or {}).items():
                try:
                    fidx = int(fidx)
                except (TypeError, ValueError):
                    continue
                sidx = max(0, fidx // stride)
                for coords in _prior_coords(boxes):
                    region_samples[cat][sidx].append({
                        "xyxy": coords, "conf": 1.0, "category": cat,
                        "region_type": "region", "source": "vlm",
                        "matches_prompt": True, "due": fidx,
                    })

        face_priors = defaultdict(list)
        for fidx, boxes in (self.person_priors.get("face") or {}).items():
            try:
                fidx = int(fidx)
            except (TypeError, ValueError):
                continue
            for coords in _prior_coords(boxes):
                if plausible_box("face", *coords, W, H):
                    face_priors[fidx].append(coords)

        tracker = FaceTracker(
            max_miss=max(30, int(round(fps * 1.0))),
            seed_miss=max(24, int(round(fps * 0.8))))
        rendered_tids = set()
        track_frames = defaultdict(int)
        rendered_cats = set()
        rendered_region_cats = set()
        max_active = 0
        prev_sig = None
        done = 0
        t0 = time.time()

        tmp_noaudio = output_path + ".noaudio.mp4"
        writer = None
        out = None
        try:
            try:
                writer = FFmpegFrameWriter(
                    output_path, W, H, fps, audio_src=input_path,
                    preset=preset)
            except Exception:
                writer = None
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
                if not out.isOpened():
                    cleanup_tmp(tmp_noaudio)
                    raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if is_scene_cut(prev_sig, frame):
                        tracker.reset()
                    prev_sig = scene_signature(frame)
                    for coords in face_priors.get(done, []):
                        tracker.seed(coords)
                    tracker.update(detect_faces(frame))
                    active = tracker.active()
                    max_active = max(max_active, len(active))
                    for track in active:
                        bx = expand_face_box(track.box, W, H)
                        if bx and render_face_ellipse(
                                frame, *bx, style=style, blocks=blocks,
                                blur=blur):
                            rendered_tids.add(track.tid)
                            track_frames[track.tid] += 1
                            rendered_cats.add("face")

                    sidx = done // stride
                    for cat in targets:
                        if cat == "face" or _cat_type(cat) != "region":
                            continue
                        boxes = _sample_boxes(
                            region_samples.get(cat), sidx, frame_idx=done,
                            stride=stride, interpolate=True)
                        for b in _dedupe_boxes(boxes):
                            x1, y1, x2, y2 = b["xyxy"]
                            if not plausible_box(cat, x1, y1, x2, y2, W, H):
                                continue
                            if not region_has_detail(frame, (x1, y1, x2, y2),
                                                     cat):
                                continue
                            if render_region(frame, x1, y1, x2, y2,
                                             style=style, blocks=blocks,
                                             blur=blur,
                                             pad=24 if cat == "license_plate" else 8):
                                rendered_region_cats.add(cat)
                                rendered_cats.add(cat)

                    if writer is not None:
                        writer.write(frame)
                    else:
                        out.write(frame)
                    done += 1
                    if progress_fn and total > 0:
                        try:
                            progress_fn(
                                min(1.0, done / max(1, total)),
                                done, total, time.time() - t0)
                        except Exception:
                            pass
            finally:
                cap.release()

            if done == 0:
                raise RuntimeError("처리된 프레임이 없습니다.")
        except Exception:
            if writer is not None:
                writer.abort()
            else:
                cleanup_tmp(tmp_noaudio)
            raise

        if writer is not None:
            writer.close()
        else:
            out.release()
            try:
                mux_audio_h264(tmp_noaudio, input_path, output_path,
                               preset=preset)
            except Exception:
                cleanup_tmp(output_path)
                shutil.copy(tmp_noaudio, output_path)
            finally:
                cleanup_tmp(tmp_noaudio)

        stats = {}
        for tid in sorted(rendered_tids):
            stats[tid] = {
                "mosaic": True,
                "direct": True,
                "frames": int(track_frames.get(tid, 0)),
                "selector": "local-face",
                "targets": targets,
                "hit_targets": ["face"],
                "cat_on": {"face": True},
                "frac": 1.0,
            }
        stats["_regions"] = len(rendered_region_cats)
        stats["_cats"] = sorted(rendered_cats)
        stats["_untracked"] = []
        stats["_faces"] = int(max_active)
        stats["_vlm_calls"] = 0
        stats["_vlm_errors"] = 0
        return stats

    def run(self, progress_fn=None):
        # 얼굴만 선택한 경우에는 사람 마스크 대신 YuNet 얼굴 추적으로
        # 바로 렌더한다. YOLO 사람 트랙이 오토바이 등에 가려져도 얼굴
        # 모자이크가 끊기지 않고, 목·어깨로 번지지 않는다.
        if self._face_only_mode():
            return self._run_face_only(progress_fn)

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
        vlm_base_url = self.vlm_base_url
        region_priors = self.region_priors
        person_priors = self.person_priors

        # 얼굴/동승자만 선택하고 모든 사람을 가리는 설정이면 VLM 재탐지와
        # 1패스 추적이 필요 없다. 한 번의 추적 패스로 바로 렌더한다.
        fast_person_only = bool(
            self.mask_all_persons and not invert and targets
            and all(c in CATEGORY_TYPE
                    and CATEGORY_TYPE.get(c) == "person"
                    and c != "name_tag" for c in targets))

        model = None
        # 사람 항목이 전혀 없으면 YOLO 모델 자체를 로드하지 않는다.
        needs_person_pass1 = (
            invert or bool(person_priors)
            or any(c not in CATEGORY_TYPE
                   or CATEGORY_TYPE.get(c) == "person" for c in targets))
        if needs_person_pass1:
            model = get_model(model_name)
            _reset_tracker(model)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        # ---- 1패스 결과 캐시 로드 (캐시 사용 시 1패스 건너뜀) ----
        cache_hit = False
        if (not fast_person_only and self.use_cache and self.cache_path
                and os.path.exists(self.cache_path)):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("version") != 2:
                    raise ValueError("cache version mismatch")
                votes = cache["votes"]
                person_box_samples = cache["person_box_samples"]
                region_samples = cache["region_samples"]
                person_cats_seen = cache.get("person_cats_seen", set())
                all_detail = cache.get("all_detail", {})
                observed_types = cache.get("observed_types", {})
                total = cache["meta"]["total"]
                fps = cache["meta"]["fps"]
                H = cache["meta"]["H"]
                W = cache["meta"]["W"]
                keyframe_stride = cache["meta"]["keyframe_stride"]
                max_sidx = max(0, (total - 1) // keyframe_stride) if total > 0 else 0
                cache_hit = True
            except Exception:
                cache_hit = False
                votes = None
                person_box_samples = None
                region_samples = None

        # observed_types와 pending_person_priors는 캐시 미스/히트와
        # 무관하게 항상 정의돼야 한다 (후처리/2패스에서 참조).
        if not cache_hit:
            observed_types = {}
        pending_person_priors = defaultdict(lambda: defaultdict(list))

        if not cache_hit:
            votes = defaultdict(lambda: defaultdict(list))
            person_box_samples = defaultdict(lambda: defaultdict(list))
            region_samples = defaultdict(lambda: defaultdict(list))

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"입력 영상을 열 수 없습니다: {input_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            # 1fps 이하의 느린 영상도 원래 속도를 유지한다. 값이 없을 때만 30.
            if not math.isfinite(fps) or fps <= 0.0:
                fps = 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            ok, first = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("입력 영상에서 프레임을 읽지 못했습니다.")
            H, W = first.shape[:2]
            del first

            if ultra:
                imgsz = max(imgsz, 960)
                conf = min(conf, 0.18)

            # 얼굴만 선택해 모든 사람을 바로 가리는 빠른 경로에서는 VLM을
            # 다시 부르지 않으므로 선택기를 만들지 않는다(API 키도 필요 없음).
            selector = None
            if not fast_person_only:
                selector = VLMSelector(query_raw=query_raw, targets=targets,
                                       api_key=api_key, vlm_model=vlm_model,
                                       include_all=False,
                                       vlm_base_url=vlm_base_url)

            # ---- 체크리스트 VLM 박스를 1패스 초기값으로 넣는다 ----
            region_samples = defaultdict(lambda: defaultdict(list))
            person_box_samples = defaultdict(lambda: defaultdict(list))
            pending_person_priors = defaultdict(lambda: defaultdict(list))
            max_sidx = max(0, (total - 1) // keyframe_stride) if total > 0 else 0

        # 캐시는 priors가 이미 반영된 1패스 결과다. 캐시 적중 시 다시 넣으면
        # 같은 박스가 중복 저장되므로 건너뛴다. 캐시 키에 priors 해시가
        # 포함돼 있어 priors가 달라지면 캐시도 새로 만든다.
        if not cache_hit:
            for cat, by_frame in (region_priors or {}).items():
                if cat not in targets or _cat_type(cat) != "region":
                    continue
                for fidx, boxes in (by_frame or {}).items():
                    try:
                        fidx = int(fidx)
                    except (TypeError, ValueError):
                        continue
                    # sample_idx는 floor로 잡아 1패스 키프레임과 맞추되,
                    # 실제 렌더는 due(frame_idx) 이후에만 한다.
                    sidx = min(max_sidx, max(0, fidx // keyframe_stride))
                    for coords in _prior_coords(boxes):
                        region_samples[cat][sidx].append({
                            "xyxy": coords, "conf": 1.0, "category": cat,
                            "region_type": "region", "source": "vlm",
                            "matches_prompt": True, "due": fidx,
                        })

            # 명찰·얼굴처럼 사람에 붙은 체크리스트 박스도 처리 단계로 넘긴다.
            # 1패스의 가장 가까운 키프레임에서 실제 사람 트랙과 매칭해
            # 상대좌표로 바꾸고, 매칭 실패 시 절대좌표 fallback을 쓴다.
            for cat, by_frame in (person_priors or {}).items():
                if cat not in targets or _cat_type(cat) != "person":
                    continue
                observed_types.setdefault(cat, "person")
                for fidx, boxes in (by_frame or {}).items():
                    try:
                        fidx = int(fidx)
                    except (TypeError, ValueError):
                        continue
                    sidx = min(max_sidx, max(0, fidx // keyframe_stride))
                    pending_person_priors[cat][sidx].extend(
                        (coords, fidx) for coords in _prior_coords(boxes))

        # ---- 1패스: 모든 프레임 추적 + 키프레임 VLM 탐지 ----
        # 1패스와 2패스가 같은 프레임 시퀀스를 추적해야 트랙 ID가 일치한다.
        # 캐시가 있으면 1패스를 건너뛴다.
        if cache_hit:
            _reset_tracker(model)
        else:
            votes = defaultdict(lambda: defaultdict(list))
            person_cats_seen = {c for c in targets
                                if CATEGORY_TYPE.get(c) == "person"}
            all_detail = defaultdict(dict)
        t0 = time.time()

        # VLM 호출은 영상 전체 키프레임에 고르게 배분한다. 무료 한도가
        # 작아도 뒤쪽 구간을 통째로 건너뛰지 않게 하기 위함이다.
        vlm_budget = VLMBudget(per_video_limit=self.vlm_call_limit)
        vlm_targets = None
        if self.vlm_call_limit is not None:
            if self.vlm_call_limit <= 0:
                vlm_targets = set()
            else:
                n_keyframes = max(
                    1, (max(1, total) + keyframe_stride - 1)
                    // keyframe_stride)
                k = min(int(self.vlm_call_limit), n_keyframes)
                if k <= 1:
                    vlm_targets = {0}
                else:
                    vlm_targets = {
                        int(round(i * (n_keyframes - 1) / float(k - 1)))
                        for i in range(k)
                    }
        vlm_errors = 0
        vlm_disabled = False

        def _process_chunk(chunk):
            """키프레임 묶음의 VLM을 병렬 호출하고 순서대로 집계한다.

            추적은 순차 스트림이지만 VLM 네트워크 대기는 병렬화할 수 있다.
            프레임은 묶음 크기만큼만 들고 있어 메모리도 제한된다.
            """
            nonlocal vlm_errors, vlm_disabled
            if not chunk:
                return
            vlm_results = {}
            if not vlm_disabled and (vlm_targets is None or vlm_targets):
                run_now = []
                for rec in chunk:
                    if (vlm_targets is not None
                            and rec["sample_idx"] not in vlm_targets):
                        continue
                    if not vlm_budget.can_call():
                        continue
                    vlm_budget.record(1)
                    run_now.append(rec)
                if run_now:
                    with ThreadPoolExecutor(
                            max_workers=min(4, len(run_now))) as ex:
                        futs = {id(rec): ex.submit(selector.detect_boxes,
                                                   rec["frame"])
                                for rec in run_now}
                        for rec in run_now:
                            try:
                                vlm_results[id(rec)] = futs[id(rec)].result()
                            except Exception as e:
                                vlm_errors += 1
                                vlm_results[id(rec)] = []
                                # 한도/인증/모델 오류는 더 호출해도 실패한다.
                                if classify_vlm_error(e) in (
                                        "quota_exceeded", "auth_error",
                                        "model_not_found"):
                                    vlm_disabled = True

            for rec in chunk:
                frame = rec["frame"]
                cand = rec["cand"]
                sample_idx = rec["sample_idx"]
                vlm_boxes = vlm_results.get(id(rec), [])

                # 체크리스트에서 넘어온 사람 부착 박스. 같은 위치를 이번
                # 키프레임 VLM이 이미 찾았다면 중복 저장하지 않는다.
                if pending_person_priors:
                    live_person = defaultdict(list)
                    for b in vlm_boxes:
                        bcat = canon_category(b.get("category")
                                              or b.get("phrase"))
                        if (bcat in targets
                                and _cat_type(bcat, b.get("region_type"))
                                == "person"):
                            live_person[bcat].append(b)
                    for cat, by_sidx in list(pending_person_priors.items()):
                        if sample_idx not in by_sidx:
                            continue
                        for coords, due in by_sidx.pop(sample_idx):
                            if any(iou(coords, b.get("xyxy", [0, 0, 0, 0])) >= 0.25
                                   for b in live_person.get(cat, [])):
                                continue
                            _store_person_box(
                                person_box_samples, cat, sample_idx,
                                {"xyxy": coords}, cand, due=due)

                judged = selector.match_items(cand, vlm_boxes)
                hits_by_key = {}
                detail_by_key = {}
                for it in cand:
                    _hit, _detail = judged.get(
                        it["key"], (False, {"selector": "vlm", "hits": []}))
                    hits_by_key[it["key"]] = set(_detail.get("hits", []))
                    detail_by_key[it["key"]] = _detail

                # 사람별 카테고리 투표. 선택된 카테고리만 반영한다.
                for it in cand:
                    tid = it["tid"]
                    hits = {c for c in hits_by_key.get(it["key"], set())
                            if c in targets}
                    person_cats_seen.update(hits)
                    for cat in hits:
                        votes[tid][cat].append(True)
                        all_detail[tid].setdefault(
                            cat, detail_by_key.get(it["key"], {}))

                # 배경 영역 후보 저장 (VLM이 찾은 번호판·간판 등)
                for b in vlm_boxes:
                    cat = canon_category(b.get("category") or b.get("phrase"))
                    if cat not in targets:
                        continue
                    rtype = _cat_type(cat, b.get("region_type"))
                    observed_types.setdefault(cat, rtype)
                    if rtype == "region":
                        region_samples[cat][sample_idx].append(b)
                    else:
                        # 명찰·얼굴처럼 사람에 붙은 항목도 별도 박스로 저장한다.
                        _store_person_box(person_box_samples, cat, sample_idx,
                                          b, cand)
                rec["frame"] = None

        # 사람 항목이 전혀 없으면 1패스에서도 YOLO 추적을 하지 않는다.
        # (번호판/간판만 가리는 영상은 키프레임 VLM 탐지와 렌더만 필요)
        run_pass1_stream = (not fast_person_only
                            and (needs_person_pass1 or vlm_targets is None
                                 or bool(vlm_targets)))

        if not cache_hit and run_pass1_stream:
            cap1 = None
            if needs_person_pass1:
                _reset_tracker(model)
                results1 = model.track(
                    source=input_path, stream=True, persist=True,
                    tracker=_TRACKER_CFG, conf=conf, classes=[0],
                    verbose=False, imgsz=imgsz, iou=0.5, max_det=30)

                def _iter1():
                    idx = -1
                    for res in results1:
                        idx += 1
                        yield idx, res
            else:
                cap1 = cv2.VideoCapture(input_path)

                def _iter1():
                    idx = -1
                    while True:
                        ok1, frm = cap1.read()
                        if not ok1:
                            break
                        idx += 1
                        yield idx, SimpleNamespace(
                            orig_img=frm, boxes=None, masks=None)

            try:
                chunk = []
                for frame_idx, r in _iter1():
                    if progress_fn and total > 0 and frame_idx % 5 == 0:
                        try:
                            progress_fn(
                                0.5 * min(1.0, (frame_idx + 1) / total),
                                frame_idx + 1, total, time.time() - t0)
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
                               if r.boxes.id is not None
                               else np.full((len(boxes),), -1, dtype=int))

                    keep = filter_contained(boxes) if len(boxes) else []
                    cand = []
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        if not keep[i]:
                            continue
                        tid = int(ids[i])
                        x1c, y1c = max(0, int(x1)), max(0, int(y1))
                        x2c, y2c = (min(frame.shape[1], int(x2)),
                                    min(frame.shape[0], int(y2)))
                        if not big_enough(x1c, y1c, x2c, y2c):
                            continue
                        cand.append({"key": (tid, len(cand)), "tid": tid,
                                     "xyxy": [x1c, y1c, x2c, y2c]})

                    chunk.append({
                        "frame_idx": frame_idx,
                        "sample_idx": sample_idx,
                        "cand": cand,
                        # VLM 작업이 추적 버퍼를 참조하지 않도록 복사한다.
                        "frame": frame.copy(),
                    })
                    if len(chunk) >= 6:
                        _process_chunk(chunk)
                        chunk = []
                _process_chunk(chunk)
            finally:
                if cap1 is not None:
                    cap1.release()

            # 사람 트랙과 한 번도 매칭되지 않은 체크리스트 박스도 절대좌표로
            # 남겨, 트랙 ID가 바뀌거나 사람 검출이 약한 구간에서도 가린다.
            for cat, by_sidx in pending_person_priors.items():
                for _sidx, entries in by_sidx.items():
                    for coords, due in entries:
                        person_box_samples[cat][_sidx].append(
                            {"tid": -1, "rel": None, "abs": coords,
                             "due": int(due)})


        # 1패스 결과 캐시 저장 (캐시 사용 시 재실행 때 1패스 생략 가능)
        if (not fast_person_only and self.use_cache and self.cache_path
                and not cache_hit):
            try:
                cache_dir = os.path.dirname(self.cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                cache = {
                    "version": 2,
                    "meta": {
                        "total": total,
                        "fps": fps,
                        "H": H,
                        "W": W,
                        "keyframe_stride": keyframe_stride,
                    },
                    "votes": dict(votes),
                    "person_box_samples": dict(person_box_samples),
                    "region_samples": dict(region_samples),
                    "person_cats_seen": set(person_cats_seen),
                    "all_detail": dict(all_detail),
                    "observed_types": dict(observed_types),
                }
                with open(self.cache_path, "wb") as f:
                    pickle.dump(cache, f)
            except Exception:
                pass

        _reset_tracker(model)

        # ---- 전역 확정 ----
        # 사람 카테고리 / 배경 카테고리 분리. 선택 해제한 항목은 targets에 없다.
        prior_person_cats = {c for c in person_priors if c in targets}
        for _cat in prior_person_cats:
            observed_types.setdefault(_cat, "person")
        person_cats = {c for c in targets
                       if c in person_cats_seen or c in prior_person_cats
                       or _cat_type(c, observed_types.get(c)) == "person"}
        region_cats = {c for c in targets if c not in person_cats}
        # 분석/체크리스트에서 한 번이라도 보인 배경 카테고리는 2패스에서 렌더한다.
        region_cats |= {c for c in region_samples
                        if c in targets and c not in person_cats}
        # 사람에 붙은 항목(명찰·얼굴 등)은 '가리는 범위'와 별개로
        # VLM 박스 위치를 직접 가린다.
        direct_cats = {c for c in person_cats if person_box_samples.get(c)}
        # 명찰은 사람 몸 전체가 아니라 찾은 위치만 가린다. 얼굴/동승자/
        # 자유형 사람 항목만 '가리는 범위'에 따라 사람 마스크를 적용한다.
        mask_cats = {c for c in person_cats if c != "name_tag"}

        cat_on = {}
        for tid, by_cat in votes.items():
            cat_on[tid] = {}
            for cat, lst in by_cat.items():
                if not lst:
                    continue
                frac = sum(1 for v in lst if v) / len(lst)
                cat_on[tid][cat] = frac >= 0.2
        # VLM이 한 번이라도 찾은 사람 부착 박스는 짧게 스쳐도 가린다.
        # (임계값 미만이라 조용히 빠지는 것을 막는다.)
        for cat in direct_cats:
            for by_sidx in person_box_samples.get(cat, {}).values():
                for entry in by_sidx:
                    tid = int(entry.get("tid", -1))
                    if tid >= 0:
                        cat_on.setdefault(tid, {})[cat] = True

        global_on = {}
        mask_on = {}
        global_hit_targets = {}
        global_detail = {}
        for tid in sorted(set(votes) | set(cat_on)):
            by_cat = votes.get(tid, {})
            on_cats = [c for c in targets
                       if cat_on.get(tid, {}).get(c) and c in person_cats]
            global_on[tid] = bool(on_cats)
            mask_on[tid] = any(c in mask_cats for c in on_cats)
            global_hit_targets[tid] = on_cats
            fracs = [sum(1 for v in lst if v) / len(lst)
                     for lst in by_cat.values() if lst]
            frac_any = max(fracs) if fracs else (1.0 if on_cats else 0.0)
            samp = {}
            for c in on_cats or list(cat_on.get(tid, {}).keys()):
                if tid in all_detail and c in all_detail[tid]:
                    samp = all_detail[tid][c]
                    break
            global_detail[tid] = {"frac": round(float(frac_any), 3),
                                  "cat_on": {k: v for k, v in
                                             cat_on.get(tid, {}).items() if v},
                                  "sample": samp}

        # 얼굴/동승자처럼 '사람이면 가린다'가 안전한 항목은 VLM이 매
        # 프레임 잡지 못해도 모든 사람 트랙에 적용한다. 명찰은 위치가
        # 사람마다 다르므로 제외하고, 찾은 박스만 직접 가린다.
        blanket_cats = ({c for c in person_cats if c in mask_cats}
                        if (self.mask_all_persons and not invert) else set())

        # ---- 2패스: 선택된 카테고리만 렌더 ----
        # ffmpeg 파이프로 바로 H.264 인코딩한다(기존 mp4v→H.264 2중
        # 인코딩 제거). 실패하면 기존 VideoWriter 경로로 대체한다.
        tmp_noaudio = output_path + ".noaudio.mp4"
        writer = None
        out = None
        try:
            writer = FFmpegFrameWriter(
                output_path, W, H, fps, audio_src=input_path,
                preset=encode_preset)
        except Exception:
            writer = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (W, H))
            if not out.isOpened():
                cleanup_tmp(tmp_noaudio)
                raise RuntimeError("VideoWriter 열기 실패. 코덱/경로 확인.")

        smooth = {}
        hist_mask = {}
        hist_miss = defaultdict(int)
        rendered_tids = set()
        rendered_direct_tids = set()
        rendered_cats = set()
        untracked_cats = set()
        track_frames = defaultdict(int)
        ALPHA = 0.55
        # 얼굴/동승자 blanket 마스크는 오토바이·행인 등에 잠깐 가려져
        # YOLO가 사람을 놓쳐도 마지막 마스크를 유지한다. 짧게 끊기면
        # 모자이크가 풀렸다 다시 생기는 깜빡임이 보인다.
        MAX_MISS = 30 if blanket_cats else 12

        rendered_region_cats = set()

        done = 0
        progress_base = 0.0 if fast_person_only else 0.5
        # 사람 항목이 없으면 2패스에서 YOLO 추적을 아예 건너뛴다.
        # (번호판만 가리는 경우 렌더 속도가 크게 빨라진다.)
        needs_person = bool(person_cats) or invert
        cap2 = None
        if needs_person:
            _reset_tracker(model)
            results2 = model.track(source=input_path, stream=True,
                                   persist=True, tracker=_TRACKER_CFG,
                                   conf=conf, classes=[0], verbose=False,
                                   imgsz=imgsz, iou=0.5, max_det=30)

            def _iter_frames():
                for res in results2:
                    yield res
        else:
            cap2 = cv2.VideoCapture(input_path)

            def _iter_frames():
                while True:
                    ok2, frm = cap2.read()
                    if not ok2:
                        break
                    yield SimpleNamespace(orig_img=frm, boxes=None,
                                          masks=None)

        try:
            for r in _iter_frames():
                frame = r.orig_img.copy()
                sidx = done // keyframe_stride


                seen = set()
                items = []
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                    ids = (r.boxes.id.cpu().numpy().astype(int)
                           if r.boxes.id is not None
                           else np.full((len(boxes),), -1, dtype=int))
                    masks_full = None
                    if r.masks is not None:
                        try:
                            masks_full = r.masks.data.cpu().numpy()
                        except Exception:
                            masks_full = None
                    keep = filter_contained(boxes)
                    # 큰 사람 안에 75% 이상 겹친 중복 검출은 그리지도,
                    # '보이는 트랙'으로 치지도 않는다. 남겨 두면 고스트가
                    # 계속 그려지므로 추적 상태에서도 지운다.
                    for _idx, _keep in enumerate(keep):
                        if _keep:
                            continue
                        _tid = int(ids[_idx])
                        if _tid >= 0:
                            smooth.pop(_tid, None)
                            hist_mask.pop(_tid, None)
                            hist_miss.pop(_tid, None)
                    boxes = boxes[keep]
                    ids = ids[keep]
                    if masks_full is not None:
                        masks_full = masks_full[np.array(keep, dtype=bool)]
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        tid = int(ids[i])
                        seen.add(tid)
                        track_frames[tid] += 1
                        if tid in smooth:
                            px1, py1, px2, py2 = smooth[tid]
                            x1 = int(ALPHA * x1 + (1 - ALPHA) * px1)
                            y1 = int(ALPHA * y1 + (1 - ALPHA) * py1)
                            x2 = int(ALPHA * x2 + (1 - ALPHA) * px2)
                            y2 = int(ALPHA * y2 + (1 - ALPHA) * py2)
                        smooth[tid] = (x1, y1, x2, y2)
                        hist_miss[tid] = 0
                        raw_box = (int(x1), int(y1), int(x2), int(y2))
                        ax1, ay1, ax2, ay2 = apply_area_limit(
                            *raw_box, area_mode)
                        # 명찰만 선택한 사람은 몸 전체를 가리지 않는다.
                        # 얼굴/동승자/자유형 사람 항목만 마스크를 적용한다.
                        # blanket_cats는 VLM 재탐지 없이 모든 사람에게
                        # 적용하는 얼굴/동승자 항목이다.
                        now_on = bool(global_on.get(tid, False))
                        should = ((not now_on) if invert
                                  else (bool(mask_on.get(tid, False))
                                        or bool(blanket_cats)))
                        mf = None
                        if masks_full is not None and i < len(masks_full):
                            try:
                                small = masks_full[i]
                                if invert:
                                    mfull = cv2.resize(
                                        small,
                                        (frame.shape[1], frame.shape[0]))
                                    mfull = ((mfull > 0.5).astype(np.uint8)
                                             * 255)
                                    if area_mode != "full":
                                        tmp = np.zeros_like(mfull)
                                        tmp[max(0, ay1):ay2,
                                            max(0, ax1):ax2] = 255
                                        mfull = cv2.bitwise_and(
                                            mfull, mfull, mask=tmp)
                                    hist_mask[tid] = mfull
                                    mf = mfull
                                else:
                                    # 모델 해상도 마스크에서 사람 영역만 잘라
                                    # ROI 크기로 줄인다. 전체 프레임 resize를
                                    # 매 프레임/사람마다 하지 않아 빠르다.
                                    mh, mw = small.shape[:2]
                                    sx = mw / float(max(1, W))
                                    sy = mh / float(max(1, H))
                                    mx1 = max(0, min(mw - 1, int(ax1 * sx)))
                                    my1 = max(0, min(mh - 1, int(ay1 * sy)))
                                    mx2 = max(mx1 + 1,
                                              min(mw, int(ax2 * sx)))
                                    my2 = max(my1 + 1,
                                              min(mh, int(ay2 * sy)))
                                    crop = small[my1:my2, mx1:mx2]
                                    if crop.size:
                                        mf = ((crop > 0.5).astype(np.uint8)
                                              * 255)
                                        mf = cv2.resize(
                                            mf, (max(1, ax2 - ax1),
                                                 max(1, ay2 - ay1)),
                                            interpolation=cv2.INTER_NEAREST)
                                        hist_mask[tid] = mf
                            except Exception:
                                mf = None
                        elif not invert:
                            mf = hist_mask.get(tid)
                        else:
                            mf = hist_mask.get(tid)
                        items.append((tid, ax1, ay1, ax2, ay2, should, mf,
                                      raw_box))

                # invert 모드에서만 '가리지 않을 사람'의 마스크를 뺀다.
                # 일반 모드에서 선택된 사람의 마스크는 다른 사람과 겹쳐도
                # 그대로 가린다(선택 항목 보호 우선).
                keep_union = None
                if invert:
                    for (_tid, _x1, _y1, _x2, _y2, _should, _mf,
                         _raw) in items:
                        if _should or _mf is None:
                            continue
                        try:
                            b = (_mf > 127).astype(np.uint8) * 255
                            b = cv2.dilate(b, np.ones((5, 5), np.uint8),
                                           iterations=1)
                            keep_union = (b if keep_union is None
                                          else cv2.bitwise_or(keep_union, b))
                        except Exception:
                            pass

                for (tid, x1, y1, x2, y2, should, mf, raw_box) in items:
                    # 명찰·얼굴 등 선택된 사람 부착 항목은 '가리는 범위'와
                    # 무관하게 VLM이 찾은 위치를 직접 가린다.
                    if direct_cats and not invert:
                        for cat in direct_cats:
                            rels = _nearest_rel_boxes(
                                person_box_samples.get(cat), sidx, tid,
                                frame_idx=done)
                            if _render_rel_boxes(
                                    frame, raw_box, rels, style, blocks,
                                    blur, area_mode=area_mode,
                                    clip_area=(cat not in ("name_tag",
                                                           "face"))):
                                rendered_tids.add(tid)
                                rendered_direct_tids.add(tid)
                                rendered_cats.add(cat)
                    if not should:
                        continue
                    render_mask = mf
                    if invert and render_mask is not None \
                            and keep_union is not None:
                        try:
                            render_mask = cv2.bitwise_and(
                                render_mask, render_mask,
                                mask=cv2.bitwise_not(keep_union))
                        except Exception:
                            pass
                    if invert:
                        ok_render = render_person_strict(
                            frame, x1, y1, x2, y2, render_mask,
                            style=style, blocks=blocks, blur=blur,
                            area_mode=area_mode)
                    else:
                        ok_render = render_person_roi(
                            frame, x1, y1, x2, y2, render_mask,
                            style=style, blocks=blocks, blur=blur,
                            area_mode=area_mode)
                    if ok_render:
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
                    x1, y1, x2, y2 = smooth[tid]
                    raw_box = (int(x1), int(y1), int(x2), int(y2))
                    if direct_cats and not invert:
                        for cat in direct_cats:
                            rels = _nearest_rel_boxes(
                                person_box_samples.get(cat), sidx, tid,
                                frame_idx=done)
                            if _render_rel_boxes(
                                    frame, raw_box, rels, style, blocks,
                                    blur, area_mode=area_mode,
                                    clip_area=(cat not in ("name_tag",
                                                           "face"))):
                                rendered_tids.add(tid)
                                rendered_direct_tids.add(tid)
                                rendered_cats.add(cat)
                    if invert:
                        if global_on.get(tid, False):
                            continue
                    elif (not mask_on.get(tid, False)
                          and not blanket_cats):
                        continue
                    x1, y1, x2, y2 = apply_area_limit(
                        *raw_box, area_mode)
                    _mf = hist_mask.get(tid)
                    if invert and _mf is not None and keep_union is not None:
                        try:
                            _mf = cv2.bitwise_and(
                                _mf, _mf, mask=cv2.bitwise_not(keep_union))
                        except Exception:
                            pass
                    if invert:
                        ok_render = render_person_strict(
                            frame, x1, y1, x2, y2, _mf,
                            style=style, blocks=blocks, blur=blur,
                            area_mode=area_mode)
                    else:
                        ok_render = render_person_roi(
                            frame, x1, y1, x2, y2, _mf,
                            style=style, blocks=blocks, blur=blur,
                            area_mode=area_mode)
                    if ok_render:
                        rendered_tids.add(tid)

                # 사람 트랙에 매칭되지 않은 사람 부착 박스(트랙 ID 불일치,
                # tid=-1)도 최근 절대 좌표로 가린다.
                if direct_cats and not invert:
                    for cat in direct_cats:
                        boxes = _missing_abs_boxes(
                            person_box_samples.get(cat), sidx, seen,
                            max_age=2, frame_idx=done)
                        for bx in _dedupe_boxes(
                                [{"xyxy": b} for _tid, b in boxes]):
                            if not plausible_box(
                                    cat, *bx["xyxy"],
                                    frame.shape[1], frame.shape[0]):
                                continue
                            if render_region(frame, *bx["xyxy"],
                                             style=style, blocks=blocks,
                                             blur=blur, pad=4):
                                rendered_direct_tids.add(-1)
                                rendered_cats.add(cat)
                                untracked_cats.add(cat)

                # 배경 영역 렌더 (체크리스트 VLM 박스 + 처리 중 VLM 박스)
                for cat in region_cats:
                    boxes = _sample_boxes(
                        region_samples.get(cat), sidx, frame_idx=done,
                        stride=keyframe_stride, interpolate=True)
                    for b in _dedupe_boxes(boxes):
                        x1, y1, x2, y2 = b["xyxy"]
                        if not plausible_box(cat, x1, y1, x2, y2, W, H):
                            continue
                        if not region_has_detail(frame, (x1, y1, x2, y2),
                                                 cat):
                            continue
                        if render_region(frame, x1, y1, x2, y2,
                                         style=style, blocks=blocks,
                                         blur=blur,
                                         pad=24 if cat == "license_plate" else 8):
                            rendered_region_cats.add(cat)
                            rendered_cats.add(cat)

                if writer is not None:
                    writer.write(frame)
                else:
                    out.write(frame)
                done += 1
                if progress_fn and total > 0:
                    try:
                        progress_fn(
                            progress_base + (1.0 - progress_base)
                            * min(1.0, done / max(1, total)),
                            done, total, time.time() - t0)
                    except Exception:
                        pass
            if done == 0:
                raise RuntimeError("처리된 프레임이 없습니다.")
        except Exception:
            if writer is not None:
                writer.abort()
            else:
                cleanup_tmp(tmp_noaudio)
            raise
        finally:
            if cap2 is not None:
                cap2.release()

        if writer is not None:
            writer.close()
        else:
            out.release()
            try:
                mux_audio_h264(tmp_noaudio, input_path, output_path,
                               preset=encode_preset)
            except Exception:
                cleanup_tmp(output_path)
                shutil.copy(tmp_noaudio, output_path)
            finally:
                cleanup_tmp(tmp_noaudio)

        if blanket_cats:
            for tid in rendered_tids:
                cat_on.setdefault(tid, {})
                for cat in blanket_cats:
                    cat_on[tid].setdefault(cat, True)
            rendered_cats.update(blanket_cats)

        stats = {}
        for tid in sorted(set(votes) | set(cat_on) | rendered_tids):
            _by_cat = votes.get(tid, {})
            stats[tid] = {
                "mosaic": tid in rendered_tids,
                "direct": tid in rendered_direct_tids,
                "frames": int(track_frames.get(tid, 0)),
                "selector": "vlm",
                "query_raw": query_raw,
                "targets": list(targets),
                "hit_targets": global_hit_targets.get(tid, []),
                "cat_on": {k: v for k, v in cat_on.get(tid, {}).items()
                           if v},
                "frac": global_detail.get(tid, {}).get("frac", 0.0),
                "sample": global_detail.get(tid, {}).get("sample", {}),
            }
        stats["_regions"] = len(rendered_region_cats)
        stats["_cats"] = sorted(rendered_cats)
        stats["_untracked"] = sorted(untracked_cats)
        stats["_vlm_calls"] = int(vlm_budget.used_this_video)
        stats["_vlm_errors"] = int(vlm_errors)
        return stats


def mosaic_video(input_path, output_path, targets=None, invert=False,
                 conf=0.20, model_name="yolo11n-seg.pt", imgsz=480,
                 blocks=30, style="pixel", blur=61, ultra=False,
                 area_mode="full", encode_preset="veryfast",
                 keyframe_stride=15, progress_fn=None,
                 query_raw="", api_key=None, vlm_model=VLM_MODEL,
                 vlm_base_url=None, region_priors=None, person_priors=None,
                 cache_path=None, use_cache=False,
                 vlm_call_limit=None, mask_all_persons=False):
    """호환 래퍼."""
    job = MosaicJob(input_path, output_path, targets=targets, invert=invert,
                    conf=conf, model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur, ultra=ultra,
                    area_mode=area_mode, encode_preset=encode_preset,
                    keyframe_stride=keyframe_stride, query_raw=query_raw,
                    api_key=api_key, vlm_model=vlm_model,
                    vlm_base_url=vlm_base_url,
                    region_priors=region_priors,
                    person_priors=person_priors,
                    cache_path=cache_path, use_cache=use_cache,
                    vlm_call_limit=vlm_call_limit,
                    mask_all_persons=mask_all_persons)
    return job.run(progress_fn)
