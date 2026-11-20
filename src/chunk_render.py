"""긴 영상의 구간별 렌더 + 완료 구간 체크포인트.

Modal GPU는 preemption될 수 있고, 긴 영상은 한 번에 끝나지 않을 수 있다.
입력을 키프레임 단위 구간으로 나눠 각각 렌더링하고, 끝난 구간은
`outputs/.chunks/<token>/state.json`에 기록한다. 프로세스가 중간에 죽어도
다음 실행에서 완료 구간을 건너뛰고 남은 구간부터 이어서 렌더한다.

- 입력 분할은 `-c copy`라 원본을 다시 인코딩하지 않는다.
- 각 구간은 기존 모자이크 파이프라인을 그대로 사용한다.
- 구간 경계의 VLM prior/수동 영역 시간은 구간 시작만큼 이동한다.
- 마지막에 구간 영상을 이어 붙이고 원본 오디오를 한 번만 믹싱한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import replace

import cv2

from .config import OUTPUT_DIR
from .errors import RenderCancelled
from .render import (cleanup_tmp, concat_videos, mux_audio_h264,
                     segment_video)

log = logging.getLogger("mosaic.chunk")


def _state_dir(output_path):
    token = hashlib.sha1(
        os.path.abspath(output_path).encode("utf-8")).hexdigest()[:16]
    base = os.path.join(os.path.dirname(os.path.abspath(output_path)),
                        ".chunks")
    return os.path.join(base, token)


def cleanup_chunk_state(output_path):
    """해당 결과의 구간 체크포인트를 지운다(취소/성공 시)."""
    shutil.rmtree(_state_dir(output_path), ignore_errors=True)
    cleanup_tmp(output_path + ".joined.mp4")


def cleanup_stale_chunks(max_age=7 * 24 * 3600):
    """오래 남은 구간 체크포인트 디렉터리를 정리한다."""
    base = os.path.join(OUTPUT_DIR, ".chunks")
    if not os.path.isdir(base):
        return 0
    cutoff = time.time() - max(3600.0, float(max_age))
    removed = 0
    for name in os.listdir(base):
        path = os.path.join(base, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _signature(plan, input_path, chunk_seconds, model_name, imgsz, conf,
               preset, query_raw):
    try:
        size = os.path.getsize(input_path)
    except OSError:
        size = 0
    payload = {
        "input": os.path.abspath(input_path),
        "size": int(size),
        "chunk_seconds": round(float(chunk_seconds), 3),
        "model": str(model_name),
        "imgsz": int(imgsz),
        "conf": round(float(conf), 3),
        "preset": str(preset),
        "query": str(query_raw or ""),
        "targets": list(plan.targets or ()),
        "settings": plan.settings.as_dict(),
        "mask_all": bool(plan.mask_all_persons),
        "region_priors": plan.region_priors or {},
        "person_priors": plan.person_priors or {},
        "manual": plan.manual_regions or [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                     default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0, 0.0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return total, fps


def _shift_priors(priors, start_frame, end_frame, margin, region):
    """전역 프레임 기준 prior를 구간 로컬 프레임 기준으로 옮긴다.

    region(주소/간판)은 구간 앞에서 한 번이라도 보인 박스가 계속 유효할 수
    있으므로 가장 가까운 이전 prior 하나를 0프레임에 남긴다. 사람/번호판은
    오래된 위치를 새 구간에 심으면 오검출이 생기므로 구간 근처만 쓴다.
    """
    out = {}
    for cat, by_frame in (priors or {}).items():
        rows = {}
        earlier = []
        for key, boxes in (by_frame or {}).items():
            try:
                fidx = int(key)
            except (TypeError, ValueError):
                continue
            if fidx < start_frame - margin:
                if region:
                    earlier.append((fidx, boxes))
                continue
            if fidx > end_frame + margin:
                continue
            local = str(max(0, fidx - start_frame))
            rows.setdefault(local, []).extend(boxes or [])
        if region and earlier:
            _fidx, boxes = max(earlier, key=lambda item: item[0])
            rows.setdefault("0", []).extend(boxes or [])
        if rows:
            out[cat] = rows
    return out


def _shift_manual(regions, start_sec, duration_sec):
    out = []
    for r in regions or []:
        try:
            t0 = float(r.get("t0", 0.0))
            t1 = float(r.get("t1", 0.0))
        except (TypeError, ValueError):
            continue
        if t1 <= 0:
            out.append(dict(r))
            continue
        nt0 = t0 - start_sec
        nt1 = t1 - start_sec
        if nt1 <= 0 or nt0 >= duration_sec:
            continue
        row = dict(r)
        row["t0"] = max(0.0, nt0)
        row["t1"] = min(float(duration_sec), nt1)
        out.append(row)
    return out


def _merge_stats(dst, src):
    for key, value in (src or {}).items():
        if key.startswith("_") and isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value
        elif key not in dst:
            dst[key] = value
    return dst


def _valid(path):
    try:
        return bool(path and os.path.exists(path)
                    and os.path.getsize(path) > 0)
    except OSError:
        return False


def _render_one(plan, seg, total, fps, stride, work_dir, index, *,
                model_name, imgsz, conf, preset, progress_fn, query_raw,
                api_key, vlm_model, vlm_base_url, cancel_fn):
    part_out = os.path.join(work_dir, f"render_{index:04d}.mp4")
    start_sec = float(seg.get("start") or 0.0)
    end_sec = float(seg.get("end") or 0.0)
    duration = max(0.001, end_sec - start_sec)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(int(total), max(start_frame + 1,
                                    int(round(end_sec * fps))))
    margin = max(2 * int(stride), int(round(fps * 2.0)))

    region_priors = _shift_priors(
        plan.region_priors, start_frame, end_frame, margin, region=True)
    person_priors = _shift_priors(
        plan.person_priors, start_frame, end_frame, margin, region=False)
    manual = _shift_manual(plan.manual_regions, start_sec, duration)
    chunk_plan = replace(plan, region_priors=region_priors,
                         person_priors=person_priors,
                         manual_regions=manual)

    def chunk_progress(frac, _done, _local_total, elapsed):
        if progress_fn is None:
            return
        try:
            frac = min(1.0, max(0.0, float(frac)))
            done = start_frame + int(round(
                frac * max(0, end_frame - start_frame)))
            progress_fn(min(1.0, done / max(1, total)), done, total, elapsed)
        except Exception:
            pass

    stats = chunk_plan.render(
        seg["path"], part_out,
        model_name=model_name, imgsz=imgsz, stride=stride, conf=conf,
        preset=preset, progress_fn=chunk_progress, query_raw=query_raw,
        api_key=api_key, vlm_model=vlm_model, vlm_base_url=vlm_base_url,
        use_cache=False, vlm_call_limit=0, cancel_fn=cancel_fn,
        mux_audio=False)
    if not _valid(part_out):
        raise RuntimeError(f"구간 {index + 1} 결과가 만들어지지 않았습니다.")
    return part_out, stats


def render_resumable(plan, input_path, output_path, *, model_name, imgsz,
                     stride, conf, preset, progress_fn=None, query_raw="",
                     api_key=None, vlm_model=None, vlm_base_url=None,
                     final_mux_audio=True, cancel_fn=None):
    """긴 영상이면 구간 체크포인트 렌더, 짧으면 기존 단일 렌더."""
    try:
        chunk_seconds = float(os.environ.get("MOSAIC_CHUNK_SEC", "120"))
    except (TypeError, ValueError):
        chunk_seconds = 120.0
    total, fps = _probe_video(input_path)
    if (chunk_seconds <= 0 or total <= 0 or fps <= 0
            or total <= int(round(chunk_seconds * fps) * 1.25)):
        return plan.render(
            input_path, output_path, model_name=model_name, imgsz=imgsz,
            stride=stride, conf=conf, preset=preset,
            progress_fn=progress_fn, query_raw=query_raw, api_key=api_key,
            vlm_model=vlm_model, vlm_base_url=vlm_base_url,
            use_cache=False, vlm_call_limit=0, cancel_fn=cancel_fn,
            mux_audio=final_mux_audio)

    work_dir = _state_dir(output_path)
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(work_dir, "state.json")
    sig = _signature(plan, input_path, chunk_seconds, model_name, imgsz,
                     conf, preset, query_raw)
    state = _load_state(state_path)
    if state.get("signature") != sig:
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        state = {"signature": sig, "segments": [], "completed": []}

    segments = state.get("segments") or []
    if (not segments
            or any(not _valid(row.get("path")) for row in segments)):
        try:
            raw_segments = segment_video(input_path, work_dir, chunk_seconds)
        except Exception:
            log.exception("영상 분할 실패 - 단일 렌더로 대체")
            shutil.rmtree(work_dir, ignore_errors=True)
            return plan.render(
                input_path, output_path, model_name=model_name, imgsz=imgsz,
                stride=stride, conf=conf, preset=preset,
                progress_fn=progress_fn, query_raw=query_raw,
                api_key=api_key, vlm_model=vlm_model,
                vlm_base_url=vlm_base_url, use_cache=False,
                vlm_call_limit=0, cancel_fn=cancel_fn,
                mux_audio=final_mux_audio)
        segments = [{"path": path, "start": start, "end": end}
                    for path, start, end in raw_segments]
        state["segments"] = segments
        state["completed"] = []
        _save_state(state_path, state)

    # 키프레임이 너무 sparse해서 한 덩어리로만 잘리면 체크포인트 이점이 없다.
    longest = max((float(s.get("end") or 0) - float(s.get("start") or 0)
                   for s in segments), default=0.0)
    if len(segments) <= 1 or longest > max(chunk_seconds * 3.0, 300.0):
        log.info("구간 분할이 충분하지 않아 단일 렌더로 대체")
        shutil.rmtree(work_dir, ignore_errors=True)
        return plan.render(
            input_path, output_path, model_name=model_name, imgsz=imgsz,
            stride=stride, conf=conf, preset=preset,
            progress_fn=progress_fn, query_raw=query_raw, api_key=api_key,
            vlm_model=vlm_model, vlm_base_url=vlm_base_url,
            use_cache=False, vlm_call_limit=0, cancel_fn=cancel_fn,
            mux_audio=final_mux_audio)

    completed = set(int(v) for v in (state.get("completed") or []))
    parts = []
    stats = {}
    saved_stats = state.get("stats") or {}
    for i, seg in enumerate(segments):
        if i in completed and _valid(
                os.path.join(work_dir, f"render_{i:04d}.mp4")):
            for key, value in (saved_stats.get(str(i)) or {}).items():
                if isinstance(value, (int, float)):
                    stats[key] = stats.get(key, 0) + value
    try:
        for i, seg in enumerate(segments):
            part_out = os.path.join(work_dir, f"render_{i:04d}.mp4")
            if i in completed and _valid(part_out):
                parts.append(part_out)
                if progress_fn is not None:
                    end_frame = min(total, max(0, int(round(
                        float(seg.get("end") or 0.0) * fps))))
                    progress_fn(min(1.0, end_frame / max(1, total)),
                                end_frame, total, 0.0)
                continue
            if cancel_fn is not None and cancel_fn():
                raise RenderCancelled()
            part_out, chunk_stats = _render_one(
                plan, seg, total, fps, stride, work_dir, i,
                model_name=model_name, imgsz=imgsz, conf=conf, preset=preset,
                progress_fn=progress_fn, query_raw=query_raw, api_key=api_key,
                vlm_model=vlm_model, vlm_base_url=vlm_base_url,
                cancel_fn=cancel_fn)
            _merge_stats(stats, chunk_stats)
            parts.append(part_out)
            completed.add(i)
            state["completed"] = sorted(completed)
            state.setdefault("stats", {})[str(i)] = {
                key: float(value) for key, value in (chunk_stats or {}).items()
                if key in ("_faces", "_regions")
                and isinstance(value, (int, float))}
            _save_state(state_path, state)

        if cancel_fn is not None and cancel_fn():
            raise RenderCancelled()
        joined = output_path + ".joined.mp4"
        concat_videos(parts, joined, preset=preset)
        try:
            if final_mux_audio:
                mux_audio_h264(joined, input_path, output_path,
                               preset=preset, copy_video=True)
            else:
                cleanup_tmp(output_path)
                shutil.copy(joined, output_path)
        except Exception:
            log.exception("최종 오디오 믹싱 실패 - 무음으로 저장")
            cleanup_tmp(output_path)
            shutil.copy(joined, output_path)
        finally:
            cleanup_tmp(joined)
        if not _valid(output_path):
            raise RuntimeError("최종 결과 영상이 만들어지지 않았습니다.")
        cleanup_chunk_state(output_path)
        return stats
    except RenderCancelled:
        cleanup_chunk_state(output_path)
        raise
    except Exception:
        # 완료된 구간은 남겨 두고, 재시도 때 이어서 렌더한다.
        raise
