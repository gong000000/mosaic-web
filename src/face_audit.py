"""렌더 결과의 얼굴 마스크 누락 자동 점검/보정.

입력 프레임에서 검출한 얼굴과 같은 위치의 출력 프레임을 비교한다.
출력이 입력과 거의 같으면 모자이크가 적용되지 않은 것으로 보고,
필요하면 해당 프레임에 강한 사각형 모자이크를 덧그려 다시 인코딩한다.

한계: 입력의 포스터/간판 얼굴이나 아주 작은 원거리 얼굴도 검출될 수
있으므로 결과는 "확인 필요" 참고 지표다. 얼굴 크기/점수 기준을 통과한
고신뢰 항목만 자동 보정한다.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .errors import RenderCancelled
from .faces import detect_faces, detect_faces_rotated, detect_faces_yolo
from .models import MODEL_LOCK, get_model
from .render import (FFmpegFrameWriter, cleanup_tmp, mux_audio_h264,
                     render_region)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    ba = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return inter / max(1.0, aa + ba - inter)


def _probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0, 0.0, 0, 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return total, fps, w, h


def _expand_box(box, w, h, side=0.20, top=0.40, bottom=0.20):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (max(0, int(x1 - bw * side)),
            max(0, int(y1 - bh * top)),
            min(w, int(x2 + bw * side)),
            min(h, int(y2 + bh * bottom)))


def _person_boxes(model, frame):
    """감사용 사람 박스. YOLO 사람 검출 실패 시 빈 목록."""
    if model is None or frame is None or frame.size == 0:
        return []
    try:
        with MODEL_LOCK:
            pr = list(model.predict(
                source=frame, conf=0.25, classes=[0], verbose=False,
                imgsz=640, max_det=30))[0]
        if pr.boxes is None or len(pr.boxes) == 0:
            return []
        return [[float(v) for v in row[:4]]
                for row in pr.boxes.xyxy.cpu().numpy()]
    except Exception:
        return []


def _inside_person(box, persons, margin=0.18):
    if not persons:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for px1, py1, px2, py2 in persons:
        if px2 - px1 < 8 or py2 - py1 < 16:
            continue
        mw, mh = (px2 - px1) * margin, (py2 - py1) * margin
        if (px1 - mw <= cx <= px2 + mw
                and py1 - mh <= cy <= py2 + mh):
            return True
    return False


def _person_size_ok(box, persons, max_w=0.75, max_h=0.55):
    """사람 박스 대비 얼굴이 지나치게 크면 옷/몸통 오검출로 본다."""
    if not persons:
        return True
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        fw, fh = x2 - x1, y2 - y1
    except (TypeError, ValueError):
        return False
    for px1, py1, px2, py2 in persons:
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 16:
            continue
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return fw <= pw * max_w and fh <= ph * max_h
    return True


def _face_position_ok(box, persons, score):
    """감사 보정도 렌더러와 같은 얼굴 위치 게이트를 쓴다.

    팔/가슴/옷자락의 YOLO 오검출을 사후 보정으로 다시 칠하는 것을 막는다.
    """
    if not persons:
        return True
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        score = float(score)
    except (TypeError, ValueError):
        return False
    for px1, py1, px2, py2 in persons:
        pw, ph = px2 - px1, py2 - py1
        if pw < 8 or ph < 16:
            continue
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
        rel = (cy - py1) / max(1.0, ph)
        if score >= 0.65:
            return rel <= 0.60
        wide = pw > 1.15 * max(1.0, ph)
        return rel <= (0.30 if wide else 0.18)
    return True


def audit_faces(input_path, output_path, cancel_fn=None):
    """입력/출력 얼굴을 비교해 마스크 누락 의심 항목을 돌려준다."""
    total_in, _fps, _w, _h = _probe(input_path)
    total_out, _ofps, ow, oh = _probe(output_path)
    total = min(total_in, total_out)
    if total <= 0 or ow <= 0 or oh <= 0:
        return {"checked": 0, "suspicious": 0, "fixable": 0,
                "frames": [], "error": "영상 프레임을 읽을 수 없습니다."}
    max_samples = max(10, _env_int("MOSAIC_FACE_AUDIT_SAMPLES", 240))
    step = max(1, total // max_samples)
    threshold = max(0.1, _env_float("MOSAIC_FACE_AUDIT_DIFF", 3.0))
    rotate = os.environ.get("MOSAIC_FACE_ROTATE", "1") != "0"
    person_model = None
    if os.environ.get("MOSAIC_FACE_AUDIT_PERSON", "1") != "0":
        try:
            person_model = get_model(
                os.environ.get("MOSAIC_AUDIT_PERSON_MODEL",
                               "yolo11n-seg.pt"))
        except Exception:
            person_model = None

    cap_i = cv2.VideoCapture(input_path)
    cap_o = cv2.VideoCapture(output_path)
    checked = 0
    suspicious = []
    fixable = []
    try:
        idx = 0
        while idx < total:
            ok_i, fi = cap_i.read()
            ok_o, fo = cap_o.read()
            if not ok_i or not ok_o:
                break
            if cancel_fn is not None and cancel_fn():
                raise RenderCancelled()
            if idx % step == 0:
                dets = [(b, s, False) for b, s in (
                    detect_faces_rotated(fi) if rotate
                    else detect_faces(fi))]
                try:
                    for _yb, _ys in detect_faces_yolo(fi):
                        if any(_box_iou(_yb, _db) > 0.45
                               for _db, _ds, _iy in dets):
                            continue
                        dets.append((_yb, _ys, True))
                except Exception:
                    pass
                persons = (_person_boxes(person_model, fi)
                           if person_model is not None else None)
                # A correctly masked face should not still be detectable in
                # the rendered frame.  Check the output too; this catches the
                # "last frames lost tracking" case where mean diff alone is
                # just above the threshold.
                try:
                    out_dets = (detect_faces_rotated(fo) if rotate
                                else detect_faces(fo))
                except Exception:
                    out_dets = []
                for box, score, is_yolo in dets:
                    try:
                        x1, y1, x2, y2 = [int(round(float(v)))
                                          for v in box[:4]]
                    except (TypeError, ValueError):
                        continue
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(ow, x2)
                    y2 = min(oh, y2)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    score = float(score)
                    if score < (0.55 if is_yolo else 0.70):
                        continue
                    area_ratio = ((x2 - x1) * (y2 - y1)
                                  / float(max(1, ow * oh)))
                    # 바퀴/건물 무늬/포스터를 얼굴로 오검출하는 경우가 있다.
                    # 사람 박스 밖의 작은 얼굴은 고신뢰가 아니면 무시하고,
                    # 사람 안이라도 몸통 전체 크기면 얼굴로 보지 않는다.
                    if person_model is not None:
                        inside = _inside_person(box, persons)
                        if score < 0.85 and not inside:
                            continue
                        if (not inside and score < 0.95
                                and area_ratio < 0.05):
                            continue
                        if not _person_size_ok(box, persons):
                            continue
                        if not _face_position_ok(box, persons, score):
                            continue
                    a = fi[y1:y2, x1:x2]
                    b = fo[y1:y2, x1:x2]
                    if a.size == 0 or b.size == 0:
                        continue
                    if a.shape != b.shape:
                        b = cv2.resize(b, (a.shape[1], a.shape[0]))
                    diff = float(np.mean(cv2.absdiff(a, b)))
                    hh, ww = a.shape[:2]
                    if hh >= 12 and ww >= 12:
                        ux1, ux2 = int(ww * 0.15), int(ww * 0.85)
                        uy1, uy2 = int(hh * 0.08), int(hh * 0.56)
                        ly1, ly2 = int(hh * 0.44), int(hh * 0.92)
                        du = float(np.mean(cv2.absdiff(
                            a[uy1:uy2, ux1:ux2], b[uy1:uy2, ux1:ux2])))
                        dl = float(np.mean(cv2.absdiff(
                            a[ly1:ly2, ux1:ux2], b[ly1:ly2, ux1:ux2])))
                        # 얼굴 위/아래 중 한쪽만 가려진 부분 마스크도 잡는다.
                        diff = min(diff, du, dl)
                    visible = (diff < 8.0 and any(
                        _box_iou(box, ob) >= 0.40 and float(oscore) >= 0.60
                        for ob, oscore in out_dets))
                    checked += 1
                    if diff >= threshold and not visible:
                        continue
                    row = {"frame": int(idx),
                           "box": [int(x1), int(y1), int(x2), int(y2)],
                           "score": round(score, 3),
                           "source": "yolo" if is_yolo else "yunet",
                           "diff": round(float(diff), 2),
                           "visible_in_output": bool(visible)}
                    suspicious.append(row)
                    side = min(x2 - x1, y2 - y1)
                    if side >= 28 and (
                            (is_yolo and score >= 0.55)
                            or score >= 0.78):
                        fixable.append(row)
            idx += 1
    finally:
        cap_i.release()
        cap_o.release()
    ratio = (checked - len(suspicious)) / float(checked) if checked else 1.0
    return {
        "version": 2,
        "person_gate": person_model is not None,
        "checked": int(checked),
        "suspicious": len(suspicious),
        "fixable": len(fixable),
        "ratio": round(float(ratio), 4),
        "step": int(step),
        "threshold": float(threshold),
        "frames": suspicious[:100],
        "fixes": fixable[:2000],
    }


def fix_audited_faces(output_path, fixes, *, style="pixel", blocks=30,
                      blur=61, preset="veryfast", strip_audio=False):
    """감사에서 나온 프레임/박스에 강한 사각형 모자이크를 덧그린다."""
    fixes = [f for f in (fixes or []) if f.get("box")]
    if not fixes:
        return 0
    total, fps, w, h = _probe(output_path)
    if total <= 0 or w <= 0 or h <= 0:
        return 0
    by_frame = {}
    for row in fixes:
        try:
            frame = int(row["frame"])
        except (TypeError, ValueError, KeyError):
            continue
        for k in range(-2, 3):
            fidx = frame + k
            if 0 <= fidx < total:
                by_frame.setdefault(fidx, []).append(row["box"])
    tmp = output_path + ".auditfix.mp4"
    muxed = output_path + ".audited.mp4"
    cleanup_tmp(tmp)
    cleanup_tmp(muxed)
    cap = cv2.VideoCapture(output_path)
    writer = None
    fixed = 0
    try:
        writer = FFmpegFrameWriter(tmp, w, h, fps, audio_src=None,
                                   preset=preset)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = by_frame.get(idx)
            if boxes:
                uniq = []
                for box in boxes:
                    if any(_box_iou(box, u) > 0.85 for u in uniq):
                        continue
                    uniq.append(box)
                for box in uniq:
                    x1, y1, x2, y2 = _expand_box(box, w, h)
                    if render_region(frame, x1, y1, x2, y2,
                                     style=style,
                                     blocks=min(60, int(blocks) + 10),
                                     blur=max(int(blur), 81), pad=0):
                        fixed += 1
            writer.write(frame)
            idx += 1
        writer.close()
        writer = None
        # Windows에서는 원본을 열어 둔 채 os.replace 하면 잠금 오류가 난다.
        cap.release()
        cap = None
        if strip_audio:
            os.replace(tmp, output_path)
        else:
            try:
                mux_audio_h264(tmp, output_path, muxed, preset=preset,
                               copy_video=True)
                os.replace(muxed, output_path)
            except Exception:
                os.replace(tmp, output_path)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.abort()
            except Exception:
                pass
        cleanup_tmp(tmp)
        cleanup_tmp(muxed)
    return int(fixed)
