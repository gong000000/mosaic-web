"""분석용 샘플 프레임 선택.

균등 샘플링만 쓰면 영상 내용이 바뀌는 지점을 놓친다. 10초 영상에서
2.2초에만 신분증이 스치면 0/3.3/6.6/9.9초 샘플에는 잡히지 않는다.
그래서 아래 세 가지를 합쳐 VLM에 보낼 프레임을 고른다.

  1) 균등 기본 샘플   - 영상 전체를 빠짐없이 훑는다
  2) 장면 전환 샘플   - 화면이 크게 바뀐 지점 직후
  3) 움직임 피크 샘플 - 변화량이 국소 최댓값인 지점 (컷 없는 CCTV용)

컷은 '균등 샘플 + 움직임 피크'로도 대체되지 않는다. 카메라 흔들림을
컷으로 오탐하지 않도록 임계값은 영상마다 중앙값/MAD로 적응적으로 정한다.
선택 개수는 상한을 두어 VLM 호출 수와 시간을 예측 가능하게 유지한다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# 비교용 서명 크기. 작을수록 빠르고, 밝기 변화에 덜 민감하다.
SIG_W, SIG_H = 64, 36

# 움직임 변화량의 절대 하한. 영상별로 계산한 임계값보다 높게 잡힌다.
CUT_ABS_FLOOR = 18.0
PEAK_ABS_FLOOR = 1.0

# 중앙값 대비 배수. MAD만 쓰면 '아주 조용한 영상'에서 노이즈까지
# 피크로 잡히고, 절대값만 쓰면 '조용한 CCTV'의 작은 움직임을 놓친다.
# 두 조건의 최댓값을 써서 양쪽 함정을 피한다.
CUT_MED_K = 5.0
PEAK_MED_K = 2.0

# 최고 변화량 대비 비율. 컷은 '최대치의 절반 이상', 피크는 '삼분의 일
# 이상'일 때만 유효한 이벤트로 본다.
CUT_MAX_K = 0.45
PEAK_MAX_K = 0.35


def frame_signature(frame_bgr):
    """비교용 저해상도 흑백 서명."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)


def scan_motion(video_path, scan_max=480):
    """영상을 저해상도로 훑어 프레임별 움직임 변화량을 구한다.

    frames를 전부 저장하지 않고 서명만 계산하므로 메모리를 거의 쓰지
    않는다. 최대 scan_max개 프레임만 정밀히 읽고, 그 사이는 grab()으로
    건너뛴다.

    returns (total, motion) - motion는 [(frame_idx, 변화량), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = 1
    if total > 0:
        step = max(1, int(math.ceil(total / float(max(1, scan_max)))))
    motion = []
    prev = None
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size:
                    sig = frame_signature(frame)
                    if prev is not None:
                        motion.append(
                            (idx, float(np.mean(cv2.absdiff(prev, sig)))))
                    prev = sig
            idx += 1
            if total and idx > total + step:
                break
    except Exception:
        pass
    finally:
        cap.release()
    if not total:
        total = idx
    return total, motion


def _median_mad(values):
    """중앙값과 MAD(절대편차 중앙값). 컷/피크 임계값 계산에 쓴다."""
    if values is None or len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def _too_close(idx, others, gap):
    for o in others:
        if abs(int(idx) - int(o)) < gap:
            return True
    return False


def plan_samples(video_path, base_count=4, max_count=10, max_cuts=4,
                  max_peaks=3, min_gap_ratio=0.05, scan_max=480):
    """분석에 쓸 프레임 인덱스를 고른다.

    base_count : 균등 기본 샘플 수 (항상 포함)
    max_count  : 전체 상한. 이 값을 넘으면 움직임 피크 → 컷 순으로 버린다.
    max_cuts   : 장면 전환 샘플 최대 개수
    max_peaks  : 움직임 피크 샘플 최대 개수
    min_gap_ratio : 샘플 사이 최소 간격 (영상 길이 대비)

    returns dict(indices, cuts, peaks, total, motion)
    """
    base_count = max(1, int(base_count))
    max_count = max(base_count, int(max_count))
    max_cuts = max(0, int(max_cuts))
    max_peaks = max(0, int(max_peaks))

    total, motion = scan_motion(video_path, scan_max=scan_max)
    empty = {"indices": [], "cuts": [], "peaks": [], "all_cuts": [],
             "total": total, "motion": motion}
    if total <= 0:
        return empty
    if total <= base_count:
        # 영상이 짧으면 모든 프레임을 본다.
        return {"indices": list(range(total)), "cuts": [], "peaks": [],
                "all_cuts": [], "total": total, "motion": motion}

    base = sorted({int(round(v)) for v in
                   np.linspace(0, total - 1, base_count)})
    scores = [s for _i, s in motion]
    med, mad = _median_mad(scores)
    top = max(scores) if scores else 0.0
    p98 = float(np.percentile(np.asarray(scores, dtype=np.float32), 98)) \
        if scores else 0.0
    cut_thr = max(CUT_ABS_FLOOR, med * CUT_MED_K, top * CUT_MAX_K, p98)
    peak_thr = max(PEAK_ABS_FLOOR, med * PEAK_MED_K, top * PEAK_MAX_K)

    min_gap = max(1, int(round(total * float(min_gap_ratio))))

    # 1) 장면 전환: 변화량이 컷 임계값 이상인 지점. 변화는 '직후' 프레임에
    #    나타나므로 그 프레임을 고른다.
    cut_candidates = [(i, s) for i, s in motion if s >= cut_thr]
    all_cuts = [int(min(total - 1, max(0, i))) for i, _s in cut_candidates]
    cuts = []
    for idx, _score in sorted(cut_candidates, key=lambda x: -x[1]):
        target = int(min(total - 1, max(0, idx)))
        if _too_close(target, base + cuts, min_gap):
            continue
        cuts.append(target)
        if len(cuts) >= max_cuts:
            break

    # 2) 움직임 피크: 변화량의 국소 최댓값. 컷 없는 CCTV에서 사람이
    #    스치는 지점을 잡는다. 같은 움직임의 연속 프레임이 중복으로
    #    잡히지 않도록 이웃 2프레임까지 비교한다.
    peak_candidates = []
    for k, (idx, score) in enumerate(motion):
        if score < peak_thr:
            continue
        lo = max(0, k - 2)
        hi = min(len(motion), k + 3)
        if score < max(motion[j][1] for j in range(lo, hi)):
            continue
        peak_candidates.append((int(idx), score))
    peaks = []
    for idx, _score in sorted(peak_candidates, key=lambda x: -x[1]):
        if _too_close(idx, base + cuts + peaks, min_gap):
            continue
        peaks.append(int(idx))
        if len(peaks) >= max_peaks:
            break

    # 우선순위: 기본 샘플 > 컷 > 피크. 상한을 넘으면 낮은 우선순위부터 버린다.
    picked = list(base)
    picked += cuts
    picked += peaks
    if len(picked) > max_count:
        keep = list(base)
        for idx in cuts:
            if len(keep) >= max_count:
                break
            keep.append(idx)
        for idx in peaks:
            if len(keep) >= max_count:
                break
            keep.append(idx)
        picked = keep

    return {"indices": sorted(set(int(i) for i in picked)),
            "cuts": sorted(set(cuts)), "peaks": sorted(set(peaks)),
            "all_cuts": sorted(set(all_cuts)),
            "total": total, "motion": motion}


def describe(plan):
    """UI/로그용 한 줄 요약."""
    if not plan:
        return "샘플 없음"
    return (f"총 {plan.get('total', 0)}프레임 중 {len(plan.get('indices', []))}장 "
            f"선택 (컷 {len(plan.get('cuts', []))}, 피크 "
            f"{len(plan.get('peaks', []))})")
