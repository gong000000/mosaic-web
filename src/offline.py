"""업로드된 영상은 미래 프레임도 쓸 수 있으므로, 1패스에서 모은 프레임별
박스를 후처리한다(지연 없는 평활 + 첫 등장 프레임 역방향 채움).

frames: 프레임 인덱스 순서의 리스트. 각 원소는 {track_id: [x1,y1,x2,y2]}.
cuts:   장면 전환이 일어난 프레임 인덱스 집합(그 프레임부터 새 장면).
"""

import numpy as np


def _cut_between(cuts, start, end):
    """(start, end] 구간에 장면 전환이 있으면 True."""
    return any(start < c <= end for c in cuts)


def backfill_and_smooth(frames, cuts=(), back=10, win=3):
    n = len(frames)
    cuts = set(cuts or ())
    by_tid = {}
    for i, boxes in enumerate(frames):
        for tid, box in boxes.items():
            by_tid.setdefault(tid, {})[i] = [float(v) for v in box]

    out = [dict() for _ in range(n)]
    for tid, seq in by_tid.items():
        seq = dict(seq)
        first = min(seq)
        # 1) 처음 검출되기 전 몇 프레임도 같은 박스로 가린다.
        for f in range(max(0, first - back), first):
            if not _cut_between(cuts, f, first):
                seq[f] = list(seq[first])
        idx = sorted(seq)
        # 2) 앞뒤 win 프레임 박스의 합집합(장면 전환은 넘지 않는다).
        #    평균 대신 합집합이라 박스가 줄어들지 않고, 미래 프레임 위치까지
        #    포함해 빠르게 움직이는 얼굴도 뒤처지지 않는다(검출 지연 상쇄).
        for i in idx:
            raw = seq[i]
            u = list(raw)
            for j in range(i - win, i + win + 1):
                if j == i or j not in seq:
                    continue
                if _cut_between(cuts, min(i, j), max(i, j)):
                    continue
                b = seq[j]
                u = [min(u[0], b[0]), min(u[1], b[1]),
                     max(u[2], b[2]), max(u[3], b[3])]
            out[i][tid] = u
    return out


def speeds(frames):
    """프레임별 {tid: 직전 프레임 대비 이동량(px)}. 여백 확대에 쓴다."""
    out = [dict() for _ in frames]
    for i in range(1, len(frames)):
        for tid, b in frames[i].items():
            p = frames[i - 1].get(tid)
            if p is None:
                continue
            dx = ((b[0] + b[2]) - (p[0] + p[2])) / 2.0
            dy = ((b[1] + b[3]) - (p[1] + p[3])) / 2.0
            out[i][tid] = (dx * dx + dy * dy) ** 0.5
    return out


def bridge_gaps(frames, cuts=(), max_gap=24, max_dist=1.0,
                return_links=False):
    """트랙 ID가 바뀌어 생긴 빈 구간을 보간으로 메운다.

    얼굴이 잠깐 안 잡혀 트랙이 끝나고 근처에서 새 트랙이 시작되면, 두 트랙의
    마지막/첫 박스를 선형 보간해 사이 프레임을 가린다(깜빡임/노출 방지).
    frames 는 제자리에서 갱신하지 않고 새 리스트를 돌려준다.
    """
    cuts = set(cuts or ())
    out = [dict(f) for f in frames]
    successors = set()
    span = {}
    for i, boxes in enumerate(frames):
        for tid in boxes:
            a, b = span.get(tid, (i, i))
            span[tid] = (min(a, i), max(b, i))
    for ta, (sa, ea) in span.items():
        best = None
        for tb, (sb, eb) in span.items():
            if tb == ta or sb <= ea or sb - ea - 1 > max_gap:
                continue
            if _cut_between(cuts, ea, sb):
                continue
            A, B = frames[ea][ta], frames[sb][tb]
            ca = ((A[0] + A[2]) / 2.0, (A[1] + A[3]) / 2.0)
            cb = ((B[0] + B[2]) / 2.0, (B[1] + B[3]) / 2.0)
            size = max(A[2] - A[0], A[3] - A[1], B[2] - B[0], B[3] - B[1], 1)
            d = (((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5) / size
            # 틈이 길수록 더 많이 움직였을 수 있으나, 빈 곳에 고스트가
            # 생기지 않게 허용 거리는 2.0으로 제한한다.
            if d <= min(2.0, max_dist + 0.3 * (sb - ea)) \
                    and (best is None or d < best[0]):
                best = (d, sb, A, B, tb)
        if best is None:
            continue
        _d, sb, A, B, tb_best = best
        successors.add(tb_best)
        gap = sb - ea
        for k, f in enumerate(range(ea + 1, sb), 1):
            if ta in out[f]:
                continue
            t = k / float(gap)
            out[f][ta] = [A[j] + (B[j] - A[j]) * t for j in range(4)]
    return (out, successors) if return_links else out


def temporal_smooth(frames, cuts=(), weights=(0.2, 0.6, 0.2)):
    """지연 없는 3프레임 중앙값 평활. 2패스에서만 쓴다(미래 프레임 기지).

    검출 박스가 한 프레임만 크게 튀는 경우(고개 돌림/가림)를 중앙값으로
    걸러낸다. 등속 이동에서는 중앙값이 현재 위치와 같아 지연이 0이고,
    크기가 갑자기 바뀌는 경우도 다음 프레임부터 반영된다.
    장면 전환을 넘지 않고, 앞뒤 프레임이 없는 경계는 그대로 둔다.
    """
    cuts = set(cuts or ())
    out = [dict(f) for f in frames]
    by_tid = {}
    for i, boxes in enumerate(frames):
        for tid in boxes:
            by_tid.setdefault(tid, set()).add(i)
    for tid, idxs in by_tid.items():
        idxs = sorted(idxs)
        for i in idxs:
            if i - 1 not in idxs or i + 1 not in idxs:
                continue
            if _cut_between(cuts, i - 1, i) or _cut_between(cuts, i, i + 1):
                continue
            try:
                a = [float(v) for v in frames[i - 1][tid]]
                b = [float(v) for v in frames[i][tid]]
                c = [float(v) for v in frames[i + 1][tid]]
            except (KeyError, TypeError, ValueError):
                continue
            # 좌표별 3프레임 중앙값. 한 프레임만 튄 값을 제거한다.
            out[i][tid] = [float(np.median([a[k], b[k], c[k]]))
                           for k in range(4)]
    return out


def missing_track_boxes(frames, back=8, max_gap=24):
    """프레임별 {tid: box} 에서 '검출이 없던' 프레임을 채운 박스만 돌려준다.

    트랙 첫 등장 전 back 프레임과 ID 전환으로 끊긴 max_gap 이하 구간을
    메운다. 반환: frame_index -> {tid: box}  (원래 있던 박스는 포함하지 않음)
    """
    bridged, succ = bridge_gaps(frames, (), max_gap=max_gap,
                                return_links=True)
    # ID 전환으로 이어진 뒤쪽 트랙은 '새로 등장'이 아니므로 역방향 채움을
    # 하지 않는다(옛 위치에 고스트가 남는 것 방지).
    pruned = [{t: b for t, b in f.items() if t not in succ} for f in bridged]
    filled = backfill_and_smooth(pruned, (), back=back, win=0)
    for i, f in enumerate(bridged):
        for t in succ:
            if t in f:
                filled[i][t] = f[t]
    extra = {}
    for i, (orig, new) in enumerate(zip(frames, filled)):
        taken = [list(b) for b in orig.values()]
        add = {}
        for tid in sorted(new):
            if tid in orig:
                continue
            box = new[tid]
            # 같은 사람이 다른 ID로 이미 검출/보간되어 있으면 중복(고스트)을
            # 만들지 않는다(ID 전환 시 옛 박스가 제자리에 남는 것 방지).
            if any(_overlap_min(box, t) > 0.5 for t in taken):
                continue
            add[tid] = box
            taken.append(list(box))
        if add:
            extra[i] = add
    return extra


def find_gap_ranges(raw_frames, filled_frames, fps=30.0):
    """원본 검출이 비어 있었는데 보간/역방향 채움으로 메워진 시간 구간 목록."""
    fps = float(fps if fps and fps > 0 else 30.0)
    gap_idxs = [
        i for i, (orig, new) in enumerate(zip(raw_frames, filled_frames))
        if not orig and new
    ]
    if not gap_idxs:
        return []
    ranges = []
    start = prev = gap_idxs[0]
    for idx in gap_idxs[1:]:
        if idx <= prev + 2:
            prev = idx
        else:
            ranges.append((round(start / fps, 2), round((prev + 1) / fps, 2)))
            start = prev = idx
    ranges.append((round(start / fps, 2), round((prev + 1) / fps, 2)))
    return ranges


def fill_same_tid_gaps(frames, cuts=(), max_gap=8, max_dist=1.0):
    """같은 트랙 ID가 잠깐 비었다가 다시 나타난 구간을 선형 보간으로 메운다.

    검출기 미스·게이트 흔들림으로 1~수 프레임이 비면 마스크가 깜빡이며
    얼굴이 노출된다. bridge_gaps가 서로 다른 ID 사이만 잇는 것과 달리,
    여기서는 같은 ID의 앞뒤 박스가 가까우면 사이를 메운다.
    장면 전환을 넘지 않고, 멀리 움직인 경우(다른 사람)는 잇지 않는다.
    """
    cuts = set(cuts or ())
    out = [dict(f) for f in frames]
    by_tid = {}
    for i, boxes in enumerate(frames):
        for tid in boxes:
            by_tid.setdefault(tid, []).append(i)
    for tid, idxs in by_tid.items():
        idxs = sorted(idxs)
        for a, b in zip(idxs, idxs[1:]):
            gap = b - a - 1
            if gap < 1 or gap > max_gap:
                continue
            if _cut_between(cuts, a, b):
                continue
            try:
                A = [float(v) for v in frames[a][tid]]
                B = [float(v) for v in frames[b][tid]]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            size = max(B[2] - B[0], B[3] - B[1],
                       A[2] - A[0], A[3] - A[1], 1.0)
            ca = ((A[0] + A[2]) / 2.0, (A[1] + A[3]) / 2.0)
            cb = ((B[0] + B[2]) / 2.0, (B[1] + B[3]) / 2.0)
            d = (((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5) / size
            if d > min(2.0, max_dist + 0.3 * (b - a)):
                continue
            for k, f in enumerate(range(a + 1, b), 1):
                if tid in out[f]:
                    continue
                t = k / float(b - a)
                out[f][tid] = [A[j] + (B[j] - A[j]) * t for j in range(4)]
    return out


def nearest_track_values(per_frame, n_frames, cuts=()):
    """트랙별 마지막 관측값을 가장 가까운 프레임에서 가져온다.

    얼굴 각도처럼 보간 박스에는 들어 있지 않은 속성이 프레임마다 0으로
    초기화되지 않게 한다. 장면 전환 경계는 넘지 않는다.
    """
    import bisect

    cuts = set(cuts or ())
    by_tid = {}
    for idx, values in enumerate(per_frame):
        for tid, value in (values or {}).items():
            by_tid.setdefault(tid, []).append((idx, value))
    out = [dict() for _ in range(n_frames)]
    for tid, points in by_tid.items():
        points.sort(key=lambda item: item[0])
        positions = [p[0] for p in points]
        for idx in range(n_frames):
            at = bisect.bisect_right(positions, idx)
            candidates = []
            if at > 0:
                candidates.append(points[at - 1])
            if at < len(points):
                candidates.append(points[at])
            candidates = [
                (pos, value) for pos, value in candidates
                if not _cut_between(cuts, min(idx, pos), max(idx, pos))
            ]
            if candidates:
                _pos, value = min(candidates,
                                  key=lambda item: abs(item[0] - idx))
                out[idx][tid] = value
    return out


def _overlap_min(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(aa, ba)
