"""수동 영역 편집 화면(Streamlit).

자동 검출이 놓친 곳을 '추가'하거나 잘못 가린 곳을 '제외'한다. 캔버스 위젯
없이 슬라이더로 박스를 지정하고, 선택한 시각의 프레임 위에 실시간으로
윤곽을 그려 준다. 움직이는 대상은 '끝 박스'를 켜서 시작~끝 사이를 보간한다.
"""
from __future__ import annotations

import cv2
import streamlit as st

from .manual import active_boxes, normalize_regions

_ADD_COLOR = (60, 200, 60)      # BGR 초록
_EXC_COLOR = (50, 50, 230)      # BGR 빨강
_PREVIEW_COLOR = (0, 200, 255)  # BGR 노랑
_KEY = "manual_regions"


def _read_frame(path, t, fps):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _box_from_pct(v):
    return [v[0] / 100.0, v[1] / 100.0, v[2] / 100.0, v[3] / 100.0]


def _draw(frame, regions, t, preview_boxes=None):
    h, w = frame.shape[:2]
    img = frame.copy()
    for mode, color in (("add", _ADD_COLOR), ("exclude", _EXC_COLOR)):
        for (x1, y1, x2, y2), shape in active_boxes(regions, t, w, h, mode):
            if shape == "ellipse":
                cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2),
                            ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360,
                            color, 3)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    for b in preview_boxes or []:
        x1, y1 = int(b[0] * w), int(b[1] * h)
        x2, y2 = int(b[2] * w), int(b[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), _PREVIEW_COLOR, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_regions():
    return normalize_regions(st.session_state.get(_KEY))


def reset_if_changed(identity):
    """영상/구간이 바뀌면 이전 영상의 수동 영역을 버린다."""
    if st.session_state.get("_manual_identity") != identity:
        st.session_state["_manual_identity"] = identity
        st.session_state[_KEY] = []


def render_editor(video_path, duration, fps, face_gaps=None):
    """편집 화면을 그리고 현재 수동 영역 목록을 돌려준다."""
    regions = st.session_state.setdefault(_KEY, [])
    n = len(regions)
    with st.expander(f"✋ 수동 영역 편집 ({n}개)" if n else
                     "✋ 수동 영역 편집 (놓친 곳 추가 / 잘못 가린 곳 제외)",
                     expanded=False):
        st.caption(
            "자동 검출이 놓친 얼굴·번호판은 '추가', 가리지 않아도 되는 곳은 "
            "'제외'로 지정하세요. 움직이는 대상은 끝 박스를 켜면 시작~끝 사이를 "
            "따라갑니다. 노란 선이 지금 편집 중인 박스입니다.")
        if face_gaps:
            gap_txt = ", ".join(f"{s:.1f}~{e:.1f}초" for s, e in face_gaps[:8])
            st.info(f"검출 보간/추적 유지가 작동한 구간: {gap_txt}. "
                    "해당 시각을 확인하고 필요하면 수동 영역을 추가하세요.")
        fps = fps if fps and fps > 0 else 30.0
        max_t = max(0.1, float(duration))
        t = st.slider("시각(초)", 0.0, max_t, 0.0,
                      step=max(0.01, round(1.0 / fps, 3)), key="mn_t")
        c1, c2 = st.columns(2)
        with c1:
            mode_label = st.radio("동작", ["추가(가림)", "제외(원본 유지)"],
                                  horizontal=True, key="mn_mode")
            shape_label = st.radio("모양", ["사각형", "타원"],
                                   horizontal=True, key="mn_shape")
        with c2:
            moving = st.checkbox("움직이는 대상 (끝 박스 사용)", key="mn_moving")
            step = max(0.01, round(1.0 / fps, 3))
            if moving and t < max_t - 2 * step:
                t_end = st.slider("끝 시각(초)", float(t), max_t,
                                  min(max_t, float(t) + 1.0), step=step)
            else:
                if moving:
                    st.caption("영상 끝에 너무 가까워 끝 박스를 쓸 수 없습니다.")
                    moving = False
                hold = st.slider("유지 시간(초)", 0.0, 10.0, 1.0, step=0.1,
                                 key="mn_hold")
                t_end = min(max_t, float(t) + float(hold))

        st.markdown("**시작 박스** (프레임 대비 %)")
        s1, s2 = st.columns(2)
        with s1:
            sx = st.slider("가로 위치(%)", 0, 100, (40, 60), key="mn_sx")
        with s2:
            sy = st.slider("세로 위치(%)", 0, 100, (30, 50), key="mn_sy")
        box0 = _box_from_pct([sx[0], sy[0], sx[1], sy[1]])
        box1 = box0
        if moving:
            st.markdown("**끝 박스** (끝 시각의 위치)")
            e1, e2 = st.columns(2)
            with e1:
                ex = st.slider("가로 위치(%)", 0, 100, (50, 70), key="mn_ex")
            with e2:
                ey = st.slider("세로 위치(%)", 0, 100, (30, 50), key="mn_ey")
            box1 = _box_from_pct([ex[0], ey[0], ex[1], ey[1]])

        frame = _read_frame(video_path, t, fps)
        if frame is not None:
            st.image(_draw(frame, normalize_regions(regions), t, [box0]),
                     caption=f"{t:.2f}초 프레임 · 초록=추가, 빨강=제외, "
                             "노랑=편집 중", width="stretch")
            if moving:
                f2 = _read_frame(video_path, t_end, fps)
                if f2 is not None:
                    st.image(_draw(f2, normalize_regions(regions), t_end,
                                   [box1]),
                             caption=f"{t_end:.2f}초 프레임(끝 박스)",
                             width="stretch")
        else:
            st.warning("이 시각의 프레임을 읽지 못했습니다.")

        if st.button("이 영역 추가", key="mn_add", type="primary"):
            mode = "add" if mode_label.startswith("추가") else "exclude"
            regions.append({
                "mode": mode, "t0": float(t), "t1": float(t_end),
                "box0": box0, "box1": box1,
                "shape": "ellipse" if shape_label == "타원" else "rect",
                "label": f"{'추가' if mode == 'add' else '제외'} "
                         f"{t:.1f}~{t_end:.1f}초"})
            st.session_state[_KEY] = regions
            st.rerun()

        if regions:
            st.markdown("**지정한 영역**")
            for i, r in enumerate(list(regions)):
                a, b = st.columns([5, 1])
                with a:
                    icon = "🟢" if r["mode"] == "add" else "🔴"
                    st.write(f"{icon} {r.get('label') or r['mode']}")
                with b:
                    if st.button("삭제", key=f"mn_del_{i}"):
                        regions.pop(i)
                        st.session_state[_KEY] = regions
                        st.rerun()
            if any(r["mode"] == "exclude" for r in regions):
                st.warning("'제외' 영역은 모자이크를 풀어 원본을 보여 줍니다. "
                           "개인정보가 없는 곳만 지정하세요.")
    return normalize_regions(regions)
