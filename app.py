"""개인정보 보호 선택적 모자이크 웹 앱 (Streamlit + YOLO11-seg + VLM)
영상을 업로드하면 VLM이 개인정보 항목(얼굴, 차량번호판, 주소, 차량동승자 등)을
탐지하고, 사용자가 선택한 항목에만 모자이크를 적용한다."""
import os
import time
import tempfile
import cv2
import streamlit as st

from src.config import CANDS, PRESETS, OUTPUT_DIR, KO
from src.prompts import warn_abstract, GUIDE
from src.preview import analyze_first_frame
from src.render import trim_video, cleanup_tmp
from src.models import get_model
from src.pipeline import mosaic_video
from src import history as H
from src import retention as RT


st.set_page_config(
    page_title="개인정보 보호 모자이크",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #8b5cf6;
        --primary-hover: #7c3aed;
        --accent: #06b6d4;
        --bg: #0b0f19;
        --card: #131a2b;
        --text: #f1f5f9;
        --muted: #94a3b8;
        --border: #334155;
        --success: #22c55e;
        --warning: #f59e0b;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .stApp {
        background: #0a0a0f;
        background-image:
          linear-gradient(rgba(18, 16, 16, 0.5) 50%, transparent 50%),
          linear-gradient(90deg, rgba(18, 16, 16, 0.5) 50%, transparent 50%);
        background-size: 20px 20px;
        background-attachment: fixed;
    }

    .main {
        background: transparent;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* 헤더 카드 */
    .header-card {
        background: rgba(15, 15, 25, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(99, 102, 241, 0.2);
        position: relative;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 0.5rem 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .header-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* 사이드바 */
    .stSidebar {
        background: var(--card) !important;
        border-right: 1px solid var(--border);
    }

    .stSidebar .stMarkdown, .stSidebar .stCaption {
        color: var(--muted) !important;
    }

    .stSidebar h2, .stSidebar h3 {
        color: var(--text) !important;
        font-size: 0.85rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .stSidebar .stSelectbox, .stSidebar .stMultiselect, .stSidebar .stSlider, .stSidebar .stRadio {
        margin-bottom: 1rem;
    }

    .stSidebar .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    .stSidebar .stButton > button:hover {
        background: var(--primary-hover);
    }

    .stSidebar .stTextInput > div > div > input {
        background: #0f172a;
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 8px;
    }

    .stSidebar .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    .stSidebar .stCheckbox > div {
        color: var(--text);
    }

    /* 버튼 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-hover) 0%, #0891b2 100%);
        box-shadow: 0 8px 10px -6px rgba(99, 102, 241, 0.4);
    }

    .stButton > button[kind="primary"]:disabled {
        background: var(--border);
        color: var(--muted);
        box-shadow: none;
    }

    /* 카드 */
    .info-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .info-card:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }

    .info-card h3 {
        color: var(--text);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }

    .info-card p {
        color: var(--muted);
        margin: 0;
        font-size: 0.9rem;
    }

    /* 타겟 선택 */
    .stMultiSelect > div > div {
        background: #0f172a;
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .stMultiSelect > div > div > div {
        color: var(--text);
    }

    /* 파일 업로더 */
    .stFileUploader {
        background: var(--card);
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .stFileUploader:hover {
        border-color: var(--primary);
        background: #131a2b;
    }

    .stFileUploader > div {
        color: var(--text);
    }

    /* 진행률 */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
    }

    /* 비디오 */
    .stVideo > div {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: #000;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    /* 구분선 */
    .stDivider {
        margin: 1.5rem 0;
        border-color: var(--border);
    }

    /* expander */
    .streamlit-expanderHeader {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
    }

    .streamlit-expanderContent {
        background: #0f172a;
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }

    /* 숨길 요소 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def load_model_cached(name):
    return get_model(name)


st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">🔒 개인정보 보호 모자이크</h1>
        <p class="header-subtitle">영상에서 개인정보(얼굴, 차량번호판, 주소, 차량동승자)를 자동 탐지하고 모자이크하는 웹 앱 · 이 PC에서만 처리</p>
    </div>
    """,
    unsafe_allow_html=True,
)
purged = RT.purge_old()
if purged:
    st.caption(f"24시간 지난 결과 {purged}개 자동 삭제됨")


with st.sidebar:
    st.header("설정")

    st.subheader("1. 영상 업로드")
    uploaded = st.file_uploader("mp4, mov, avi", type=["mp4", "mov", "avi"])

    st.subheader("2. 개인정보 탐지 방식")
    use_vlm = st.checkbox("VLM 사용 (OpenAI API 키 필요)", value=True,
                          help="OpenAI API 키를 입력하면 VLM이 영상을 보고 개인정보를 더 정확히 탐지합니다.")
    api_key = ""
    if use_vlm:
        api_key = st.text_input("OpenAI API 키", type="password", key="vlm_key",
                                help="영상은 절대 외부로 전송되지 않으며, 첫 프레임만 전송됩니다.")

    st.subheader("3. 가릴 항목 선택")
    st.caption("VLM이 탐지한 항목 중에서 가릴 대상을 선택하세요.")
    targets = st.multiselect(
        "가릴 항목 (여러 개 가능)",
        CANDS,
        default=["face", "license_plate"],
        format_func=lambda x: KO.get(x, x),
    )
    if not targets:
        targets = ["face"]

    st.subheader("4. 속도 모드")
    speed_label = st.radio("속도", ["빠름", "보통", "정밀"], index=1,
                           help="빠름: 키프레임 1개만 탐지 / 보통: 간격 탐지 / 정밀: 모든 프레임 탐지")
    speed_map = {
        "빠름": ("yolo11n-seg.pt", 480, "veryfast", 1),
        "보통": ("yolo11m-seg.pt", 640, "veryfast", 5),
        "정밀": ("yolo11x-seg.pt", 960, "slow", 1),
    }
    model_name, imgsz, encode_preset, keyframe_stride = speed_map[speed_label]

    st.subheader("5. 모자이크 설정")
    area_label = st.selectbox("가리는 범위", ["전체", "상반신", "얼굴만"], index=0)
    area_mode = {"전체": "full", "상반신": "upper", "얼굴만": "face_only"}[area_label]
    style_label = st.selectbox("모자이크 모양", ["모자이크", "흐리게", "검게"], index=0)
    style = {"모자이크": "pixel", "흐리게": "blur", "검게": "solid"}[style_label]
    if style == "모자이크":
        blocks = st.slider("강도", 18, 44, 30)
        blur = 61
    elif style == "흐리게":
        blocks = 30
        blur = st.slider("흐림 정도", 21, 101, 61, step=2)
    else:
        blocks, blur = 30, 61

    st.divider()
    auto_del = st.checkbox("받으면 결과 자동 삭제", value=False)

if not uploaded:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
    tmp_in.write(uploaded.getvalue())
    in_path = tmp_in.name

cap0 = cv2.VideoCapture(in_path)
fps0 = int(cap0.get(cv2.CAP_PROP_FPS)) or 30
total0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
dur = total0 / max(1, fps0)
cap0.release()
st.write(f"길이 {dur:.1f}초 · {total0}프레임")

use_trim = st.checkbox("구간만 처리", value=False)
if use_trim:
    t1, t2 = st.columns(2)
    with t1:
        s_sec = st.number_input("시작(초)", 0.0, float(max(0, dur - 0.5)) if dur > 0.5 else 0.0, 0.0, step=0.5)
    with t2:
        e_sec = st.number_input("끝(초, 0=끝까지)", 0.0, float(dur), 0.0, step=0.5)
    trim_path = in_path + "_trim.mp4"
    with st.spinner("구간 자르는 중..."):
        trim_video(in_path, trim_path, s_sec, e_sec if e_sec > 0 else None)
    proc_in = trim_path
else:
    proc_in = in_path

with st.spinner("준비 중..."):
    load_model_cached(model_name)
    cap = cv2.VideoCapture(proc_in)
    ok, first = cap.read()
    cap.release()

st.subheader("2. 개인정보 항목 확인")
if ok:
    det = load_model_cached(model_name)
    anno, persons = analyze_first_frame(
        first, det, conf=0.20, imgsz=imgsz,
        query_raw="", targets=targets,
        api_key=api_key if use_vlm else None,
        vlm_model="gpt-4o-mini",
        use_vlm=use_vlm,
    )
    if persons:
        st.write(f"사람 {len(persons)}명 찾음. 아래에서 개인정보 항목을 선택하세요.")
        st.image(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB),
                 caption="첫 화면 전체 (초록 번호=개인정보 탐지 후보)")
        for p in persons:
            bi, crop, is_t, top1, hits = p["bi"], p["crop"], p["is_t"], p["top1"], p["hits"]
            badge = "가려짐" if is_t else "유지"
            hits_str = ", ".join([KO.get(h, h) for h in hits]) if hits else "탐지 안 됨"
            c1, c2 = st.columns([1, 2])
            with c1:
                try:
                    disp = cv2.resize(crop, (200, max(1, int(200 * crop.shape[0] / max(1, crop.shape[1])))),
                                      interpolation=cv2.INTER_CUBIC)
                except Exception:
                    disp = crop
                st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), width=200)
            with c2:
                st.markdown(f"**사람 {bi+1} · {badge}**")
                st.caption(f"탐지 항목: {hits_str}")
    else:
        st.warning("사람을 찾지 못했다. 다른 영상으로 해봐라.")
else:
    st.warning("영상을 읽을 수 없습니다.")

st.subheader("3. 처리")
run = st.button("모자이크 시작", type="primary")
if not run:
    st.stop()

ts = time.strftime("%m%d_%H%M%S")
out_name = f"{os.path.splitext(uploaded.name)[0]}_{ts}.mp4"
out_path = os.path.join(OUTPUT_DIR, out_name)
bar = st.progress(0)
info = st.empty()


def _pg(p, d, t, el):
    bar.progress(p)
    if el > 1:
        info.write(f"{d}/{t} 프레임 · {el:.1f}초")


t_start = time.time()
stats = {}
with st.spinner("처리 중..."):
    stats = mosaic_video(
        proc_in, out_path,
        targets=targets, invert=False, conf=0.20,
        model_name=model_name, imgsz=imgsz,
        blocks=blocks, style=style, blur=blur,
        ultra=(speed_label == "정밀"), area_mode=area_mode,
        keyframe_stride=keyframe_stride,
        encode_preset=encode_preset, progress_fn=_pg,
        query_raw="", api_key=api_key if use_vlm else None,
        vlm_model="gpt-4o-mini",
        use_vlm=use_vlm,
    ) or {}
elapsed = time.time() - t_start

st.success(f"완료 · {elapsed:.1f}초 걸림")
final_n = sum(1 for v in stats.values() if v.get("mosaic"))
st.write(f"최종 {final_n}명 가려짐")
for tid, v in sorted(stats.items()):
    c1, c2 = st.columns([1, 3])
    with c1:
        th = v.get("thumb")
        if th is not None:
            try:
                st.image(cv2.cvtColor(th, cv2.COLOR_BGR2RGB), width=96)
            except Exception:
                pass
    with c2:
        st.write(f"추적 {tid} · {'가려짐' if v.get('mosaic') else '유지'}")
H.add(uploaded.name, "", targets[0] if targets else "", False, out_name, elapsed,
      query_raw="",
      selector=("vlm" if use_vlm else "world"),
      summary=f"{final_n}명 가려짐")
st.markdown(
    """
    <div class="info-card">
        <h3 style="margin-top:0;">📊 결과 비교</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**📹 원본**")
    st.video(proc_in, format="video/mp4")
with c2:
    st.markdown("**🔒 모자이크 적용**")
    st.video(out_path, format="video/mp4")
clicked = False
with open(out_path, "rb") as f:
    clicked = st.download_button("결과 저장", f, file_name=out_name, mime="video/mp4")
if clicked and auto_del:
    RT.remove_on_download(out_path)
    st.info("받기 끝나서 결과 파일을 지웠다. (보관 정책)")

with st.expander("지난 작업", expanded=False):
    for r in H.all_rows():
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"{r['at']} · {r['src']} → {r['file']}")
        with c2:
            p = os.path.join(OUTPUT_DIR, r["file"])
            if os.path.exists(p):
                with open(p, "rb") as f:
                    st.download_button("받기", f, file_name=r["file"], key=f"dl_{r['file']}")
        with c3:
            if st.button("삭제", key=f"del_{r['file']}"):
                H.delete(r["file"])
                st.rerun()

cleanup_tmp(in_path, in_path + "_trim.mp4", out_path + ".noaudio.mp4")
