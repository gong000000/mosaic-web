"""개인정보 보호 선택적 모자이크 웹 앱 (Streamlit + YOLO11-seg + VLM)
영상을 업로드하면 VLM이 개인정보 항목(얼굴, 차량번호판, 주소, 차량동승자 등)을
탐지하고, 사용자가 선택한 항목에만 모자이크를 적용한다."""
import hashlib
import os
import secrets
import shutil
import time
import tempfile
import cv2
import streamlit as st

from src.config import (CATEGORY_TYPE, CANDS, PRESETS, OUTPUT_DIR,
                        VLM_PROVIDERS, label_ko)
from src.preview import (analyze_video, render_preview_frame,
                         build_filmstrip, build_preview_video)
from src.vlm_package import prompt_categories
from src.render import normalize_video, trim_video
from src.models import get_model, MODEL_LOCK
from src import history as H
from src import retention as RT


# blur.me 스튜디오처럼 '항상 빠름' 고정. UI에서 속도/샘플을 고르지 않는다.
FAST_MODEL = "yolo11n-seg.pt"
FAST_IMGSZ = 480
FAST_STRIDE = 15
FAST_SAMPLES = 4
FAST_PRESET = "veryfast"


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

    /* 숨길 요소: 헤더는 사이드바 토글 때문에 남겨둔다. */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def load_model_cached(name):
    return get_model(name)


def _friendly_vlm_error(err, api_key=""):
    """VLM 오류를 사용자가 바로 조치할 수 있는 문장으로 바꾼다.
    키 문자열이 메시지에 섞여도 화면에 노출하지 않는다."""
    msg = str(err or "")
    if api_key:
        msg = msg.replace(str(api_key), "***")
    low = msg.lower()
    if ("401" in msg or "403" in msg or "invalid api key" in low
            or "unauthorized" in low or "permission" in low):
        return ("API 키가 올바르지 않거나 권한이 없습니다. "
                "제공자와 키를 확인하세요.")
    if ("429" in msg or "quota" in low or "rate limit" in low
            or "resource_exhausted" in low):
        compact = low.replace(" ", "").replace("_", "")
        if ("per day" in low or "daily" in low or "perday" in compact
                or "freetier" in compact):
            return ("VLM 무료 등급의 일일 호출 한도를 초과했습니다. "
                    "내일 다시 시도하거나 다른 모델/유료 키를 사용하세요.")
        return ("VLM 호출 한도를 초과했습니다. 잠시 후 다시 시도하거나 "
                "더 짧은 영상을 사용하세요.")
    if "404" in msg or "model not found" in low or "not found" in low:
        return "모델명을 찾을 수 없습니다. 사이드바의 모델명을 확인하세요."
    if "json" in low and "해석" in msg:
        return msg
    return msg


def _unique_out_path(src_name):
    """같은 초에 두 작업이 끝나도 결과 파일을 덮어쓰지 않게 한다."""
    stem = os.path.splitext(os.path.basename(src_name))[0]
    ts = time.strftime("%m%d_%H%M%S")
    token = secrets.token_hex(2)
    cand = os.path.join(OUTPUT_DIR, f"{stem}_{ts}_{token}.mp4")
    i = 1
    while os.path.exists(cand):
        cand = os.path.join(OUTPUT_DIR,
                            f"{stem}_{ts}_{token}_{i}.mp4")
        i += 1
    return cand


def _sweep_old_temp_dirs(max_hours=24):
    """24시간 지난 업로드 임시 폴더를 정리한다(서버 재시작과 무관하게)."""
    root = tempfile.gettempdir()
    now = time.time()
    removed = 0
    try:
        names = os.listdir(root)
    except Exception:
        return 0
    for name in names:
        if not name.startswith("mosaic_"):
            continue
        p = os.path.join(root, name)
        try:
            if os.path.isdir(p) and now - os.path.getmtime(p) > max_hours * 3600:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    return removed


if not st.session_state.get("tmp_swept"):
    _sweep_old_temp_dirs(RT.RETENTION_HOURS)
    st.session_state["tmp_swept"] = True


st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">🔒 개인정보 보호 모자이크</h1>
        <p class="header-subtitle">영상에서 개인정보(얼굴, 번호판, 간판, 명찰, 신분증 등)를 찾아 체크리스트로 보여주고, 선택한 항목만 모자이크하는 웹 앱</p>
    </div>
    """,
    unsafe_allow_html=True,
)
purged = RT.purge_old()
stale_rows = H.remove_missing()
if purged or stale_rows:
    st.caption(f"{RT.RETENTION_HOURS}시간 지난 결과 {purged}개 자동 삭제됨"
               + (f" · 만료된 기록 {stale_rows}개 정리" if stale_rows else ""))



def _apply_preset():
    """프리셋을 사이드바/본문 프롬프트에 동시에 반영한다."""
    txt = PRESETS.get(st.session_state.get("preset_sel", ""), "")
    st.session_state["prompt_txt"] = txt
    st.session_state["prompt_edit"] = txt


def _sync_prompt_from_sidebar():
    st.session_state["prompt_edit"] = st.session_state.get("prompt_txt", "")


def _sync_prompt_from_main():
    st.session_state["prompt_txt"] = st.session_state.get("prompt_edit", "")


if "prompt_txt" not in st.session_state:
    st.session_state["prompt_txt"] = PRESETS["얼굴 + 차량번호판"]
if "prompt_edit" not in st.session_state:
    st.session_state["prompt_edit"] = st.session_state["prompt_txt"]

with st.sidebar:
    st.header("설정")

    st.subheader("1. 영상 업로드")
    uploaded = st.file_uploader("mp4, mov, avi", type=["mp4", "mov", "avi"])

    st.subheader("2. 무엇을 가릴까 (프롬프트)")
    st.selectbox("프리셋", list(PRESETS.keys()), key="preset_sel",
                 on_change=_apply_preset)
    prompt_raw = st.text_area(
        "프롬프트", key="prompt_txt", height=90,
        on_change=_sync_prompt_from_sidebar,
        help="이 문장이 VLM 요청에 포함됩니다. 미리보기 화면에서도 수정할 수 있습니다.",
    )

    st.subheader("3. VLM")
    provider_label = st.selectbox(
        "VLM 제공자", list(VLM_PROVIDERS.keys()),
        help="OpenAI 또는 Google Gemini. Gemini는 OpenAI 호환 API로 연결합니다.")
    provider = VLM_PROVIDERS[provider_label]
    vlm_base_url = provider["base_url"]
    vlm_model = st.text_input(
        "모델명", value=provider["model"],
        key=f"vlm_model_{provider_label}",
        help="제공자의 비전 모델 이름.")
    api_key = st.text_input(
        "API 키", type="password", key=f"vlm_key_{provider_label}",
        help="VLM 판정에 쓰는 영상 프레임만 전송됩니다. 영상 파일 자체는 전송되지 않습니다.")

    st.divider()
    auto_del = st.checkbox("받으면 결과 자동 삭제", value=False)

if not uploaded:
    st.markdown(
        """
        <div class="info-card">
            <h3>시작하기</h3>
            <p>1) 왼쪽에서 영상을 올립니다. 2) 프롬프트와 VLM API 키를 입력합니다.
            3) 분석이 끝나면 체크리스트와 실시간 미리보기가 나타나고,
            별도 버튼 없이 자동으로 모자이크 영상이 만들어집니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# 세션마다 임시 폴더 1개만 쓴다. 업로드가 바뀔 때만 다시 기록한다.
tmp_dir = st.session_state.get("tmp_dir")
if not tmp_dir or not os.path.isdir(tmp_dir):
    tmp_dir = tempfile.mkdtemp(prefix="mosaic_")
    st.session_state["tmp_dir"] = tmp_dir

raw_bytes = uploaded.getvalue()
digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
upload_id = (uploaded.name, uploaded.size, digest)
raw_suffix = os.path.splitext(uploaded.name)[1].lower()
suffix = raw_suffix if raw_suffix in (".mp4", ".mov", ".avi") else ".mp4"
in_path = os.path.join(tmp_dir, "input" + suffix)
norm_path = os.path.join(tmp_dir, "input_norm.mp4")
trim_path = os.path.join(tmp_dir, "trim.mp4")


def _unlink_quiet(*paths):
    for p in paths:
        if not p:
            continue
        try:
            os.remove(p)
        except OSError:
            pass


if st.session_state.get("upload_id") != upload_id or not os.path.exists(in_path):
    old_in = st.session_state.get("input_path")
    _unlink_quiet(old_in if old_in != in_path else None,
                  norm_path, trim_path)
    with open(in_path, "wb") as f:
        f.write(raw_bytes)
    st.session_state["upload_id"] = upload_id
    st.session_state["input_path"] = in_path
    st.session_state.pop("trim_key", None)
    st.session_state.pop("preview_key", None)
    st.session_state.pop("preview_pos", None)
    st.session_state.pop("result", None)
    st.session_state.pop("quota_sig", None)


def _probe_video(path):
    """열기·첫 프레임·길이를 한 번에 확인한다. 실패하면 None."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return total, fps
    finally:
        cap.release()


@st.cache_data(show_spinner=False, max_entries=6)
def _read_frame_cached(path, idx, token):
    """미리보기용 프레임 1장. 업로드 내용(token)이 바뀌면 캐시도 갈린다."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def _filmstrip_frames_cached(path, token):
    """필름스트립용 대표 프레임. 시간 순으로 균등하게 뽑는다."""
    import numpy as np
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            return None
        want = 48
        idxs = []
        if total > 1:
            idxs = [int(round(i * (total - 1) / float(want - 1)))
                    for i in range(min(want, total))]
        else:
            idxs = [0]
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return np.array(frames) if frames else None
    finally:
        cap.release()


@st.cache_data(show_spinner=False, max_entries=6)
def _filmstrip_cached(path, token, progress):
    frames = _filmstrip_frames_cached(path, token)
    if frames is None or len(frames) == 0:
        return None
    import numpy as np
    frames = [np.asarray(f) for f in frames]
    return build_filmstrip(frames, progress=float(progress))


def _get_preview_video_path(proc_in, items, selected_cats, style, blocks,
                            blur, area_mode, cache_token):
    """미리보기 영상을 tmp_dir에 만들고 경로를 돌려준다.

    Streamlit 캐시에 파일 전체를 들고 있지 않으므로 새로고침에도
    파일 핸들을 유지하지 않는다. 설정/선택이 바뀌면 파일이 갈린다.
    """
    import hashlib, json
    sig = hashlib.sha1(json.dumps({
        "items": [(it.get("category"),
                   sorted((int(k), [list(map(int, b)) for b in v])
                            for k, v in (it.get("samples") or {}).items()))
                  for it in (items or [])],
        "sel": sorted(selected_cats or []),
        "style": style, "blocks": int(blocks), "blur": int(blur),
        "area": area_mode, "tok": cache_token,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    out_path = os.path.join(tmp_dir, f"preview_{sig}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    pv = build_preview_video(proc_in, items, selected_cats, style=style,
                              blocks=blocks, blur=blur, area_mode=area_mode,
                              max_frames=90, tmp_dir=tmp_dir)
    if pv and os.path.exists(pv):
        if pv != out_path:
            try:
                shutil.copy(pv, out_path)
            except OSError:
                out_path = pv
        return out_path
    return None


probe = _probe_video(in_path)
# MOV/AVI는 브라우저 미리보기와 OpenCV 읽기가 불안정하므로 MP4 사본을 만든다.
needs_norm = suffix != ".mp4" or probe is None
if needs_norm:
    if not os.path.exists(norm_path):
        with st.spinner("브라우저 호환 MP4로 변환 중..."):
            try:
                normalize_video(in_path, norm_path)
            except Exception as e:
                st.error(f"영상 변환 실패: {e}")
                st.stop()
    probe = _probe_video(norm_path)
    src_path = norm_path
else:
    src_path = in_path

if probe is None:
    st.error("영상을 읽을 수 없습니다. 손상되었거나 지원하지 않는 코덱입니다. "
             "MP4(H.264)로 변환해 다시 올려주세요.")
    st.stop()
total0, fps0 = probe
if not (fps0 > 0):
    fps0 = 30.0
dur = total0 / fps0
st.write(f"길이 {dur:.1f}초 · {total0}프레임")

use_trim = st.checkbox("구간만 처리", value=False)
trim_sig = None
if use_trim:
    t1, t2 = st.columns(2)
    with t1:
        s_sec = st.number_input("시작(초)", 0.0,
                                float(max(0.0, dur - 0.5)) if dur > 0.5 else 0.0,
                                0.0, step=0.5)
    with t2:
        e_sec = st.number_input("끝(초, 0=끝까지)", 0.0, float(dur), 0.0, step=0.5)
    if e_sec > 0 and e_sec <= s_sec:
        st.error("끝 시간은 시작 시간보다 커야 합니다.")
        st.stop()
    if s_sec <= 0 and e_sec <= 0:
        proc_in = src_path
    else:
        trim_sig = (upload_id, float(s_sec), float(e_sec))
        if (st.session_state.get("trim_key") != trim_sig
                or not os.path.exists(trim_path)):
            with st.spinner("구간 자르는 중..."):
                try:
                    trim_video(src_path, trim_path, s_sec,
                               e_sec if e_sec > 0 else None)
                except Exception as e:
                    st.error(f"구간 자르기 실패: {e}")
                    st.stop()
            st.session_state["trim_key"] = trim_sig
            st.session_state.pop("preview_key", None)
            st.session_state.pop("result", None)
            st.session_state["preview_pos"] = 0
        proc_in = trim_path
else:
    s_sec, e_sec = 0.0, 0.0
    proc_in = src_path

proc_probe = _probe_video(proc_in)
proc_total = int(proc_probe[0]) if proc_probe else total0

st.subheader("1. 개인정보 항목 확인")

analysis_targets = list(CANDS)
analysis_key = None
ok = False
preview_err = ""
anno, persons, base_items = None, [], []

if not api_key.strip():
    st.warning("VLM API 키를 입력해야 영상을 분석할 수 있습니다.")
else:
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    analysis_key = (upload_id, tuple(analysis_targets), key_hash,
                    vlm_base_url, vlm_model, provider_label,
                    FAST_MODEL, FAST_IMGSZ, FAST_SAMPLES, trim_sig)

    if st.button("다시 분석", key="reanalyze"):
        st.session_state.pop("preview_key", None)
        st.session_state.pop("result", None)
        st.session_state.pop("quota_sig", None)
        st.rerun()

    st.caption(f"VLM: {provider_label} · {vlm_model or '(모델명 없음)'} · "
               "영상이 바뀌면 자동으로 다시 분석합니다. "
               "프롬프트만 바꾸면 VLM을 다시 호출하지 않고 체크 항목만 갱신합니다.")
    if st.session_state.get("preview_key") != analysis_key:
        st.session_state.pop("result", None)
        st.session_state.pop("quota_sig", None)
        err = None
        anno, persons, found = None, [], []
        with st.spinner("영상 샘플 분석 중... (VLM)"):
            try:
                det = load_model_cached(FAST_MODEL)
                # analyze_video 내부에서 YOLO predict만 MODEL_LOCK으로
                # 직렬화한다. 바깥에서 락을 잡으면 병렬 워커가 락을
                # 기다리다 교착되므로 여기서는 잡지 않는다.
                anno, persons, found = analyze_video(
                    proc_in, det, conf=0.20, imgsz=FAST_IMGSZ,
                    query_raw=prompt_raw, targets=analysis_targets,
                    api_key=api_key,
                    vlm_model=vlm_model,
                    vlm_base_url=vlm_base_url,
                    samples=FAST_SAMPLES,
                )
            except Exception as e:
                err = e
        st.session_state["preview_key"] = analysis_key
        st.session_state["preview_err"] = (
            _friendly_vlm_error(err, api_key) if err else "")
        st.session_state["preview_ok"] = anno is not None and err is None
        st.session_state["preview_anno"] = anno
        st.session_state["preview_persons"] = persons
        st.session_state["preview_items"] = found
        st.session_state["analysis_n"] = st.session_state.get("analysis_n", 0) + 1

    ok = st.session_state.get("preview_ok", False)
    preview_err = st.session_state.get("preview_err", "")
    anno = st.session_state.get("preview_anno")
    persons = st.session_state.get("preview_persons", [])
    base_items = st.session_state.get("preview_items", [])

if preview_err:
    st.error(f"분석 실패: {preview_err}")
    # 한도/인증 오류는 자동 재시도로 호출을 더 태우지 않는다.
    st.session_state["quota_sig"] = analysis_key
    st.session_state["quota_msg"] = f"분석 실패: {preview_err}"
elif ok and persons:
    st.write(f"사람 {len(persons)}명 · 개인정보 후보 {len(base_items)}종 찾음")
    if anno is not None:
        st.image(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB),
                 caption="첫 샘플 화면 (초록 번호=개인정보 탐지 후보)")
elif ok:
    st.write(f"사람 없음 · 개인정보 후보 {len(base_items)}종 찾음")

analysis_id = str(st.session_state.get("analysis_n", 0))

# 프롬프트에 적힌 카테고리를 기준으로 체크박스 기본값을 정한다.
# '얼굴과 번호판' → '얼굴만'으로 바꾸면 번호판 키가 새로 만들어져
# 자동으로 체크 해제된다(VLM 재호출 없음).
prompt_cats = prompt_categories(prompt_raw)
items = []
seen_cats = set()
if ok:
    for it in base_items:
        copy = dict(it)
        copy["samples"] = dict(it.get("samples") or {})
        copy["prompt_match"] = copy.get("category") in prompt_cats
        items.append(copy)
        seen_cats.add(copy.get("category"))
    # 프롬프트가 요구했지만 샘플에서 못 찾은 항목도 체크리스트에 남긴다.
    # 사람 항목(얼굴 등)은 처리 단계에서 모든 사람에게 적용할 수 있다.
    for cat in sorted(prompt_cats - seen_cats):
        items.append({
            "category": cat,
            "label_ko": label_ko(cat),
            "type": CATEGORY_TYPE.get(cat, "region"),
            "prompt_match": True,
            "boxes": [],
            "samples": {},
            "source": "prompt",
            "count": 0,
            "box_count": 0,
            "thumb": None,
            "missing": True,
        })

if items:
    prompt_items = [it for it in items if it.get("category") in prompt_cats]
    extra_items = [it for it in items if it.get("category") not in prompt_cats]

    filter_id = hashlib.sha1(
        ("|".join(sorted(prompt_cats)) or "all").encode("utf-8")
    ).hexdigest()[:8]

    def _default_checked(cat):
        return (cat in prompt_cats) if prompt_cats else True

    def _check_key(cat):
        return f"chk_{analysis_id}_{filter_id}_{cat}"

    def _render_item(it):
        cat = it["category"]
        unit = "명" if it.get("type") == "person" else "곳"
        cnt = it.get("count", 0)
        box_cnt = it.get("box_count", 0)
        label = f"{it.get('label_ko', cat)} · {cnt}{unit}"
        if it.get("missing"):
            label += " · 이번 샘플에서 못 찾음"
        elif box_cnt and box_cnt != cnt:
            label += f" (총 {box_cnt}회 탐지)"
        c1, c2 = st.columns([1, 9])
        with c1:
            th = it.get("thumb")
            if th is not None:
                try:
                    st.image(cv2.cvtColor(th, cv2.COLOR_BGR2RGB), width=84)
                except Exception:
                    pass
        with c2:
            checked = st.checkbox(label, value=_default_checked(cat),
                                  key=_check_key(cat))
        return checked

    if prompt_items:
        st.markdown("**요청한 항목**")
        for it in prompt_items:
            _render_item(it)
    if extra_items:
        st.markdown("**추가로 찾은 개인정보**")
        for it in extra_items:
            _render_item(it)
    st.caption("체크한 항목만 가립니다. 프롬프트를 바꾸면 체크 상태가 "
               "프롬프트에 맞게 다시 정해집니다.")

    found_now = {it["category"] for it in items if not it.get("missing")}
    missing = sorted(prompt_cats - found_now)
    if missing:
        st.caption("샘플에서 못 찾은 요청 항목: "
                   + ", ".join(label_ko(c) for c in missing)
                   + " (처리 단계에서 한 번 더 찾습니다)")

    selected_cats = [it["category"] for it in items
                     if st.session_state.get(_check_key(it["category"]),
                                            _default_checked(it["category"]))]
else:
    selected_cats = []
    if ok:
        st.warning("VLM이 이번 샘플에서 가릴 항목을 찾지 못했습니다. "
                   "프롬프트를 구체화하거나 '다시 분석'을 눌러보세요.")

# ---------------------------------------------------------------- 설정/미리보기
st.subheader("2. 모자이크 설정")

area_label = st.segmented_control(
    "가리는 범위",
    ["👤 얼굴만", "🧍 상반신", "🧑 전체"],
    default="👤 얼굴만",
    key="area_mode_ui",
    help="사람 항목에 적용됩니다.")
area_label = area_label or "👤 얼굴만"
area_mode = {"👤 얼굴만": "face_only",
             "🧍 상반신": "upper",
             "🧑 전체": "full"}[area_label]

style_label = st.segmented_control(
    "모자이크 모양",
    ["🟦 모자이크", "🌫️ 흐리게", "⬛ 검게"],
    default="🟦 모자이크",
    key="style_ui")
style_label = style_label or "🟦 모자이크"
style = {"🟦 모자이크": "pixel",
         "🌫️ 흐리게": "blur",
         "⬛ 검게": "solid"}[style_label]
if style == "pixel":
    blocks = st.slider("강도", 18, 44, 30,
                       help="값이 클수록 더 강하게 가립니다.")
    blur = 61
elif style == "blur":
    blocks = 30
    blur = st.slider("흐림 정도", 21, 101, 61, step=2)
else:
    blocks, blur = 30, 61

st.caption("번호판·간판 같은 배경 항목은 '가리는 범위'와 무관하게 "
           "탐지된 영역만 가립니다. 명찰도 찾은 위치만 가립니다.")

st.text_area(
    "프롬프트 수정 (미리보기에서 바로 반영)",
    key="prompt_edit", height=80,
    on_change=_sync_prompt_from_main,
    help="예: '얼굴과 차량번호판을 모자이크' → '얼굴만 모자이크'로 바꾸면 "
         "차량번호판 체크가 자동으로 해제됩니다.")

st.subheader("3. 실시간 미리보기")
selected_tuple = tuple(sorted(selected_cats))

if proc_total > 0:
    frame_idx = st.slider(
        "타임라인", 0, max(0, proc_total - 1), 0,
        key="preview_pos",
        help="타임라인을 움직이면 그 위치의 프레임을 모자이크 적용해 보여줍니다.")
    compare_original = st.toggle("원본과 비교", value=False)

    frame_time = float(frame_idx) / float(fps0) if fps0 > 0 else 0.0

    # 원본 영상 재생 (원본 길이·속도 그대로, 배속 없음)
    st.video(proc_in, format="video/mp4", start_time=frame_time,
             width="stretch")
    st.caption(f"{frame_time:.1f}s / {dur:.1f}s")

    # 선택한 항목에 모자이크를 적용한 단일 프레임 (타임라인 위치)
    src_frame = _read_frame_cached(proc_in, int(frame_idx),
                                   f"{upload_id[2]}|{trim_sig}")
    if src_frame is None:
        st.warning("이 위치의 프레임을 읽지 못했습니다.")
    else:
        shown = src_frame.copy()
        if selected_cats and items:
            try:
                shown = render_preview_frame(
                    src_frame, items, selected_cats, style=style,
                    blocks=blocks, blur=blur, area_mode=area_mode,
                    frame_idx=int(frame_idx))
            except Exception as e:
                st.caption(f"미리보기 렌더 오류: {e}")

        if compare_original:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("원본")
                st.image(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB),
                         width="stretch")
            with c2:
                st.caption("모자이크 미리보기")
                st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                         width="stretch")
        else:
            st.caption("모자이크 미리보기 (타임라인 위치)")
            st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                     width="stretch")

    # 필름스트립 (원본 썸네일, 어디에 어떤 장면이 있는지 한눈에)
    filmstrip = None
    try:
        filmstrip = _filmstrip_cached(
            proc_in, f"{upload_id[2]}|{trim_sig}",
            float(frame_idx) / float(max(1, proc_total - 1)))
    except Exception as e:
        st.caption(f"타임라인 미리보기 생성 실패: {e}")
    if filmstrip is not None:
        st.image(filmstrip, width="stretch")
else:
    st.info("미리보기를 만들 프레임이 없습니다.")

# ---------------------------------------------------------------- 모자이크/다운로드
st.subheader("4. 모자이크 확인 · 다운로드")

if not selected_cats and ok:
    st.caption("가릴 항목을 하나 이상 체크하면 미리보기 영상에 모자이크가 적용됩니다.")

pv_out = None
if ok and selected_cats and proc_total > 0:
    try:
        pv_out = _get_preview_video_path(
            proc_in=proc_in, items=items, selected_cats=selected_cats,
            style=style, blocks=blocks, blur=blur, area_mode=area_mode,
            cache_token=f"{upload_id[2]}|{trim_sig}")
    except Exception as e:
        st.error(f"미리보기 영상 생성 실패: {e}")
        pv_out = None

if pv_out and os.path.exists(pv_out):
    with open(pv_out, "rb") as f:
        clicked = st.download_button(
            "🔒 미리보기 모자이크 영상 받기", f,
            file_name=os.path.splitext(uploaded.name)[0] + "_preview.mp4",
            mime="video/mp4")
    if clicked and auto_del:
        RT.remove_on_download(pv_out)
        st.info("받기 끝나서 미리보기 영상을 지웠다. (보관 정책)")
elif ok and selected_cats and proc_total > 0:
    st.info("미리보기 영상을 만들지 못했습니다.")

with st.expander("지난 작업", expanded=False):
    for row in H.all_rows():
        fname = row.get("file")
        if not fname:
            continue
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"{row.get('at', '')} · {row.get('src', '')} → {fname}")
            extra = row.get("summary") or ""
            if extra:
                st.caption(extra)
        with c2:
            p = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    st.download_button("받기", f, file_name=fname,
                                       key=f"dl_{fname}")
        with c3:
            if st.button("삭제", key=f"del_{fname}"):
                H.delete(fname)
                st.rerun()

