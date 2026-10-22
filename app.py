"""개인정보 보호 선택적 모자이크 웹 앱 (Streamlit + YOLO11-seg + VLM)
영상을 업로드하면 VLM이 개인정보 항목(얼굴, 차량번호판, 주소, 차량동승자 등)을
탐지하고, 사용자가 선택한 항목에만 모자이크를 적용한다."""
import os
import shutil
import time
import tempfile
import cv2
import streamlit as st

from src.config import CANDS, PRESETS, OUTPUT_DIR, VLM_MODEL, label_ko
from src.preview import analyze_video
from src.vlm_package import prompt_categories
from src.render import trim_video
from src.models import get_model, MODEL_LOCK
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


def _unique_out_path(src_name):
    """같은 초에 두 작업이 끝나도 결과 파일을 덮어쓰지 않게 한다."""
    stem = os.path.splitext(os.path.basename(src_name))[0]
    ts = time.strftime("%m%d_%H%M%S")
    cand = os.path.join(OUTPUT_DIR, f"{stem}_{ts}.mp4")
    i = 1
    while os.path.exists(cand):
        cand = os.path.join(OUTPUT_DIR, f"{stem}_{ts}_{i}.mp4")
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
        <p class="header-subtitle">영상에서 프롬프트가 지정한 항목과 추가 개인정보(얼굴, 번호판, 간판, 명찰, 신분증 등)를 찾아 체크리스트로 보여주고, 선택한 항목만 모자이크하는 웹 앱</p>
    </div>
    """,
    unsafe_allow_html=True,
)
purged = RT.purge_old()
if purged:
    st.caption(f"{RT.RETENTION_HOURS}시간 지난 결과 {purged}개 자동 삭제됨")


def _apply_preset():
    """프리셋 선택 시 프롬프트를 위젯 생성 전에 채운다."""
    st.session_state["prompt_txt"] = PRESETS.get(
        st.session_state.get("preset_sel", ""), "")


if "prompt_txt" not in st.session_state:
    st.session_state["prompt_txt"] = PRESETS["얼굴 + 차량번호판"]

with st.sidebar:
    st.header("설정")

    st.subheader("1. 영상 업로드")
    uploaded = st.file_uploader("mp4, mov, avi", type=["mp4", "mov", "avi"])

    st.subheader("2. 무엇을 가릴까 (프롬프트)")
    st.selectbox("프리셋", list(PRESETS.keys()), key="preset_sel",
                 on_change=_apply_preset)
    prompt_raw = st.text_area(
        "프롬프트", key="prompt_txt", height=90,
        help="이 문장이 VLM에 그대로 전달됩니다. VLM은 영상 프레임을 보고 판단합니다.",
    )

    st.subheader("3. 가릴 항목 선택")
    st.caption("영상을 분석하면 아래에 프롬프트 항목과 추가 개인정보 후보가 나옵니다. "
               "둘 다 기본으로 체크되며, 가리지 않을 항목은 해제하세요.")

    st.subheader("4. VLM (OpenAI)")
    api_key = st.text_input("OpenAI API 키", type="password", key="vlm_key",
                            help="입력하면 VLM 판정에 쓰는 영상 프레임(키프레임)이 "
                                 "OpenAI로 전송됩니다. 영상 파일 자체는 전송되지 않습니다. "
                                 "비워두면 로컬 YOLO-World로 대체합니다.")

    st.subheader("5. 속도 모드")
    speed_label = st.radio("속도", ["빠름", "보통", "정밀"], index=1,
                           help="빠름: 작은 모델·15프레임 간격 / 보통: 중간 모델·5프레임 간격 / "
                                "정밀: 큰 모델·모든 프레임. 간격마다 VLM이 판정한다.")
    speed_map = {
        "빠름": ("yolo11n-seg.pt", 480, "veryfast", 15),
        "보통": ("yolo11m-seg.pt", 640, "veryfast", 5),
        "정밀": ("yolo11x-seg.pt", 960, "slow", 1),
    }
    model_name, imgsz, encode_preset, keyframe_stride = speed_map[speed_label]

    st.subheader("6. 모자이크 설정")
    area_label = st.selectbox("가리는 범위", ["전체", "상반신", "얼굴만"], index=0)
    area_mode = {"전체": "full", "상반신": "upper", "얼굴만": "face_only"}[area_label]
    st.caption("사람 항목에 적용됩니다. '얼굴만/상반신'을 고르면 명찰·문신 등 다른 부위는 "
               "가려지지 않을 수 있으니, 해당 항목을 함께 가리려면 '전체'를 권장합니다. "
               "배경 항목은 항상 탐지 박스 영역만 가립니다.")
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

# 세션마다 임시 폴더 1개만 쓴다. 업로드가 바뀔 때만 다시 기록한다.
tmp_dir = st.session_state.get("tmp_dir")
if not tmp_dir or not os.path.isdir(tmp_dir):
    tmp_dir = tempfile.mkdtemp(prefix="mosaic_")
    st.session_state["tmp_dir"] = tmp_dir

upload_id = (uploaded.name, uploaded.size, getattr(uploaded, "file_id", ""))
suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
in_path = os.path.join(tmp_dir, "input" + suffix)
trim_path = os.path.join(tmp_dir, "trim.mp4")

if st.session_state.get("upload_id") != upload_id or not os.path.exists(in_path):
    old_in = st.session_state.get("input_path")
    if old_in and old_in != in_path:
        try:
            os.remove(old_in)
        except OSError:
            pass
    try:
        os.remove(trim_path)
    except OSError:
        pass
    with open(in_path, "wb") as f:
        f.write(uploaded.getvalue())
    st.session_state["upload_id"] = upload_id
    st.session_state["input_path"] = in_path
    st.session_state.pop("trim_key", None)

cap0 = cv2.VideoCapture(in_path)
fps0 = float(cap0.get(cv2.CAP_PROP_FPS) or 0.0)
if not (fps0 > 1.0):
    fps0 = 30.0
total0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
dur = total0 / max(1.0, fps0)
cap0.release()
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
        proc_in = in_path
    else:
        trim_sig = (upload_id, float(s_sec), float(e_sec))
        if st.session_state.get("trim_key") != trim_sig or not os.path.exists(trim_path):
            with st.spinner("구간 자르는 중..."):
                try:
                    trim_video(in_path, trim_path, s_sec,
                               e_sec if e_sec > 0 else None)
                except Exception as e:
                    st.error(f"구간 자르기 실패: {e}")
                    st.stop()
            st.session_state["trim_key"] = trim_sig
        proc_in = trim_path
else:
    s_sec, e_sec = 0.0, 0.0
    proc_in = in_path

st.subheader("1. 개인정보 항목 확인")
if st.button("다시 분석"):
    st.session_state.pop("preview_key", None)
    st.rerun()
st.caption("같은 설정이면 VLM을 다시 호출하지 않는다. API 키/프롬프트를 바꾸면 자동으로 다시 분석한다.")

# 분석 단계에서는 프롬프트 항목 + 모든 개인정보 카테고리를 후보로 찾는다.
# 체크리스트에서 실제로 가릴 항목을 고른다.
analysis_targets = list(CANDS)
preview_key = (upload_id, prompt_raw, tuple(analysis_targets),
               api_key, model_name, imgsz, trim_sig, VLM_MODEL)
if st.session_state.get("preview_key") != preview_key:
    err = None
    anno, persons, items = None, [], []
    with st.spinner("영상 샘플 분석 중... (앞/중간/끝)"):
        try:
            det = load_model_cached(model_name)
            with MODEL_LOCK:
                anno, persons, items = analyze_video(
                    proc_in, det, conf=0.20, imgsz=imgsz,
                    query_raw=prompt_raw, targets=analysis_targets,
                    api_key=api_key,
                    vlm_model=VLM_MODEL,
                    samples=3,
                )
        except Exception as e:
            err = e
    st.session_state["preview_key"] = preview_key
    st.session_state["preview_err"] = str(err) if err else ""
    st.session_state["preview_ok"] = anno is not None and err is None
    st.session_state["preview_anno"] = anno
    st.session_state["preview_persons"] = persons
    st.session_state["preview_items"] = items
    st.session_state["analysis_n"] = st.session_state.get("analysis_n", 0) + 1

ok = st.session_state.get("preview_ok", False)
preview_err = st.session_state.get("preview_err", "")
anno = st.session_state.get("preview_anno")
persons = st.session_state.get("preview_persons", [])
items = st.session_state.get("preview_items", [])
# 분석이 바뀌면 체크박스 키도 바뀌어서 기본값(체크)으로 새로 시작한다.
analysis_id = str(st.session_state.get("analysis_n", 0))

if preview_err:
    st.error(f"분석 실패: {preview_err}")
elif ok and persons:
    st.write(f"사람 {len(persons)}명 · 개인정보 후보 {len(items)}종 찾음")
    st.image(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB),
             caption="첫 샘플 화면 (초록 번호=개인정보 탐지 후보)")
elif ok:
    st.write(f"사람 없음 · 배경 개인정보 후보 {len(items)}종 찾음")
else:
    st.warning("영상을 읽을 수 없습니다.")

if items:
    prompt_items = [it for it in items if it.get("prompt_match")]
    extra_items = [it for it in items if not it.get("prompt_match")]

    def _render_item(it):
        cat = it["category"]
        unit = "명" if it.get("type") == "person" else "곳"
        label = f"{it.get('label_ko', cat)} · {it.get('count', 0)}{unit}"
        c1, c2 = st.columns([1, 9])
        with c1:
            th = it.get("thumb")
            if th is not None:
                try:
                    st.image(cv2.cvtColor(th, cv2.COLOR_BGR2RGB), width=84)
                except Exception:
                    pass
        with c2:
            checked = st.checkbox(label, value=True,
                                  key=f"chk_{analysis_id}_{cat}")
            src = "VLM" if it.get("source") == "vlm" else "로컬"
            st.caption(f"{'프롬프트 매칭' if it.get('prompt_match') else '추가 개인정보 후보'} · {src}")
        return checked

    if prompt_items:
        st.markdown("**프롬프트에서 찾은 항목**")
        for it in prompt_items:
            _render_item(it)
    if extra_items:
        st.markdown("**추가 개인정보 후보**")
        for it in extra_items:
            _render_item(it)
    st.caption("프롬프트 항목과 추가 후보 모두 기본으로 체크됩니다. "
               "가리지 않을 항목은 체크를 해제하세요.")
    found_cats = {it["category"] for it in items}
    missing = sorted(prompt_categories(prompt_raw) - found_cats)
    if missing:
        st.caption("프롬프트 항목 중 이번 샘플에서 못 찾은 것: "
                   + ", ".join(label_ko(c) for c in missing))

    selected_cats = [it["category"] for it in items
                     if st.session_state.get(f"chk_{analysis_id}_{it['category']}", True)]
else:
    selected_cats = []
    if ok:
        st.warning("가릴 항목을 찾지 못했습니다. 프롬프트를 구체적으로 쓰거나 "
                   "API 키를 확인하세요. (키가 없으면 로컬 YOLO-World로 찾습니다.)")

st.subheader("2. 처리")
run = st.button("모자이크 시작", type="primary", disabled=not selected_cats)
if not selected_cats:
    st.caption("최소 한 항목을 체크해야 시작할 수 있습니다.")

# 결과는 session_state에 남긴다. 다운로드/위젯 클릭으로 스크립트가 다시 돌아도
# 결과 화면이 사라지지 않는다.
run_sig = (preview_key, tuple(selected_cats), style, blocks, blur,
           area_mode, keyframe_stride, speed_label)
res = st.session_state.get("result")
if res is not None and res.get("sig") != run_sig:
    st.session_state.pop("result", None)
    res = None

if run:
    out_path = _unique_out_path(uploaded.name)
    out_name = os.path.basename(out_path)
    bar = st.progress(0)
    info = st.empty()

    def _pg(p, d, t, el):
        bar.progress(max(0.0, min(1.0, float(p))))
        if el > 1:
            info.write(f"{d}/{t} 프레임 · {el:.1f}초")

    t_start = time.time()
    stats = {}
    err = None
    with st.spinner("처리 중..."):
        try:
            with MODEL_LOCK:
                stats = mosaic_video(
                    proc_in, out_path,
                    targets=selected_cats, invert=False, conf=0.20,
                    model_name=model_name, imgsz=imgsz,
                    blocks=blocks, style=style, blur=blur,
                    ultra=(speed_label == "정밀"), area_mode=area_mode,
                    keyframe_stride=keyframe_stride,
                    encode_preset=encode_preset, progress_fn=_pg,
                    query_raw=prompt_raw, api_key=api_key,
                    vlm_model=VLM_MODEL,
                ) or {}
        except Exception as e:
            err = e
    elapsed = time.time() - t_start
    if err is not None:
        st.error(f"처리 실패: {err}")
        for p in (out_path, out_path + ".noaudio.mp4"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        st.session_state.pop("result", None)
        st.stop()

    region_n = stats.pop("_regions", 0)
    final_n = sum(1 for v in stats.values()
                  if isinstance(v, dict) and v.get("mosaic"))
    sel_label = next((v.get("selector") for v in stats.values()
                      if isinstance(v, dict) and v.get("selector")),
                     "vlm" if api_key.strip() else "world")
    try:
        H.add(uploaded.name, prompt_raw, ",".join(selected_cats), False,
              out_name, elapsed, query_raw=prompt_raw, selector=sel_label,
              summary=f"{final_n}명 가려짐 · 배경 {region_n}종")
    except Exception:
        pass
    res = {
        "sig": run_sig,
        "out_name": out_name,
        "out_path": out_path,
        "stats": stats,
        "elapsed": elapsed,
        "final_n": final_n,
        "region_n": region_n,
        "selected_cats": list(selected_cats),
    }
    st.session_state["result"] = res

with st.expander("지난 작업", expanded=False):
    for r in H.all_rows():
        fname = r.get("file")
        if not fname:
            continue
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"{r.get('at', '')} · {r.get('src', '')} → {fname}")
            extra = " · ".join(str(x) for x in
                               (r.get("selector"), r.get("summary")) if x)
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
                if (st.session_state.get("result") or {}).get("out_name") == fname:
                    st.session_state.pop("result", None)
                st.rerun()

if not res:
    st.stop()

out_name = res["out_name"]
out_path = res["out_path"]
stats = res["stats"]
elapsed = res["elapsed"]
final_n = res["final_n"]
region_n = res.get("region_n", 0)

st.success(f"완료 · {elapsed:.1f}초 걸림")
st.write(f"최종 사람 {final_n}명 · 배경 {region_n}종 가려짐")
st.caption("선택한 항목: " + ", ".join(label_ko(c) for c in res.get("selected_cats", [])))
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
        cats = v.get("cat_on", {})
        cats_str = ", ".join(label_ko(c) for c in cats) if cats else "-"
        st.write(f"추적 {tid} · {'가려짐' if v.get('mosaic') else '유지'} · {cats_str}")

st.markdown(
    """
    <div class="info-card">
        <h3 style="margin-top:0;">📊 결과 비교</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
if os.path.exists(out_path):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📹 원본**")
        st.video(proc_in, format="video/mp4")
    with c2:
        st.markdown("**🔒 모자이크 적용**")
        st.video(out_path, format="video/mp4")
    clicked = False
    with open(out_path, "rb") as f:
        clicked = st.download_button("결과 저장", f, file_name=out_name,
                                     mime="video/mp4")
    if clicked and auto_del:
        RT.remove_on_download(out_path)
        st.info("받기 끝나서 결과 파일을 지웠다. (보관 정책)")
else:
    st.info(f"결과 파일이 삭제되었다. ({RT.RETENTION_HOURS}시간 보관 정책 또는 다운로드 후 자동 삭제)")
