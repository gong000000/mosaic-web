"""개인정보 보호 선택적 모자이크 웹 앱 (Streamlit + YOLO11-seg + VLM).

흐름:
  1) VLM이 영상 샘플에서 개인정보 항목을 찾아 체크리스트로 보여준다.
  2) 사용자가 가릴 항목/효과를 고른다.
  3) '모자이크 시작'을 누르면 전체 영상을 렌더링한다.
  4) 완성된 모자이크 영상을 재생해 확인하고 다운로드한다.

렌더는 사용자가 버튼을 누를 때만 실행한다. 업로드나 설정 변경만으로
전체 영상을 자동 렌더하지 않는다.
"""
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time

import cv2
import streamlit as st

from src.config import (CANDS, CATEGORY_TYPE, GEMINI_BASE_URL, OUTPUT_DIR,
                        PRESETS, ROOT_DIR, VLM_MODEL, env_api_key, label_ko)
from src.models import get_model
from src.preview import analyze_video
from src.render import normalize_video, trim_video
from src.render_service import MosaicSettings, RenderPlan
from src.vlm_package import prompt_categories
from src import history as H
from src import retention as RT
from src.auth import require_login

# blur.me 스튜디오처럼 항상 빠른 고정 설정을 쓴다.
FAST_MODEL = "yolo11n-seg.pt"
FAST_IMGSZ = 480
FAST_STRIDE = 15
FAST_SAMPLES = 4
FAST_PRESET = "veryfast"
FAST_CONF = 0.20

# 렌더 단계에서 VLM을 다시 호출하지 않는다. 체크리스트 분석 결과
# (priors)와 YOLO 추적만으로 전체 영상을 만든다. 같은 영상에 VLM 비용을
# 두 번 쓰지 않고, 렌더 결과도 분석 결과와 일치한다.
RENDER_VLM_CALLS = 0


st.set_page_config(
    page_title="개인정보 보호 모자이크",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css():
    path = os.path.join(ROOT_DIR, "assets", "style.css")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
current_user = require_login()


def _friendly_vlm_error(err, api_key=""):
    """VLM 오류를 사용자가 바로 조치할 수 있는 문장으로 바꾼다.
    키 문자열이 메시지에 섞여도 화면에 노출하지 않는다."""
    msg = str(err or "")
    if api_key:
        msg = msg.replace(str(api_key), "***")
    low = msg.lower()
    if ("401" in msg or "403" in msg or "invalid api key" in low
            or "unauthorized" in low or "permission" in low):
        return ("Gemini API 키가 올바르지 않거나 권한이 없습니다. "
                "`.env`의 `GEMINI_API_KEY`를 확인하세요.")
    if ("429" in msg or "quota" in low or "rate limit" in low
            or "resource_exhausted" in low):
        compact = low.replace(" ", "").replace("_", "")
        if ("per day" in low or "daily" in low or "perday" in compact
                or "freetier" in compact):
            return ("Gemini 일일 호출 한도를 초과했습니다. "
                    "내일 다시 시도하거나 결제가 설정된 키를 사용하세요.")
        return ("Gemini 호출 한도를 초과했습니다. 잠시 후 다시 시도하거나 "
                "더 짧은 영상을 사용하세요.")
    if "404" in msg or "model not found" in low:
        return ("모델명을 찾을 수 없습니다. `.env`의 `VLM_MODEL` 값을 "
                "확인하세요.")
    if "json" in low and "해석" in msg:
        return msg
    return msg


def _unique_out_path(src_name, user):
    """같은 초에 두 작업이 끝나도 결과 파일을 덮어쓰지 않게 한다.
    파일명 앞에 사용자 ID를 붙여 '지난 작업'을 계정별로 구분한다."""
    stem = os.path.splitext(os.path.basename(src_name))[0]
    ts = time.strftime("%m%d_%H%M%S")
    token = secrets.token_hex(2)
    base = f"{user}--{stem}_{ts}_{token}"
    path = os.path.join(OUTPUT_DIR, f"{base}.mp4")
    i = 1
    while os.path.exists(path):
        path = os.path.join(OUTPUT_DIR, f"{base}_{i}.mp4")
        i += 1
    return path


def _sweep_old_temp_dirs(max_hours=24):
    """24시간 지난 업로드 임시 폴더를 정리한다(서버 재시작과 무관하게)."""
    root = tempfile.gettempdir()
    now = time.time()
    removed = 0
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    for name in names:
        if not name.startswith("mosaic_"):
            continue
        path = os.path.join(root, name)
        try:
            if (os.path.isdir(path)
                    and now - os.path.getmtime(path) > max_hours * 3600):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


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


def _unlink_quiet(*paths):
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _invalidate_work():
    """분석/렌더 파생 상태를 지운다. 업로드·구간·키가 바뀔 때."""
    for key in ("analysis_key", "analysis", "render_out", "render_sig",
                "render_summary", "render_sec"):
        st.session_state.pop(key, None)


def _build_items(base_items, prompt_cats):
    """분석 결과를 체크리스트 항목으로 바꾼다.

    프롬프트가 요구했지만 샘플에서 못 찾은 항목도 '못 찾음'으로 남겨
    처리 단계에서 다시 찾을 수 있게 한다.
    """
    items = []
    seen = set()
    for item in base_items or []:
        copy = dict(item)
        copy["samples"] = dict(item.get("samples") or {})
        copy["prompt_match"] = copy.get("category") in prompt_cats
        items.append(copy)
        seen.add(copy.get("category"))
    for cat in sorted(prompt_cats - seen):
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
    return items


def _item_label(item):
    unit = "명" if item.get("type") == "person" else "곳"
    count = int(item.get("count", 0) or 0)
    label = f"{item.get('label_ko') or label_ko(item.get('category'))} · {count}{unit}"
    if item.get("missing"):
        label += " · 이번 샘플에서 못 찾음"
    return label


def _fmt_size(path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return "-"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.0f} KB"


def _render_checklist(items, analysis_id, prompt_cats):
    """체크리스트를 그리고 선택된 카테고리 목록을 돌려준다."""
    if not items:
        return []

    prompt_items = [it for it in items if it.get("category") in prompt_cats]
    extra_items = [it for it in items if it.get("category") not in prompt_cats]
    filter_id = hashlib.sha1(
        ("|".join(sorted(prompt_cats)) or "all").encode("utf-8")
    ).hexdigest()[:8]

    def _default_checked(cat):
        return (cat in prompt_cats) if prompt_cats else True

    def _check_key(cat):
        return f"chk_{analysis_id}_{filter_id}_{cat}"

    def _render_item(item):
        cat = item["category"]
        c1, c2 = st.columns([1, 9])
        with c1:
            thumb = item.get("thumb")
            if thumb is not None:
                try:
                    st.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                             width=84)
                except Exception:
                    pass
        with c2:
            # 샘플에서 못 찾은 항목은 렌더 단계에서 VLM을 다시 부르지
            # 않으므로 가릴 위치가 없다. 얼굴만 사람 추적으로 적용한다.
            missing = bool(item.get("missing"))
            allowed = (not missing) or cat == "face"
            st.checkbox(_item_label(item),
                        value=_default_checked(cat) if allowed else False,
                        key=_check_key(cat), disabled=not allowed)

    if prompt_items:
        st.markdown("**요청한 항목**")
        for item in prompt_items:
            _render_item(item)
    if extra_items:
        st.markdown("**추가로 찾은 개인정보**")
        for item in extra_items:
            _render_item(item)

    selected = [
        item["category"] for item in items
        if st.session_state.get(_check_key(item["category"]),
                                _default_checked(item["category"]))
    ]
    return selected


if not st.session_state.get("tmp_swept"):
    _sweep_old_temp_dirs(RT.RETENTION_HOURS)
    st.session_state["tmp_swept"] = True


def _apply_preset():
    text = PRESETS.get(st.session_state.get("preset_sel", ""), "")
    st.session_state["prompt_txt"] = text


if "prompt_txt" not in st.session_state:
    st.session_state["prompt_txt"] = PRESETS["얼굴 + 차량번호판"]

# 프롬프트는 사이드바에서 한 번만 입력한다.
prompt_raw = st.session_state.get("prompt_txt", "")

st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">개인정보 모자이크</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

RT.purge_old()
H.remove_missing()

with st.sidebar:
    st.caption(f"로그인: {current_user}")
    if st.button("로그아웃"):
        tmp_dir = st.session_state.get("tmp_dir")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        st.session_state.clear()
        st.rerun()

    st.header("설정")

    st.subheader("영상")
    uploaded = st.file_uploader("mp4, mov, avi", type=["mp4", "mov", "avi"])

    st.subheader("가릴 항목")
    st.selectbox("프리셋", list(PRESETS.keys()), key="preset_sel",
                 on_change=_apply_preset)
    st.text_area("프롬프트", key="prompt_txt", height=90)

    st.divider()
    auto_del = st.checkbox("받으면 결과 자동 삭제", value=False)

# Gemini 하나만 쓴다. 모델명은 .env의 VLM_MODEL, 엔드포인트는 고정이다.
vlm_model = VLM_MODEL
vlm_base_url = GEMINI_BASE_URL
api_key = env_api_key()

if not uploaded:
    st.write("영상을 업로드하세요.")
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

if st.session_state.get("upload_id") != upload_id or not os.path.exists(in_path):
    old_in = st.session_state.get("input_path")
    _unlink_quiet(old_in if old_in != in_path else None, norm_path, trim_path)
    with open(in_path, "wb") as f:
        f.write(raw_bytes)
    st.session_state["upload_id"] = upload_id
    st.session_state["input_path"] = in_path
    st.session_state["trim_key"] = None
    _invalidate_work()

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
        s_sec = st.number_input(
            "시작(초)", 0.0,
            float(max(0.0, dur - 0.5)) if dur > 0.5 else 0.0,
            0.0, step=0.5)
    with t2:
        e_sec = st.number_input("끝(초, 0=끝까지)", 0.0, float(dur), 0.0,
                                step=0.5)
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
            _invalidate_work()
        proc_in = trim_path
else:
    s_sec, e_sec = 0.0, 0.0
    proc_in = src_path

if st.session_state.get("trim_key") != trim_sig:
    # 구간 설정을 껐다 켜거나 바꾸면 이전 분석/렌더는 현재 영상과 다르다.
    if st.session_state.get("trim_key") is not None or trim_sig is not None:
        _invalidate_work()
    st.session_state["trim_key"] = trim_sig

st.subheader("개인정보 항목")

if not api_key:
    st.error("`.env`에 `GEMINI_API_KEY`를 설정한 뒤 앱을 다시 실행하세요.")
    st.stop()

key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
analysis_key = (upload_id, tuple(CANDS), key_hash, vlm_base_url, vlm_model,
                FAST_MODEL, FAST_IMGSZ, FAST_SAMPLES, trim_sig)

if st.button("다시 분석", key="reanalyze"):
    _invalidate_work()
    st.rerun()

if st.session_state.get("analysis_key") != analysis_key:
    _invalidate_work()
    with st.spinner("영상 샘플 분석 중..."):
        err = None
        anno, persons, found = None, [], []
        try:
            detector = get_model(FAST_MODEL)
            anno, persons, found = analyze_video(
                proc_in, detector, conf=FAST_CONF, imgsz=FAST_IMGSZ,
                query_raw=prompt_raw, targets=list(CANDS),
                api_key=api_key, vlm_model=vlm_model,
                vlm_base_url=vlm_base_url, samples=FAST_SAMPLES)
        except Exception as e:
            err = e
    st.session_state["analysis_key"] = analysis_key
    st.session_state["analysis"] = {
        "ok": anno is not None and err is None,
        "err": _friendly_vlm_error(err, api_key) if err else "",
        "anno": anno,
        "persons": persons,
        "items": found,
    }

analysis = st.session_state.get("analysis") or {}
ok = bool(analysis.get("ok"))
preview_err = analysis.get("err") or ""
persons = analysis.get("persons") or []
base_items = analysis.get("items") or []
person_total = max(
    [int(it.get("count", 0) or 0) for it in base_items
     if it.get("type") == "person"] + [len(persons)])

if preview_err:
    st.error(f"분석 실패: {preview_err}")
elif ok:
    st.write(f"사람 {person_total}명 · 개인정보 후보 {len(base_items)}종")

prompt_cats = prompt_categories(prompt_raw)
items = _build_items(base_items, prompt_cats) if ok else []

selected_cats = []
if items:
    analysis_id = hashlib.sha1(
        json.dumps(analysis_key, default=str, ensure_ascii=False)
        .encode("utf-8")).hexdigest()[:8]
    selected_cats = _render_checklist(items, analysis_id, prompt_cats)
elif ok:
    st.warning("이번 샘플에서 가릴 항목을 찾지 못했습니다.")

# ---------------------------------------------------------------- 설정
st.subheader("모자이크 설정")

area_label = st.segmented_control(
    "가리는 범위",
    ["👤 얼굴만", "🧍 상반신", "🧑 전체"],
    default="👤 얼굴만",
    key="area_mode_ui")
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
    blocks = st.slider("강도", 18, 44, 30)
    blur = 61
elif style == "blur":
    blocks = 30
    blur = st.slider("흐림 정도", 21, 101, 61, step=2)
else:
    blocks, blur = 30, 61

# ---------------------------------------------------------------- 렌더
st.subheader("모자이크 시작")
settings = MosaicSettings(style=style, blocks=int(blocks),
                          blur=int(blur), area_mode=area_mode)

if not ok:
    pass
elif not selected_cats:
    st.warning("가릴 항목을 하나 이상 체크하세요.")
else:
    source_token = f"{upload_id[2]}|{trim_sig}"
    plan = RenderPlan.from_items(items, selected_cats, settings)
    current_sig = plan.render_signature(
        source_token, FAST_MODEL, FAST_IMGSZ, FAST_STRIDE, FAST_CONF)

    if st.button("모자이크 시작", type="primary", key="start_mosaic"):
        out_path = _unique_out_path(uploaded.name, current_user)
        progress = st.progress(0.0, text="렌더 준비 중...")

        def on_progress(frac, done, total, elapsed):
            try:
                progress.progress(
                    min(1.0, max(0.0, float(frac))),
                    text=f"{int(done)}/{max(1, int(total))} 프레임 · "
                         f"{float(elapsed):.0f}초")
            except Exception:
                pass

        try:
            pass1_sig = plan.pass1_signature(
                source_token, FAST_MODEL, FAST_IMGSZ, FAST_STRIDE,
                FAST_CONF)
            cache_path = os.path.join(tmp_dir, f"pass1_{pass1_sig}.pkl")
            started = time.time()
            stats = plan.render(
                proc_in, out_path,
                model_name=FAST_MODEL, imgsz=FAST_IMGSZ,
                stride=FAST_STRIDE, conf=FAST_CONF, preset=FAST_PRESET,
                progress_fn=on_progress, query_raw=prompt_raw,
                api_key=api_key, vlm_model=vlm_model,
                vlm_base_url=vlm_base_url, cache_path=cache_path,
                use_cache=True, vlm_call_limit=RENDER_VLM_CALLS)
            elapsed = time.time() - started
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("결과 영상이 만들어지지 않았습니다.")
            # 인원수는 렌더 중 동시 트랙 수보다 분석 체크리스트의
            # 동시 인원(person_total)을 쓴다. 로컬 추적은 해상/중복으로
            # 순간 최대값이 부풀 수 있어 요약이 실제보다 커진다.
            person_selected = any(
                CATEGORY_TYPE.get(c) == "person" for c in selected_cats)
            if "face" in selected_cats:
                face_count = person_total
            else:
                face_count = stats.get("_faces")
            if face_count is None:
                face_count = sum(1 for key, value in stats.items()
                                 if isinstance(key, int) and value.get("mosaic"))
            summary = f"{len(selected_cats)}개 항목"
            if person_selected:
                summary += f" · 사람 {int(face_count)}명"
            regions = int(stats.get("_regions", 0) or 0)
            if regions:
                summary += f" · 배경 {regions}곳"
            st.session_state["render_out"] = out_path
            st.session_state["render_sig"] = current_sig
            st.session_state["render_summary"] = summary
            st.session_state["render_sec"] = elapsed
            H.add(src_name=uploaded.name, prompt=prompt_raw,
                  target=",".join(selected_cats), invert=False,
                  out_name=os.path.basename(out_path), seconds=elapsed,
                  query_raw=prompt_raw, selector="vlm", summary=summary)
            progress.progress(1.0, text="완료")
            st.rerun()
        except Exception as e:
            st.session_state.pop("render_out", None)
            st.session_state.pop("render_sig", None)
            st.session_state.pop("render_summary", None)
            _unlink_quiet(out_path)
            st.error(f"모자이크 렌더 실패: "
                     f"{_friendly_vlm_error(e, api_key)}")

# ---------------------------------------------------------------- 미리보기/다운로드
st.subheader("미리보기 · 다운로드")
render_out = st.session_state.get("render_out")
render_valid = bool(render_out and os.path.exists(render_out))
if render_valid:
    st.video(render_out, format="video/mp4", width="stretch")
    summary = st.session_state.get("render_summary") or ""
    detail = " · ".join(part for part in (
        summary, _fmt_size(render_out)) if part)
    if detail:
        st.caption(detail)
    try:
        with open(render_out, "rb") as f:
            video_bytes = f.read()
    except OSError:
        video_bytes = b""
    if video_bytes:
        download_name = (os.path.splitext(uploaded.name)[0]
                         + "_mosaic.mp4")
        out_name = os.path.basename(render_out)

        def _after_download(name=out_name):
            if auto_del:
                H.delete(name)
                st.session_state.pop("render_out", None)
                st.session_state.pop("render_sig", None)
                st.session_state.pop("render_summary", None)
                st.session_state.pop("render_sec", None)

        st.download_button(
            "모자이크 영상 다운로드", video_bytes,
            file_name=download_name, mime="video/mp4",
            on_click=_after_download, key="download_mosaic")
else:
    st.write("모자이크 시작을 누르면 영상이 여기에 표시됩니다.")

with st.expander("지난 작업", expanded=False):
    prefix = f"{current_user}--"
    rows = [r for r in H.all_rows()
            if str(r.get("file", "")).startswith(prefix)]
    if not rows:
        st.write("저장된 작업이 없습니다.")
    for row in rows:
        fname = row.get("file")
        if not fname:
            continue
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"{row.get('at', '')} · {row.get('src', '')} → {fname}")
        with c2:
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button("받기", f, file_name=fname,
                                       key=f"dl_{fname}")
        with c3:
            if st.button("삭제", key=f"del_{fname}"):
                H.delete(fname)
                if st.session_state.get("render_out") == os.path.join(
                        OUTPUT_DIR, fname):
                    _invalidate_work()
                st.rerun()
