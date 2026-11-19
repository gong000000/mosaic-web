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
import logging
import os
import secrets
import shutil
import tempfile
import time

import cv2
import streamlit as st

from src.config import (CANDS, CATEGORY_TYPE, GEMINI_BASE_URL, OUTPUT_DIR,
                        PRESET_CATS, ROOT_DIR, VLM_MODEL, env_api_key,
                        label_ko)
from src.models import get_model
from src.preview import analyze_video
from src.render import normalize_video, trim_video
from src.render_service import MosaicSettings
from src import history as H
from src import jobqueue as JQ
from src import retention as RT
from src.auth import require_login

# blur.me 스튜디오처럼 항상 빠른 고정 설정을 쓴다.
# GPU(L4)에서는 yolo11m-seg + 960px가 충분히 빠르고, 사람 마스크 경계가
# n/640보다 훨씬 정밀하다. 테스트/저사양 환경은 환경변수로 낮출 수 있다.
FAST_MODEL = (os.environ.get("MOSAIC_MODEL")
              or os.environ.get("MOSAIC_SEG_MODEL")
              or "yolo11m-seg.pt")
try:
    FAST_IMGSZ = int(os.environ.get("MOSAIC_IMGSZ", "960"))
except (TypeError, ValueError):
    FAST_IMGSZ = 960
FAST_STRIDE = 15
# FAST_SAMPLES 는 균등 기본 샘플 수이고, 실제로는 여기에 '장면 전환',
# '컷 직전', '움직임 피크' 샘플이 더해져 FAST_MAX_SAMPLES 까지 선택된다.
FAST_SAMPLES = 8
FAST_MAX_SAMPLES = 20
FAST_PRESET = "veryfast"
FAST_CONF = 0.20
# 번호판 확대 탐지(차량 크롭) 허용 호출 수. 영상 1개 기준.
FAST_PLATE_BUDGET = 28

# 렌더 단계에서 VLM을 다시 호출하지 않는다. 체크리스트 분석 결과
# (priors)와 YOLO 추적만으로 전체 영상을 만든다. 같은 영상에 VLM 비용을
# 두 번 쓰지 않고, 렌더 결과도 분석 결과와 일치한다.

# 렌더는 지속형 큐가 백그라운드에서 하나씩 처리한다.
_HISTORY_PAGE_SIZE = 10


st.set_page_config(
    page_title="개인정보 보호 모자이크",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)
JQ.start_worker()


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
                "서버의 `GEMINI_API_KEY` 설정을 확인하세요(관리자에게 문의).")
    if ("429" in msg or "quota" in low or "rate limit" in low
            or "resource_exhausted" in low):
        compact = low.replace(" ", "").replace("_", "")
        if ("per day" in low or "daily" in low or "perday" in compact
                or "freetier" in compact):
            return ("Gemini 일일 호출 한도를 초과했습니다. "
                    "내일 다시 시도하거나 결제가 설정된 키를 사용하세요.")
        if ("per minute" in low or "requestsperminute" in compact
                or "rpm" in low):
            return ("Gemini 분당 호출 한도에 걸렸습니다. 잠시 후 다시 "
                    "시도하세요. 분석을 연속으로 많이 돌리면 걸릴 수 있습니다.")
        return ("Gemini 호출 한도를 초과했습니다. 잠시 후 다시 시도하거나 "
                "더 짧은 영상을 사용하세요.")
    if "404" in msg or "model not found" in low:
        return ("모델명을 찾을 수 없습니다. 서버의 `VLM_MODEL` 설정을 "
                "확인하세요(관리자에게 문의).")
    if "json" in low and "해석" in msg:
        return msg
    logging.getLogger("mosaic.app").error("VLM 오류(미분류): %s", msg[:500])
    return "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


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
    """열기·첫 프레임·길이·해상도를 한 번에 확인한다. 실패하면 None."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        h, w = frame.shape[:2]
        return total, fps, w, h
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


def _build_items(base_items, preset_cats):
    """분석 결과를 체크리스트 항목으로 바꾼다."""
    items = []
    seen = set()
    for item in base_items or []:
        copy = dict(item)
        copy["samples"] = dict(item.get("samples") or {})
        copy["prompt_match"] = copy.get("category") in preset_cats
        items.append(copy)
        seen.add(copy.get("category"))
    for cat in sorted(preset_cats - seen):
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


def _job_status_label(job):
    status = job.get("status")
    if status == "queued":
        return "대기 중"
    if status == "running":
        return "렌더 중"
    if status == "done":
        return "완료"
    if status == "cancelled":
        return "취소됨"
    if status == "failed":
        return "실패"
    if status == "retried":
        return "재시도됨"
    return str(status or "-")


def _render_job_card(job, pos, user):
    jid = int(job["id"])
    status = job.get("status")
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"**{pos}번째 작업** · {_job_status_label(job)}")
            st.caption(
                f"#{jid} · "
                f"{time.strftime('%m-%d %H:%M', time.localtime(float(job.get('created') or 0)))}")
        with c2:
            if status in ("queued", "running"):
                if job.get("cancel_requested"):
                    st.button("취소 중", key=f"cancel_{jid}", disabled=True)
                elif st.button("취소", key=f"cancel_{jid}"):
                    JQ.cancel(jid, user)
                    st.rerun()
        frac = min(1.0, max(0.0, float(job.get("progress") or 0)))
        if status == "queued":
            st.progress(0.0, text=f"대기 중 · {JQ.position(jid)}번째")
        elif status == "running":
            st.progress(frac, text=f"렌더 중 · {int(frac * 100)}%")
        elif status == "cancelled":
            st.warning("작업이 취소되었습니다.")
        elif status == "failed":
            st.error(f"렌더 실패: {job.get('error') or ''}")
            if job.get("input_path") and os.path.exists(job["input_path"]):
                if st.button("다시 시도", key=f"retry_{jid}"):
                    new_id = JQ.retry(jid, user)
                    if new_id:
                        ids = list(st.session_state.get("job_ids") or [])
                        ids.append(new_id)
                        st.session_state["job_ids"] = ids[-20:]
                        st.rerun()
                    else:
                        st.error("입력 영상이 만료되어 다시 시도할 수 없습니다.")
        elif status == "done":
            out = job.get("output_path") or ""
            if not out or not os.path.exists(out):
                st.warning("결과 파일이 삭제되었습니다.")
                return
            st.video(out, format="video/mp4", width="stretch")
            detail = " · ".join(part for part in (
                job.get("summary") or "", _fmt_size(out)) if part)
            if detail:
                st.caption(detail)
            out_name = os.path.basename(out)

            def _after_download(jid=jid, name=out_name, path=out):
                if st.session_state.get("auto_del", False):
                    H.delete(name)
                    ids = list(st.session_state.get("job_ids") or [])
                    st.session_state["job_ids"] = [
                        x for x in ids if int(x) != int(jid)]
                    if st.session_state.get("render_out") == path:
                        st.session_state.pop("render_out", None)

            with open(out, "rb") as f:
                st.download_button(
                    "모자이크 영상 다운로드", f,
                    file_name=out_name, mime="video/mp4",
                    on_click=_after_download,
                    key=f"download_mosaic_{jid}")


@st.fragment(run_every=2.0)
def _render_active_job_cards(items, user):
    for pos, jid in items:
        job = JQ.get(jid)
        if not job:
            continue
        if job.get("status") not in ("queued", "running"):
            st.rerun()
            return
        _render_job_card(job, pos, user)


def _render_history(user, expanded=False):
    """로그인한 계정의 지난 작업만 페이지 단위로 보여준다."""
    with st.expander("지난 작업", expanded=expanded):
        rows = H.all_rows(user=user)
        try:
            u = RT.usage()
            st.caption(
                f"저장 결과 {u['files']}개 · "
                f"{u['bytes'] / (1024 ** 3):.2f} GB "
                f"(상한 {u['max_files']}개 / "
                f"{u['max_bytes'] / (1024 ** 3):.0f} GB)")
        except Exception:
            pass
        try:
            active = JQ.active_jobs(user)
        except Exception:
            active = []
        if active:
            st.markdown("**대기/진행 중**")
            for j in active:
                if j.get("status") == "queued":
                    text = f"대기 {JQ.position(j['id'])}번째"
                else:
                    pct = int(float(j.get("progress") or 0) * 100)
                    text = f"렌더 중 {pct}%"
                at = time.strftime(
                    "%m-%d %H:%M",
                    time.localtime(float(j.get("created") or 0)))
                st.caption(f"#{j.get('id')} · {text} · {at}")
            st.caption("완료되면 아래 지난 작업에 표시됩니다.")
        if not rows:
            st.write("저장된 작업이 없습니다.")
            return
        total_pages = max(1, (len(rows) + _HISTORY_PAGE_SIZE - 1)
                          // _HISTORY_PAGE_SIZE)
        page = 1
        if total_pages > 1:
            page = int(st.number_input(
                f"페이지 (총 {len(rows)}건)", 1, total_pages, 1,
                key="hist_page"))
        start = (page - 1) * _HISTORY_PAGE_SIZE
        for row in rows[start:start + _HISTORY_PAGE_SIZE]:
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


def _render_checklist(items, analysis_id, preset_cats):
    """체크리스트를 그리고 선택된 카테고리 목록을 돌려준다."""
    if not items:
        return []

    prompt_items = [it for it in items if it.get("category") in preset_cats]
    extra_items = [it for it in items if it.get("category") not in preset_cats]
    filter_id = hashlib.sha1(
        ("|".join(sorted(preset_cats)) or "all").encode("utf-8")
    ).hexdigest()[:8]

    def _default_checked(cat):
        return (cat in preset_cats) if preset_cats else False

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


# VLM 분석은 항상 모든 개인정보 후보를 찾는다(프라이버시 우선).
# 사용자가 고른 프리셋은 '자동으로 체크할 항목'만 정한다.
ANALYSIS_QUERY = ("얼굴, 차량번호판, 주소·간판, 명찰, 신분증, "
                  "휴대폰 화면, 차량동승자를 모두 찾아줘")

st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">개인정보 모자이크</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# 위젯을 누를 때마다 앱이 다시 실행되므로, 디스크 스캔성 정리는 세션당
# 5분에 한 번만 한다.
if time.time() - st.session_state.get("_last_purge", 0) > 300:
    RT.enforce()
    H.remove_missing()
    st.session_state["_last_purge"] = time.time()

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
    st.caption(
        "분석을 위해 영상에서 뽑은 일부 프레임(모자이크 전 원본)이 "
        "Google Gemini API로 전송됩니다. 업로드하면 이에 동의한 것으로 봅니다.")

    st.subheader("가릴 항목")
    preset_name = st.selectbox("프리셋", list(PRESET_CATS.keys()),
                               key="preset_sel")
    preset_cats = set(PRESET_CATS.get(preset_name, ()))
    if preset_name == "직접 선택":
        st.caption("분석 후 아래 체크리스트에서 가릴 항목을 직접 고르세요.")

    st.divider()
    strip_audio_opt = st.checkbox(
        "오디오 제거", value=False,
        help="목소리·대화 속 이름/주소가 남지 않도록 결과 영상에서 소리를 뺍니다.")
    auto_del = st.checkbox("받으면 결과 자동 삭제", value=False)
    st.session_state["auto_del"] = bool(auto_del)

# 로그인만 해도 지난 작업을 볼 수 있게 업로드 여부와 무관하게 보여준다.
_render_history(current_user, expanded=not uploaded)

vlm_model = VLM_MODEL
vlm_base_url = GEMINI_BASE_URL
api_key = env_api_key()

if not uploaded:
    st.write("영상을 업로드하세요.")
    st.stop()

tmp_dir = st.session_state.get("tmp_dir")
if not tmp_dir or not os.path.isdir(tmp_dir):
    tmp_dir = tempfile.mkdtemp(prefix="mosaic_")
    st.session_state["tmp_dir"] = tmp_dir

raw_bytes = uploaded.getvalue()
_fid = getattr(uploaded, "file_id", None)
_cached = st.session_state.get("_digest_cache")
if _fid is not None and _cached and _cached[0] == _fid:
    digest = _cached[1]
else:
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    if _fid is not None:
        st.session_state["_digest_cache"] = (_fid, digest)
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
needs_norm = (suffix != ".mp4" or probe is None
              or (probe is not None and max(probe[2], probe[3]) > 1920))
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
total0, fps0, width0, height0 = probe
if not (fps0 > 0):
    fps0 = 30.0
dur = total0 / fps0
try:
    max_seconds = float(os.environ.get("MOSAIC_MAX_SECONDS", "900"))
except (TypeError, ValueError):
    max_seconds = 900.0
if max_seconds > 0 and dur > max_seconds:
    st.warning(f"영상이 깁니다({dur:.0f}초). 구간을 자르지 않으면 "
               f"최대 {max_seconds:.0f}초 제한에 걸립니다.")
st.write(f"길이 {dur:.1f}초 · {total0}프레임 · "
         f"{width0}×{height0}")

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
    if st.session_state.get("trim_key") is not None or trim_sig is not None:
        _invalidate_work()
    st.session_state["trim_key"] = trim_sig

_proc_probe = _probe_video(proc_in)
if _proc_probe is not None:
    _ptot, _pfps = _proc_probe[0], _proc_probe[1]
    _pdur = _ptot / (_pfps if _pfps > 0 else 30.0)
    if max_seconds > 0 and _pdur > max_seconds:
        st.error(f"처리 구간이 너무 깁니다({_pdur:.0f}초). "
                 f"최대 {max_seconds:.0f}초까지 처리할 수 있습니다. "
                 "'구간만 처리'로 필요한 부분만 선택하세요.")
        st.stop()

st.subheader("개인정보 항목")

if not api_key:
    st.error("GEMINI_API_KEY 가 설정되지 않았습니다. 서버 설정(.env 또는 Modal Secret)을 확인하세요.")
    st.stop()

key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
analysis_key = (upload_id, tuple(CANDS), key_hash, vlm_base_url, vlm_model,
                FAST_MODEL, FAST_IMGSZ, FAST_SAMPLES, FAST_MAX_SAMPLES,
                FAST_PLATE_BUDGET, trim_sig)

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
                query_raw=ANALYSIS_QUERY, targets=list(CANDS),
                api_key=api_key, vlm_model=vlm_model,
                vlm_base_url=vlm_base_url, samples=FAST_SAMPLES,
                max_samples=FAST_MAX_SAMPLES,
                plate_zoom_budget=FAST_PLATE_BUDGET)
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

items = _build_items(base_items, preset_cats) if ok else []

selected_cats = []
if items:
    analysis_id = hashlib.sha1(
        json.dumps(analysis_key, default=str, ensure_ascii=False)
        .encode("utf-8")).hexdigest()[:8]
    selected_cats = _render_checklist(items, analysis_id, preset_cats)
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
    if st.button("모자이크 시작", type="primary", key="start_mosaic"):
        out_path = _unique_out_path(uploaded.name, current_user)
        try:
            job_id = JQ.enqueue(
                user=current_user,
                input_path=proc_in,
                output_path=out_path,
                items=items,
                selected_cats=selected_cats,
                settings=settings,
                model_name=FAST_MODEL,
                imgsz=FAST_IMGSZ,
                stride=FAST_STRIDE,
                conf=FAST_CONF,
                preset=FAST_PRESET,
                query_raw=ANALYSIS_QUERY,
                vlm_model=vlm_model,
                vlm_base_url=vlm_base_url,
                src_name=uploaded.name,
                preset_name=preset_name,
                person_total=person_total,
                strip_audio=strip_audio_opt,
            )
        except Exception as e:
            _unlink_quiet(out_path)
            st.error(f"렌더 작업 등록 실패: "
                     f"{_friendly_vlm_error(e, api_key)}")
        else:
            ids = list(st.session_state.get("job_ids") or [])
            if job_id not in ids:
                ids.append(job_id)
            st.session_state["job_ids"] = ids[-20:]
            st.rerun()

# ---------------------------------------------------------------- 작업 카드
st.subheader("작업 진행 · 결과")
job_ids = []
for jid in list(st.session_state.get("job_ids") or []):
    try:
        job_ids.append(int(jid))
    except (TypeError, ValueError):
        pass
jobs = []
for pos, jid in enumerate(job_ids, 1):
    job = JQ.get(jid)
    if job and str(job.get("user")) == str(current_user):
        jobs.append((pos, job))

if not jobs:
    st.write("모자이크 시작을 누르면 영상이 여기에 표시됩니다.")
else:
    done_jobs = [(pos, job) for pos, job in jobs
                 if job.get("status") == "done"]
    active_jobs = ((pos, job["id"]) for pos, job in jobs
                   if job.get("status") in ("queued", "running"))
    other_jobs = [(pos, job) for pos, job in jobs
                  if job.get("status") in ("cancelled", "failed")]
    for pos, job in done_jobs:
        _render_job_card(job, pos, current_user)
    if done_jobs:
        _last = done_jobs[-1][1]
        st.session_state["render_out"] = _last.get("output_path") or ""
        st.session_state["render_summary"] = _last.get("summary") or ""
        st.session_state["render_sec"] = max(
            0.0, float(_last.get("finished") or 0)
            - float(_last.get("started") or 0))
    if active_jobs:
        _render_active_job_cards(tuple(active_jobs), current_user)
    for pos, job in other_jobs:
        _render_job_card(job, pos, current_user)
