import os
import time
import itertools
import tempfile
import cv2
import streamlit as st

from src.config import CANDS, OUTPUT_DIR
from src.prompts import parse_korean_prompt, warn_abstract, GUIDE
from src.clothing import hist_corr
from src.preview import analyze_first_frame
from src.render import trim_video, cleanup_tmp
from src.models import get_model
from src.pipeline import mosaic_video
from src import history as H
from src import retention as RT

KO = {
    "green shirt": "초록 셔츠",
    "white sleeveless vest": "흰 민소매",
    "white blazer": "흰 재킷",
    "beige plaid shirt": "베이지 체크",
    "striped sweater": "줄무늬 니트",
    "red shirt": "빨간 셔츠",
    "blue shirt": "파란 셔츠",
    "black shirt": "검정 셔츠",
    "dark navy uniform": "짙은 제복",
    "light skin": "흰 피부",
    "dark skin": "검은 피부",
    "yellow skin": "노란 피부",
}

WORLD_KO = {
    "hat": "모자", "cap": "모자", "beanie": "비니",
    "glasses": "안경", "sunglasses": "선글라스",
    "umbrella": "우산", "safety helmet": "안전모", "hard hat": "안전모",
    "helmet": "헬멧", "red helmet": "빨간 안전모",
    "face mask": "마스크", "scarf": "목도리", "gloves": "장갑",
    "backpack": "가방", "bag": "가방", "handbag": "가방",
    "tie": "넥타이", "necktie": "넥타이",
    "headphones": "이어폰", "earphones": "이어폰",
    "green shirt": "초록 셔츠", "red shirt": "빨간 셔츠", "blue shirt": "파란 셔츠",
    "black shirt": "검정 셔츠", "white vest": "흰 조끼", "white blazer": "흰 재킷",
    "plaid shirt": "체크 셔츠", "striped sweater": "줄무늬 니트", "navy uniform": "짙은 제복",
}


def _ko(name):
    return KO.get(name, WORLD_KO.get(name, name))

st.set_page_config(page_title="프롬프트 모자이크", layout="wide")
st.markdown(
    "<style>"
    ".block-container{max-width:1080px;padding-top:2rem}"
    "video{border-radius:12px}"
    ".stButton>button[kind='primary']{border-radius:10px;padding:.6rem 1.4rem}"
    "</style>",
    unsafe_allow_html=True,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def load_model_cached(name):
    return get_model(name)


st.title("프롬프트 모자이크")
st.caption("문장으로 골라 가리는 영상 모자이크 · 외부 전송 없음, 이 PC에서만 처리")
purged = RT.purge_old()
if purged:
    st.caption(f"24시간 지난 결과 {purged}개 자동 삭제됨")

with st.sidebar:
    st.header("설정")

    def _fill_prompt(t):
        st.session_state["prompt_raw"] = t

    if "prompt_raw" not in st.session_state:
        st.session_state["prompt_raw"] = "초록 셔츠 입은 사람 모자이크해줘"
    prompt_raw = st.text_input(
        "지시문", key="prompt_raw",
        help=GUIDE,
    )
    w = warn_abstract(prompt_raw)
    if w:
        st.warning(w)
    with st.expander("지시문 예시"):
        for ex in ("초록 셔츠 입은 사람 모자이크해줘",
                   "빨간 옷 입은 사람만 모자이크해줘",
                   "모자 쓴 사람 가려줘",
                   "흰옷 입은 사람 제외하고 모두 가려줘"):
            st.button(ex, key=f"ex_{ex}", on_click=_fill_prompt, args=(ex,))
    st.divider()
    auto_target, auto_inv, auto_note = parse_korean_prompt(prompt_raw)
    if auto_note:
        st.caption(auto_note)
    from src.world import is_fallback_targets, open_phrases_for_raw
    world_only = bool(open_phrases_for_raw(prompt_raw)) and is_fallback_targets(prompt_raw, auto_target)
    if world_only:
        st.caption("색 선택은 비워둬도 된다. 열린 말은 월드가 본다.")
    targets = st.multiselect(
        "가릴 옷 (여러 개 가능)", CANDS,
        default=[] if world_only else ([t for t in auto_target if t in CANDS] or [CANDS[0]]),
        format_func=lambda x: KO.get(x, x),
    )
    if not targets and not world_only:
        targets = [CANDS[0]]
    target = targets
    invert = st.checkbox("고른 옷 제외하고 모두 가리기", value=auto_inv)
    judge_label = st.radio("판정 방식", ["자동(권장)", "색 규칙", "월드 검출"], index=0,
                           help="자동: 색·피부는 규칙, 안경·우산·모자·안전모는 월드 검출. 모두 이 PC에서 돈다.")
    world_mode = {"자동(권장)": "auto", "색 규칙": "heuristic", "월드 검출": "world"}[judge_label]
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
    speed_label = st.radio("속도", ["빠름", "보통", "정밀"], index=1)
    speed_map = {
        "빠름": ("yolo11n-seg.pt", 480, "veryfast"),
        "보통": ("yolo11m-seg.pt", 640, "veryfast"),
        "정밀": ("yolo11x-seg.pt", 960, "slow"),
    }
    model_name, imgsz, encode_preset = speed_map[speed_label]
    with st.expander("세부 조정"):
        conf = st.slider("검출", 0.08, 0.60, 0.20,
                         help="사람 찾기 민감도. 낮추면 멀리·작게 보이는 사람도 찾는데 "
                              "간판 속 인물까지 사람으로 볼 수 있다.")
        margin = st.slider("판정", 0.01, 0.20, 0.04,
                           help="색 1등과 2등의 점수 차이. 낮추면 많이 가려지고(오탐↑), "
                                "높이면 확실할 때만 가린다(미탐↑).")
        key_stride = st.slider("키프레임 간격 (클수록 빠름)", 1, 10, 3,
                               help="몇 프레임마다 판정할지. 클수록 1패스가 빠르다. "
                                    "최종 렌더는 매 프레임이라 가림 품질은 같다.")
    auto_del = st.checkbox("받으면 결과 자동 삭제", value=False)

st.subheader("1. 영상 올리기")
uploaded = st.file_uploader("mp4, mov, avi", type=["mp4", "mov", "avi"])
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

st.subheader("2. 대상 확인")
ref_hists, ref_flags = [], []
ref_bank, n_checked, ref_thresh_auto = None, 0, 0.82
if ok:
    det = load_model_cached(model_name)
    anno, persons = analyze_first_frame(first, det, conf, imgsz, targets, margin,
                                        query_raw=prompt_raw, world_mode=world_mode)
    if persons:
        st.write(f"사람 {len(persons)}명 찾음. 직접 고르면 그 사람만 처리한다.")
        st.caption("첫 화면에 없어도 뒤에 나오는 사람은 처리 때 자동으로 판단한다.")
        try:
            from src.world import open_phrases_for_raw as _ops
            _w = _ops(prompt_raw)
            if _w and world_mode != "heuristic":
                st.caption(f"‘{','.join(_w)}’는 월드 검출로 판단한다. 이 PC에서만 돈다.")
        except Exception:
            pass
        st.image(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB), caption="첫 화면 전체 (초록 번호=가려짐 후보)")
        for p in persons:
            bi, crop, is_t, top1, hh = p["bi"], p["crop"], p["is_t"], p["top1"], p["hh"]
            hits = p.get("hits", [])
            badge = "가려짐" if ((not is_t) if invert else is_t) else "유지"
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
                st.caption(f"추정: {_ko(top1)}" + (f" (해당: {', '.join(_ko(h) for h in hits)})" if hits else ""))
                ck = st.checkbox(f"{bi+1}번 직접 지정", value=False, key=f"ck_{uploaded.name}_{bi}")
            ref_hists.append(hh)
            ref_flags.append(bool(ck))
        n_checked = sum(1 for c in ref_flags if c)
        if n_checked:
            ref_bank = [(h, c) for h, c in zip(ref_hists, ref_flags)]
            checked_hists = [h for h, c in ref_bank if c]
            maxc = 0.0
            nums = [i + 1 for i, c in enumerate(ref_flags) if c]
            for a, b in itertools.combinations(range(len(checked_hists)), 2):
                c = hist_corr(checked_hists[a], checked_hists[b])
                maxc = max(maxc, c)
                if c >= 0.82:
                    st.warning(f"{nums[a]}번과 {nums[b]}번이 비슷하다. 한 명만 고르는 게 낫다.")
            ref_thresh_auto = min(0.95, max(0.82, maxc + 0.05))
            st.info(f"직접 지정 {n_checked}명 적용된다. 안 고른 사람은 제외 기준으로 함께 본다.")
        else:
            ref_thresh_auto = 0.82
    else:
        st.warning("사람을 못 찾았다. 다른 영상으로 해봐라.")
        ref_thresh_auto = 0.82
else:
    ref_thresh_auto = 0.82

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
    stats = mosaic_video(proc_in, out_path, target=targets, invert=invert, conf=conf,
                 margin=margin, model_name=model_name, imgsz=imgsz,
                 blocks=blocks, style=style, blur=blur,
                 ultra=(speed_label == "정밀"), area_mode=area_mode,
                 ref_bank=ref_bank,
                 ref_thresh=ref_thresh_auto, keyframe_stride=key_stride,
                 encode_preset=encode_preset, progress_fn=_pg, query_raw=prompt_raw,
                 world_mode=world_mode) or {}
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
        st.write(f"사람 {tid} · {'가려짐' if v.get('mosaic') else '유지'}")
try:
    _sels = [v.get("selector", "heuristic") for v in stats.values()]
    _sel = max(set(_sels), key=_sels.count) if _sels else "heuristic"
except Exception:
    _sel = "heuristic-ref" if ref_bank else "heuristic"
H.add(uploaded.name, prompt_raw, target, invert, out_name, elapsed,
      query_raw=prompt_raw, selector=_sel,
      summary=f"{final_n}명 가려짐")
c1, c2 = st.columns(2)
with c1:
    st.write("원본")
    st.video(proc_in, format="video/mp4")
with c2:
    st.write("결과")
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
