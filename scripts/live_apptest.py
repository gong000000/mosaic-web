"""배포 환경(Modal 컨테이너) 안에서 앱 전체 흐름을 검사하는 스크립트.

브라우저 로그인/파일 선택 창만 우회하고, 그 뒤의 실제 앱 코드
(업로드 → Gemini 분석 → 체크리스트 → 렌더 → 다운로드 버튼)을
Streamlit AppTest로 실행한다.

사용:
  python scripts/live_apptest.py /data/_smoke/face.mp4
  python scripts/live_apptest.py /data/_smoke/face.mp4 --stub   # VLM 호출 없이

결과물과 작업 기록은 테스트 사용자(smoketest)로 만들고, 끝나면 지운다.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
USER = "smoketest"


def _stub_vlm(frame, phrases, api_key, **kwargs):
    """YuNet 로컬 얼굴만으로 체크리스트를 만들게 하는 스텁."""
    return []


def _patch_vlm():
    import src.preview as pv
    import src.vlm_package as vp
    vp.detect_with_vlm = _stub_vlm
    pv.detect_with_vlm = _stub_vlm
    # preview가 import한 심볼도 바꾼다.
    try:
        import src.preview as pv2
        pv2.detect_with_vlm = _stub_vlm
    except Exception:
        pass


def _keys(at):
    els = list(at.button) + list(at.download_button)
    return [str(getattr(el, "key", "")) for el in els]


def _errors(at):
    return [str(e.value) for e in at.error]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--stub", action="store_true",
                        help="Gemini를 호출하지 않고 로컬 얼굴만 사용")
    parser.add_argument("--keep", action="store_true",
                        help="결과/기록을 지우지 않는다")
    args = parser.parse_args()
    if not os.path.exists(args.video):
        print(f"SKIP 영상 없음: {args.video}")
        return 2

    if args.stub:
        _patch_vlm()

    from streamlit.testing.v1 import AppTest
    from src import history as H
    from src import retention as RT
    RT.purge_old = lambda *a, **k: 0

    at = AppTest.from_file(APP, default_timeout=1800)
    at.session_state["user"] = USER
    t0 = time.time()
    at.run(timeout=600)
    if _errors(at):
        print("FAIL 초기 실행:", _errors(at))
        return 1

    with open(args.video, "rb") as f:
        data = f.read()
    at.sidebar.file_uploader[0].upload(
        os.path.basename(args.video), data, "video/mp4")
    t1 = time.time()
    at.run(timeout=1800)
    analysis = at.session_state.get("analysis")
    if not isinstance(analysis, dict) or not analysis.get("ok"):
        print("FAIL 분석:", analysis.get("err") if isinstance(analysis, dict)
              else "analysis 상태 없음")
        out = at.session_state.get("render_out")
        if out and os.path.exists(out):
            os.remove(out)
        return 1
    analyze_sec = time.time() - t1
    items = analysis.get("items") or []
    cats = {it.get("category") for it in items}
    print(f"OK 분석 {analyze_sec:.1f}초 · 항목 {sorted(cats)}")

    # 얼굴만 프리셋으로 고정 (번호판 등 추가 항목은 체크 해제)
    try:
        at.sidebar.selectbox(key="preset_sel").select("얼굴만")
        at.run(timeout=600)
    except Exception as e:
        print("WARN 프리셋 선택 실패:", e)

    buttons = [b for b in at.button if b.key == "start_mosaic"]
    if not buttons:
        print("FAIL 모자이크 시작 버튼 없음:", _keys(at))
        return 1
    t2 = time.time()
    buttons[0].click()
    at.run(timeout=1800)
    render_sec = time.time() - t2
    out = at.session_state.get("render_out") or ""
    if not out or not os.path.exists(out) or os.path.getsize(out) < 1024:
        print("FAIL 렌더 결과 없음:", out, _errors(at))
        return 1
    if "download_mosaic" not in _keys(at):
        print("FAIL 다운로드 버튼 없음:", _keys(at))
        return 1
    print(f"OK 렌더 {render_sec:.1f}초 · {os.path.getsize(out)/1024:.0f} KB")
    print(f"OUTPUT {out}")
    print(f"OK 다운로드 버튼 노출 · 사용자 {USER}")
    print(f"TOTAL {time.time()-t0:.1f}초")

    if not args.keep:
        try:
            H.delete(os.path.basename(out))
        except Exception:
            pass
        try:
            os.remove(out)
        except OSError:
            pass
        tmp_dir = at.session_state.get("tmp_dir")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print("LIVE_APPTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
