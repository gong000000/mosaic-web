"""실제 Gemini API 키로 분석 품질을 확인하는 스크립트.

앱과 같은 고정 설정(샘플 4장, yolo11n-seg, imgsz 480)으로 분석하고,
찾은 카테고리/박스 수/소요 시간을 출력한다. 키는 화면에 출력하지 않는다.

사용:
  python scripts/live_vlm.py C:\\path\\video.mp4

키는 `.env`/환경변수에서 읽고, 없으면 getpass로 안전하게 입력받는다.
"""
import argparse
import getpass
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from src.config import (CANDS, GEMINI_BASE_URL, VLM_MODEL, env_api_key,
                        label_ko)
from src.models import get_model
from src.preview import analyze_video


def _safe(text, api_key):
    text = str(text or "")
    if api_key:
        text = text.replace(str(api_key), "***")
    return text


def main():
    parser = argparse.ArgumentParser(description="실제 Gemini VLM 분석 검증")
    parser.add_argument("video", nargs="?",
                        default=r"C:\Users\gong0000\Downloads\face.mp4")
    parser.add_argument("--vlm-model", default="",
                        help="Gemini 모델명 (비우면 .env의 VLM_MODEL)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"SKIP 영상 없음: {args.video}")
        return 0

    api_key = env_api_key()
    source = ".env/환경변수"
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass(
            "Gemini API 키 (화면에 표시되지 않음): ").strip()
        source = "직접 입력"
    if not api_key:
        print("API 키가 없습니다. .env에 GEMINI_API_KEY를 설정하거나, "
              "터미널에서 직접 입력하세요.")
        return 2

    model_name = args.vlm_model or VLM_MODEL
    print(f"VLM Gemini · {model_name} · 키 출처: {source}")
    print(f"영상 {args.video}")

    t0 = time.time()
    err = None
    anno, persons, items = None, [], []
    try:
        detector = get_model("yolo11n-seg.pt")
        anno, persons, items = analyze_video(
            args.video, detector, conf=0.20, imgsz=480, samples=4,
            query_raw="얼굴, 차량번호판, 주소·간판, 명찰, 신분증, "
                      "휴대폰 화면, 차량동승자를 모두 찾아줘",
            targets=list(CANDS), api_key=api_key, vlm_model=model_name,
            vlm_base_url=GEMINI_BASE_URL)
    except Exception as e:
        err = _safe(f"{type(e).__name__}: {e}", api_key)

    elapsed = time.time() - t0
    if err:
        print(f"FAIL {err}")
        return 1

    print(f"사람 {len(persons)}명 · 항목 {len(items)}종 · {elapsed:.1f}초")
    for item in items:
        cat = item.get("category")
        print(f"  - {label_ko(cat)} ({cat}) · "
              f"{item.get('count', 0)}건 · 박스 {item.get('box_count', 0)}개")
    if not items:
        print("  (VLM이 개인정보 항목을 찾지 못했습니다)")

    if anno is not None:
        out_dir = os.path.join(tempfile.gettempdir(), "mosaic_live_vlm")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, time.strftime("live_%m%d_%H%M%S.jpg"))
        cv2.imwrite(out, anno)
        print(f"어노테이션 이미지: {out}")
    print("LIVE_VLM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
