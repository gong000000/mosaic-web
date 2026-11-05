"""검증용. 첫 프레임에서 VLM이 찾은 개인정보 항목을 출력한다.

VLM API 키가 필수다. 전체 영상을 처리하지 않고 첫 프레임만 분석한다.
실행 예:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate.py street.mp4 --api-key "키"
  $env:VLM_API_KEY="키"; .\\.venv\\Scripts\\python.exe scripts\\evaluate.py street.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from src.config import CANDS, VLM_MODEL, VLM_PROVIDERS
from src.models import get_model
from src.preview import analyze_first_frame_items


def main(video, api_key="", provider="OpenAI", model="", conf=0.20,
         model_name="yolo11m-seg.pt", imgsz=640):
    api_key = str(api_key or os.environ.get("VLM_API_KEY", "")).strip()
    if not api_key:
        print("VLM_KEY_REQUIRED: --api-key 또는 VLM_API_KEY 환경 변수가 필요합니다.")
        return 2

    preset = VLM_PROVIDERS.get(provider, VLM_PROVIDERS["OpenAI"])
    vlm_model = model or preset["model"] or VLM_MODEL
    base_url = preset.get("base_url")

    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    cap.release()
    if not ok:
        print("READ_FAIL")
        return 1

    _anno, persons, items = analyze_first_frame_items(
        first, get_model(model_name), conf, imgsz,
        query_raw="", targets=list(CANDS), api_key=api_key,
        vlm_model=vlm_model, vlm_base_url=base_url)

    print(f"provider={provider} model={vlm_model} persons={len(persons)} "
          f"items={len(items)}")
    for it in items:
        boxes = it.get("boxes") or []
        print(f"item category={it['category']} label={it['label_ko']} "
              f"type={it['type']} boxes={len(boxes)} "
              f"prompt_match={it.get('prompt_match')}")
        for xyxy in boxes[:3]:
            print(f"  box={xyxy}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="face.mp4")
    ap.add_argument("--api-key", default="",
                    help="VLM API 키 (없으면 VLM_API_KEY 환경 변수 사용)")
    ap.add_argument("--provider", default="OpenAI",
                    choices=list(VLM_PROVIDERS.keys()))
    ap.add_argument("--vlm-model", default="",
                    help="제공자 모델명 (비우면 제공자 기본값)")
    ap.add_argument("--model", default="yolo11m-seg.pt",
                    help="사람/차량 검출용 YOLO11-seg 모델")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    a = ap.parse_args()
    raise SystemExit(main(a.video, a.api_key, a.provider, a.vlm_model,
                          a.conf, a.model, a.imgsz))
