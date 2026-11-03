"""회귀 고정. face.mp4의 5명 트랙이 모두 가려지는지 확인한다.
VLM은 스텁으로 대체해 트랙 ID 일치와 렌더만 검증한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\regression.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
TMP = r"C:\Users\gong0000\AppData\Local\Temp\opencode\regression_tmp.mp4"

# 2026-10-17 실측 고정값. 바꾸려면 전원 합의.
EXPECTED = {1: True, 2: True, 3: True, 4: True, 5: True}


def main():
    import src.pipeline as P
    from src.select import VLMSelector

    if not os.path.exists(FACE):
        print(f"SKIP face.mp4 없음: {FACE}")
        return 0

    class FaceAllSelector(VLMSelector):
        """VLM 대신 모든 사람을 face로 판정하는 테스트 스텁."""

        def detect_boxes(self, frame_bgr):
            return []

        def match_items(self, items, vlm_boxes):
            out = {}
            for it in items:
                out[it["key"]] = (True, {
                    "selector": "vlm", "targets": ["face"], "hits": ["face"],
                    "vlm_hit": True, "vlm_boxes": 0,
                    "vlm_match": {"phrase": "face", "category": "face",
                                  "label_ko": "얼굴", "conf": 1.0,
                                  "iou": 1.0, "source": "vlm"},
                })
            return out

    real = P.VLMSelector
    P.VLMSelector = FaceAllSelector
    fails = 0
    try:
        s = P.mosaic_video(
            FACE, TMP, targets=["face"], conf=0.20,
            model_name="yolo11m-seg.pt", imgsz=640,
            ultra=False, keyframe_stride=3, query_raw="회귀",
            api_key="fake-key")
        got = {k: v["mosaic"] for k, v in s.items()
               if isinstance(k, int)}
        if got == EXPECTED:
            print(f"OK face {got}")
        else:
            fails += 1
            print(f"FAIL face got={got} exp={EXPECTED}")
    finally:
        P.VLMSelector = real
        try:
            if os.path.exists(TMP):
                os.remove(TMP)
        except Exception:
            pass
    print("REGRESSION_PASS" if fails == 0 else f"REGRESSION_FAIL {fails}건")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
