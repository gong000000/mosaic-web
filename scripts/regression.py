"""회귀 고정. face.mp4의 5명 트랙이 모두 가려지는지 확인한다.
VLM은 스텁으로 대체해 트랙 ID 일치와 렌더만 검증한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\regression.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _sample(env, name):
    """테스트 영상 경로: 환경변수 > 저장소의 samples/ > ~/Downloads."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.environ.get(env),
              os.path.join(root, "samples", name),
              os.path.join(os.path.expanduser("~"), "Downloads", name)):
        if p and os.path.exists(p):
            return p
    return os.path.join(root, "samples", name)


FACE = _sample("MOSAIC_TEST_FACE", "face.mp4")
TMP = os.path.join(tempfile.gettempdir(), "regression_tmp.mp4")

# 트랙 ID는 모델/해상도에 따라 달라질 수 있으므로 ID를 고정하지 않고
# '검출된 사람 트랙이 모두 가려졌는가'를 본다.
MIN_PERSONS = 5


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
            model_name=os.environ.get("MOSAIC_MODEL", "yolo11m-seg.pt"),
            imgsz=int(os.environ.get("MOSAIC_IMGSZ", "960")),
            ultra=False, keyframe_stride=15, query_raw="회귀",
            api_key="fake-key")
        got = {k: v["mosaic"] for k, v in s.items()
               if isinstance(k, int)}
        if len(got) >= MIN_PERSONS and all(got.values()):
            print(f"OK face {len(got)} tracks all masked")
        else:
            fails += 1
            print(f"FAIL face got={got} (need >={MIN_PERSONS} all True)")
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
