"""회귀 테스트. face.mp4 대상별 결과가 기대값과 다르면 실패(exit 1).
실행: .\\.venv\\Scripts\\python.exe scripts\\regression.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
TMP = r"C:\Users\gong0000\AppData\Local\Temp\opencode\regression_tmp.mp4"

# 기대값은 face.mp4 실측 기준. 1번 흰 조끼는 본인 착용이라 True가 정상이다.
EXPECTED = {
    "green shirt": {1: False, 2: False, 3: True, 4: False, 5: False},
    "white sleeveless vest": {1: True, 2: True, 3: False, 4: True, 5: True},
    "light skin": {1: True, 2: True, 3: False, 4: True, 5: True},
    "dark skin": {1: False, 2: False, 3: True, 4: False, 5: False},
    "yellow skin": {1: False, 2: False, 3: False, 4: False, 5: False},
}


def main():
    from src.pipeline import mosaic_video
    if not os.path.exists(FACE):
        print(f"SKIP face.mp4 없음: {FACE}")
        return 0
    fails = 0
    for tgt, exp in EXPECTED.items():
        s = mosaic_video(FACE, TMP, target=[tgt], conf=0.20,
                         model_name="yolo11m-seg.pt", imgsz=640,
                         ultra=False, keyframe_stride=3, query_raw="회귀")
        got = {k: v["mosaic"] for k, v in sorted(s.items())}
        if got == exp:
            print(f"OK {tgt} {got}")
        else:
            fails += 1
            print(f"FAIL {tgt} got={got} exp={exp}")
        try:
            if os.path.exists(TMP):
                os.remove(TMP)
        except Exception:
            pass
    print("REGRESSION_PASS" if fails == 0 else f"REGRESSION_FAIL {fails}건")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
