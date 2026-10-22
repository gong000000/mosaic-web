"""자유 형식(free-form) person 카테고리 통합 검사.

VLM이 표준 카테고리에 없는 항목(예: green_shirt, region_type=person)을
돌려줄 때도 사람 트랙에 투표되어 모자이크되는지,
그리고 체크하지 않은(선택 해제한) 카테고리는 투표/렌더에 절대 남지 않는지 확인한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\freeform.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"

# face.mp4 첫 프레임 1번 사람 박스 [47, 61, 395, 647] 안쪽에 겹치는 영역
STUB_BOX = [150, 250, 320, 500]


def main():
    if not os.path.exists(FACE):
        print(f"SKIP face.mp4 없음: {FACE}")
        return 0

    import src.pipeline as P
    from src.select import VLMSelector

    class StubSelector(VLMSelector):
        """VLM 대신 자유 형식 person 박스 1개를 돌려준다."""

        def detect_boxes(self, frame_bgr):
            return [{
                "category": "green_shirt",
                "label_ko": "초록 옷",
                "xyxy": list(STUB_BOX),
                "conf": 0.9,
                "matches_prompt": True,
                "region_type": "person",
                "source": "vlm",
            }]

    tmp_dir = tempfile.mkdtemp(prefix="mosaic_freeform_")
    tmp_out = os.path.join(tmp_dir, "out.mp4")

    def run(targets):
        s = P.mosaic_video(FACE, tmp_out, targets=targets, conf=0.20,
                           model_name="yolo11n-seg.pt", imgsz=640,
                           ultra=False, keyframe_stride=5,
                           query_raw="초록 옷 입은 사람", api_key="fake-key")
        if not os.path.exists(tmp_out):
            raise RuntimeError(f"출력 없음: {tmp_out}")
        return s

    real = P.VLMSelector
    P.VLMSelector = StubSelector
    fails = []
    try:
        # 1) green_shirt 를 선택하면 그 사람이 가려져야 한다.
        s1 = run(["green_shirt"])
        on = {tid: v for tid, v in s1.items()
              if isinstance(tid, int) and v.get("mosaic")}
        print("선택 시 stats:", {k: v.get("mosaic") for k, v in s1.items()
                                if isinstance(k, int)})
        if not on:
            fails.append("green_shirt 선택했는데 가려진 트랙 없음")
        for tid, v in on.items():
            if "green_shirt" not in (v.get("hit_targets") or []):
                fails.append(f"tid {tid}: mosaic인데 hit_targets에 green_shirt 없음")
        if not any("green_shirt" in (v.get("cat_on") or {})
                   for v in s1.values() if isinstance(v, dict)):
            fails.append("어느 트랙에도 green_shirt cat_on 없음")

        # 2) face 를 선택하면 green_shirt 는 절대 렌더되지 않아야 한다.
        s2 = run(["face"])
        print("미선택 시 stats:", {k: v.get("mosaic") for k, v in s2.items()
                                  if isinstance(k, int)})
        for tid, v in s2.items():
            if not isinstance(tid, int):
                continue
            cats = set(v.get("cat_on") or {}) | set(v.get("hit_targets") or [])
            if "green_shirt" in cats:
                fails.append(f"tid {tid}: 미선택 green_shirt 가 투표/렌더에 남음")
        face_on = sum(1 for tid, v in s2.items()
                      if isinstance(tid, int) and v.get("mosaic"))
        if face_on != 5:
            fails.append(f"face 선택 시 가려진 사람 수 {face_on} != 5")
    finally:
        P.VLMSelector = real
        for p in (tmp_out, tmp_out + ".noaudio.mp4"):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if fails:
        for f in fails:
            print("FAIL", f)
        print("FREEFORM_FAIL")
        return 1
    print("FREEFORM_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
