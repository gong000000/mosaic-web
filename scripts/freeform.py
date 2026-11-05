"""자유 형식(free-form) person 카테고리와 명찰 직접 마스킹 검사.

VLM이 표준 카테고리에 없는 항목(예: green_shirt, region_type=person)을
돌려줄 때도 사람 트랙에 투표되어 모자이크되는지,
체크하지 않은(선택 해제한) 카테고리는 투표/렌더에 절대 남지 않는지,
VLM이 찾지 않은 얼굴을 임의로 만들지 않는지 확인한다.
또한 '가리는 범위=얼굴만'이어도 선택한 명찰은 VLM 박스 위치에서
직접 가려지는지 확인한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\freeform.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

FACE = r"C:\Users\gong0000\Downloads\face.mp4"

# face.mp4 첫 프레임 1번 사람 박스 [47, 61, 395, 647] 안쪽에 겹치는 영역
STUB_BOX = [150, 250, 320, 500]
# 사람과 무관한 비교 영역(인코딩 잡음 확인용)
CTRL_BOX = [700, 100, 900, 300]


def _read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"프레임을 읽지 못함: {path} ({idx})")
    return frame


def _region_diff(a, b, box):
    x1, y1, x2, y2 = box
    ca = a[y1:y2, x1:x2].astype(np.int16)
    cb = b[y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(ca - cb).mean())


def _remove_video(path):
    for p in (path, path + ".noaudio.mp4"):
        try:
            os.remove(p)
        except OSError:
            pass


def check_name_tag_direct(tmp_dir):
    """얼굴만 범위에서도 선택한 명찰이 별도 박스로 가려지는지 확인한다."""
    import src.pipeline as P
    from src.select import VLMSelector

    class NameTagSelector(VLMSelector):
        def detect_boxes(self, frame_bgr):
            return [{
                "category": "name_tag",
                "label_ko": "명찰",
                "xyxy": list(STUB_BOX),
                "conf": 0.9,
                "matches_prompt": True,
                "region_type": "person",
                "source": "vlm",
            }]

    paths = {name: os.path.join(tmp_dir, f"tag_{name}.mp4")
             for name in ("selected", "unselected", "base")}

    def run(targets, out):
        return P.mosaic_video(
            FACE, out, targets=targets, conf=0.20,
            model_name="yolo11n-seg.pt", imgsz=640, ultra=False,
            keyframe_stride=5, area_mode="face_only",
            query_raw="명찰", api_key="fake-key")

    real = P.VLMSelector
    P.VLMSelector = NameTagSelector
    try:
        stats = run(["face", "name_tag"], paths["selected"])
        run(["face"], paths["unselected"])
        run(["license_plate"], paths["base"])
        tag = _read_frame(paths["selected"], 30)
        face = _read_frame(paths["unselected"], 30)
    finally:
        P.VLMSelector = real
        for p in paths.values():
            _remove_video(p)

    if not any(v.get("mosaic") for k, v in stats.items()
               if isinstance(k, int)):
        return "명찰 선택 시 가려진 트랙 없음"

    selected_diff = _region_diff(face, tag, STUB_BOX)
    control_diff = _region_diff(face, tag, CTRL_BOX)
    if selected_diff <= 10.0:
        return f"명찰 영역이 가려지지 않음 (diff={selected_diff:.2f})"
    if control_diff >= selected_diff / 3.0:
        return f"명찰 외 영역까지 과도하게 가려짐 (diff={control_diff:.2f})"
    return None


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
        #    Stub VLM은 face 를 돌려주지 않으므로, VLM 이 찾지 않은 얼굴을
        #    임의로 만들지 않는지(가려진 사람 0명)도 함께 확인한다.
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
        if face_on != 0:
            fails.append(f"VLM이 face를 찾지 않았는데 {face_on}명 가려짐")

        # 3) 얼굴만 범위에서도 선택한 명찰은 직접 가려져야 한다.
        P.VLMSelector = VLMSelector
        err = check_name_tag_direct(tmp_dir)
        if err:
            fails.append(err)
    finally:
        P.VLMSelector = real
        _remove_video(tmp_out)
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
