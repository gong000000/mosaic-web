"""강건성 검사. 엣지 입력에도 죽지 않으면 통과. 전체 처리 없이 빠르게.
VLM 호출은 스텁으로 바꿔 키 없이 검사한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\robustness.py
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACE = r"C:\Users\gong0000\Downloads\face.mp4"
YTDOWN = r"C:\Users\gong0000\Downloads\YTDown.com_YouTube_Media__VuJA-VQRcY_Central-Cee-Doja-Official-Music-Video_002_720p.mp4"
TMP = os.path.join(os.environ.get("TEMP", "."), "mosaic_robust_black.mp4")


def check(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__} {e}")
        return 1
    print(f"OK {name}")
    return 0


@contextlib.contextmanager
def fake_vlm(fn):
    """preview/select가 참조하는 detect_with_vlm을 모두 스텁으로 바꾼다."""
    import src.vlm_package as vp
    import src.preview as pv
    real_vp = vp.detect_with_vlm
    real_pv = pv.detect_with_vlm
    vp.detect_with_vlm = fn
    pv.detect_with_vlm = fn
    try:
        yield
    finally:
        vp.detect_with_vlm = real_vp
        pv.detect_with_vlm = real_pv


def _empty(frame, phrases, api_key, **kwargs):
    return []


def _face_box(frame, phrases, api_key, **kwargs):
    h, w = frame.shape[:2]
    return [{
        "category": "face", "label_ko": "얼굴",
        "xyxy": [int(w * 0.10), int(h * 0.10),
                 int(w * 0.40), int(h * 0.45)],
        "conf": 0.9, "matches_prompt": True,
        "region_type": "person", "source": "vlm",
    }]


def t_preview_no_person():
    import cv2
    from src.models import get_model
    from src.preview import analyze_first_frame_items
    if not os.path.exists(YTDOWN):
        return
    cap = cv2.VideoCapture(YTDOWN)
    ok, fr = cap.read()
    cap.release()
    assert ok
    with fake_vlm(_empty):
        anno, persons, items = analyze_first_frame_items(
            fr, get_model("yolo11n-seg.pt"), 0.20, 480,
            query_raw="초록", targets=["green shirt"], api_key="fake-key")
    assert isinstance(anno, type(fr))
    assert isinstance(persons, list) and isinstance(items, list)


def t_preview_closeup():
    import cv2
    from src.models import get_model
    from src.preview import analyze_first_frame_items
    if not os.path.exists(FACE):
        return
    cap = cv2.VideoCapture(FACE)
    ok, fr = cap.read()
    cap.release()
    assert ok
    with fake_vlm(_face_box):
        anno, persons, items = analyze_first_frame_items(
            fr, get_model("yolo11n-seg.pt"), 0.20, 480,
            query_raw="얼굴", targets=["face"], api_key="fake-key")
    assert isinstance(anno, type(fr)) and isinstance(persons, list)
    assert len(persons) >= 1
    assert all("detail" in p for p in persons)
    # VLM이 찾은 얼굴만 체크리스트에 올라와야 한다.
    assert any(it["category"] == "face" for it in items)


def t_black_video():
    import cv2
    import numpy as np
    from src.models import get_model
    from src.preview import analyze_video
    os.makedirs(os.path.dirname(TMP), exist_ok=True)
    vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                         (100, 100))
    for _ in range(5):
        vw.write(np.zeros((100, 100, 3), np.uint8))
    vw.release()
    with fake_vlm(_empty):
        anno, persons, items = analyze_video(
            TMP, get_model("yolo11n-seg.pt"), 0.20, 480,
            query_raw="아무것도", targets=["face"], api_key="fake-key")
    assert anno is None or anno.shape[2] == 3
    assert persons == []
    assert items == []
    try:
        os.remove(TMP)
    except OSError:
        pass


def t_vlm_parse_and_normalize():
    from src.vlm_package import _normalize_item, _parse_json_array
    # 빈 응답은 '탐지 없음'이 아니라 잘린/거부된 응답일 수 있으므로 None.
    assert _parse_json_array("") is None
    assert _parse_json_array(None) is None
    assert _parse_json_array("```json\n[{\"a\": 1}]\n```") == [{"a": 1}]
    assert _parse_json_array("설명 [1, 2] 뒤") == [1, 2]
    assert _parse_json_array("not json") is None
    # 출력 한도에서 잘린 배열은 완전한 객체만 살린다.
    truncated = ('[{"category":"face","bbox":[0.1,0.2,0.3,0.4]},'
                 '{"category":"license_plate","bbox":[0.5,0.6,0.7,0.8]},'
                 '{"category":"face","bbox":[0.9,')
    salvaged = _parse_json_array(truncated)
    assert isinstance(salvaged, list) and len(salvaged) == 2
    assert salvaged[1]["category"] == "license_plate"

    item = {"category": "license_plate", "bbox": [0.1, 0.1, 0.5, 0.5],
            "coord": "norm", "matches_prompt": "false",
            "region_type": "REGION", "conf": "0.9"}
    norm = _normalize_item(item, 100, 200, 1.0, "번호판")
    assert norm is not None
    assert norm["category"] == "license_plate"
    assert norm["xyxy"] == [10, 20, 50, 100]
    assert norm["matches_prompt"] is False
    assert norm["region_type"] == "region"
    assert abs(norm["conf"] - 0.9) < 1e-6

    # Gemini식 0~1000 정규화 + box_2d([ymin,xmin,ymax,xmax])
    item2 = {"category": "license_plate", "bbox": [100, 100, 500, 500],
             "coord": "norm"}
    assert _normalize_item(item2, 100, 200, 1.0, "")["xyxy"] == \
        [10, 20, 50, 100]
    item3 = {"category": "face", "box_2d": [100, 200, 500, 600]}
    assert _normalize_item(item3, 1000, 1000, 1.0, "",
                           coord_default="gemini")["xyxy"] == \
        [200, 100, 600, 500]

    # 축소 전송본 기준 픽셀 좌표는 scale을 곱한다.
    item4 = {"category": "face", "bbox": [10, 20, 30, 40],
             "coord": "pixel"}
    assert _normalize_item(item4, 100, 200, 2.0, "")["xyxy"] == \
        [20, 40, 60, 80]

    # 제공자가 포괄 category('vehicle')와 구체 label을 같이 줘도
    # 번호판을 놓치지 않아야 한다.
    item5 = {"category": "vehicle", "label": "license plate",
             "bbox": [0.1, 0.1, 0.5, 0.5], "coord": "norm"}
    n5 = _normalize_item(item5, 100, 200, 1.0, "번호판")
    assert n5["category"] == "license_plate", n5
    assert n5["label_ko"] == "차량번호판"
    item6 = {"category": "other", "label": "차량 번호판",
             "bbox": [0.1, 0.1, 0.5, 0.5], "coord": "relative"}
    assert _normalize_item(item6, 100, 200, 1.0, "")["category"] == \
        "license_plate"
    # 복수형/붙여쓰기도 같은 항목으로 묶는다.
    item7 = {"category": "faces", "bbox": [0.2, 0.1, 0.58, 0.42]}
    assert _normalize_item(item7, 100, 200, 1.0, "얼굴")["category"] == "face"

    # 문신은 어떤 응답이 와도 항목으로 만들지 않는다.
    assert _normalize_item({"category": "tattoo",
                            "bbox": [0.1, 0.1, 0.2, 0.2]},
                           100, 100, 1.0, "") is None
    assert _normalize_item({"bbox": [1, 2, 3]}, 100, 100, 1.0, "") is None
    assert _normalize_item({"bbox": ["a", 1, 2, 3]},
                           100, 100, 1.0, "") is None


def _capture_call(frame, phrases, query_raw, include_all=True,
                  base_url=None):
    """detect_with_vlm 1회를 모의 실행하고 실제 요청 payload를 돌려준다."""
    import src.vlm_package as vp

    call = {}

    class _Completions:
        def create(self, **kw):
            call.update(kw)

            class _Msg:
                content = "[]"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    real_client_fn = vp._client

    def _fake_client(api_key, timeout, base_url=None):
        return _FakeClient()

    vp._client = _fake_client
    try:
        out = vp.detect_with_vlm(
            frame, phrases, "k", model="gemini-3-flash-preview",
            query_raw=query_raw, include_all=include_all,
            base_url=base_url)
        assert out == []
    finally:
        vp._client = real_client_fn
    return call


def t_vlm_provider():
    """Gemini 경로: base_url/모델/이미지 전송을 모의 검증."""
    import numpy as np
    import openai
    import src.vlm_package as vp

    captured = {}
    real_openai = openai.OpenAI

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    openai.OpenAI = FakeOpenAI
    try:
        vp._CLIENTS.clear()
        c = vp._client(
            "k", 30,
            "https://generativelanguage.googleapis.com/v1beta/openai/")
        assert c is not None
        assert captured["base_url"].endswith("/openai/")
        assert captured["api_key"] == "k"
        # SDK 자체 재시도를 끄고, 429에는 추가 호출을 하지 않는다.
        assert captured["max_retries"] == 0
    finally:
        openai.OpenAI = real_openai
        vp._CLIENTS.clear()

    call = _capture_call(
        np.zeros((64, 64, 3), np.uint8),
        ["face", "license_plate"],
        "얼굴과 차량번호판만 가려줘",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    assert call["model"] == "gemini-3-flash-preview"
    assert call["max_tokens"] == 8192
    assert call["reasoning_effort"] == "low"
    parts = call["messages"][1]["content"]
    assert any(p.get("type") == "image_url"
               and p["image_url"]["url"].startswith(
                   "data:image/jpeg;base64,")
               for p in parts if isinstance(p, dict))
    assert call["temperature"] == 0
    prompt = parts[0]["text"]
    assert "얼굴과 차량번호판만 가려줘" in prompt
    assert "0~1 정규화 좌표" in prompt
    assert "문신" not in prompt

    # VLMSelector가 base_url을 detect_with_vlm까지 넘기는지
    from src.select import VLMSelector
    seen2 = {}
    real_detect = vp.detect_with_vlm

    def _fake_detect(frame, phrases, api_key, **kw):
        seen2.update(kw)
        return [{"xyxy": [0, 0, 1, 1], "category": "face", "conf": 1.0}]

    vp.detect_with_vlm = _fake_detect
    try:
        sel = VLMSelector(api_key="k", targets=["face"],
                          vlm_base_url="https://example.test/openai/")
        boxes = sel.detect_boxes(np.zeros((32, 32, 3), np.uint8))
        assert boxes
        assert seen2.get("base_url") == "https://example.test/openai/"
    finally:
        vp.detect_with_vlm = real_detect


def t_vlm_requires_key():
    import numpy as np
    from src.vlm_package import VLMError, detect_with_vlm
    try:
        detect_with_vlm(np.zeros((8, 8, 3), np.uint8), ["face"], "")
    except VLMError:
        return
    raise AssertionError("API 키 없이 VLM이 호출됨")


def t_selector_requires_key():
    from src.select import VLMSelector
    try:
        VLMSelector(api_key="")
    except ValueError:
        return
    raise AssertionError("API 키 없이 VLMSelector가 생성됨")


def str_prompt_has_no_tattoo():
    from src.config import CANDS
    from src.vlm_package import TARGET_EN, _build_prompt
    assert "tattoo" not in CANDS
    assert "tattoo" not in TARGET_EN
    text = _build_prompt(CANDS, "얼굴과 번호판", True, None)
    assert "문신" not in text and "tattoo" not in text.lower()
    assert "차량이 보이면 앞/뒤 번호판" in text


def t_quota_no_retry():
    """429/일일 한도는 재시도하지 않고 예산도 원자적으로 막는지 확인."""
    from src.vlm_budget import (VLMBudget, VLMRetryPolicy,
                                classify_vlm_error)
    err = RuntimeError("429 quota exceeded: free tier per day")
    assert classify_vlm_error(err) == "quota_exceeded"
    policy = VLMRetryPolicy(max_retries=3)
    assert policy.should_retry(err, 0) is False
    assert policy.should_retry(RuntimeError("503 unavailable"), 0) is True
    budget = VLMBudget(per_video_limit=2)
    assert budget.reserve() is True
    assert budget.reserve() is True
    assert budget.reserve() is False
    assert budget.remaining() == 0


def t_mosaic_blocks():
    """가로로 긴 번호판도 정사각 블록으로 충분히 가려지는지 확인."""
    import numpy as np
    from src.render import mosaic_block_size, render_mosaic
    w, h = 90, 30
    roi = np.zeros((h, w, 3), np.uint8)
    roi[:, ::2] = (255, 255, 255)
    bs = mosaic_block_size(w, h, 30)
    assert bs >= 4, bs
    assert mosaic_block_size(h, w, 30) == bs
    mos = render_mosaic(roi, "pixel", blocks=30)
    assert mos.shape == roi.shape
    assert np.array_equal(mos[:bs, :bs],
                          np.full((bs, bs, 3), mos[0, 0], np.uint8))
    # 1px 세로 줄무늬가 그대로 남으면 안 된다.
    assert np.count_nonzero(np.diff(mos[0, :, 0])) < w // 2
    assert np.count_nonzero(np.diff(mos[:, 0, 0])) < h // 2


def main():
    fails = 0
    fails += check("preview-0person", t_preview_no_person)
    fails += check("preview-closeup", t_preview_closeup)
    fails += check("black-video", t_black_video)
    fails += check("vlm-parse-normalize", t_vlm_parse_and_normalize)
    fails += check("vlm-provider", t_vlm_provider)
    fails += check("vlm-requires-key", t_vlm_requires_key)
    fails += check("selector-requires-key", t_selector_requires_key)
    fails += check("prompt-no-tattoo", str_prompt_has_no_tattoo)
    fails += check("quota-no-retry", t_quota_no_retry)
    fails += check("mosaic-blocks", t_mosaic_blocks)
    print("ROBUSTNESS_PASS" if fails == 0 else f"ROBUSTNESS_FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
