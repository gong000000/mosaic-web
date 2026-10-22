"""VLM 기반 개인정보 탐지 모듈.
판정용 영상 프레임을 VLM에 전송하여 개인정보 항목(얼굴, 차량번호판, 주소, 차량동승자 등)을
탐지한다. 사용자 프롬프트가 지정한 항목과, 그 밖에 발견되는 개인정보 후보를 함께 돌려준다.
외부 전송은 사용자가 API 키를 입력했을 때만 수행되며, 키가 없으면 로컬 YOLO-World로 폴백한다.

- detect_with_vlm(): VLM으로 개인정보 항목 탐지 (API 키 필요)
- fallback_world(): YOLO-World로 폴백 탐지
"""
import base64
import json
import math
import threading

import cv2

from .config import (CANDS, VLM_MODEL, canon_category, categories_in_text,
                     category_type, label_ko)

VLM_MAX_W = 1280
JPEG_Q = 80

# CANDS 키 → VLM에 설명할 영문 이름
TARGET_EN = {
    "face": "human face",
    "license_plate": "vehicle license plate",
    "address": "address signboard or document text",
    "passenger": "passenger inside a vehicle",
    "name_tag": "name tag or employee badge",
    "id_card": "id card or document",
    "tattoo": "tattoo",
    "phone_screen": "smartphone screen",
}

_CLIENTS = {}
_CLIENT_LOCK = threading.Lock()


def prompt_categories(query_raw):
    """프롬프트에 명시된 표준 카테고리 집합."""
    return categories_in_text(query_raw)


def _client(api_key, timeout):
    """API 키/타임아웃별 OpenAI 클라이언트 캐시. 매 키프레임 새로 만들지 않는다."""
    try:
        from openai import OpenAI
    except Exception:
        return None
    key = (str(api_key), float(timeout))
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(api_key=str(api_key), timeout=timeout, max_retries=1)
            _CLIENTS[key] = client
    return client


def _jpg_b64(img_bgr, max_w=0, quality=JPEG_Q):
    try:
        h, w = img_bgr.shape[:2]
        if max_w and w > max_w:
            nh = max(1, int(h * max_w / w))
            img_bgr = cv2.resize(img_bgr, (max_w, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(bytes(buf)).decode("ascii")
    except Exception:
        return ""


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "예", "맞음")
    return bool(value)


def _parse_json_array(content):
    """VLM 응답에서 JSON 배열만 뽑아낸다. 코드펜스/설명 문장이 섞여도 처리."""
    if not content:
        return []
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        text = text[i:j + 1]
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, dict):
        if "bbox" in data or "box" in data:
            return [data]
        data = (data.get("items") or data.get("results")
                or data.get("detections") or data.get("objects") or [])
    return data if isinstance(data, list) else []


def _normalize_item(item, w0, h0, scale, query_raw):
    """VLM 항목 1개를 앱 공통 형식으로 바꾼다. 실패하면 None."""
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox") or item.get("box") or []
    if len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    if max(vals) <= 1.0:      # 정규화 좌표(0~1)
        vals = [vals[0] * w0, vals[1] * h0, vals[2] * w0, vals[3] * h0]
    else:                     # 축소 전송본 기준 좌표
        vals = [v * scale for v in vals]
    x1, y1, x2, y2 = [int(round(v)) for v in vals]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w0, x2), min(h0, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    cat_raw = item.get("category") or item.get("label_ko") or item.get("label") or ""
    cat = canon_category(cat_raw)
    label = str(item.get("label_ko") or item.get("label") or label_ko(cat))
    try:
        conf = float(item.get("confidence", item.get("conf", 0.5)))
    except (TypeError, ValueError):
        conf = 0.5
    if not math.isfinite(conf):
        conf = 0.5
    prompt_set = prompt_categories(query_raw)
    match = item.get("matches_prompt")
    if match is None:
        match = cat in prompt_set or (label and label.lower() in str(query_raw).lower())
    rtype = item.get("region_type")
    if rtype not in ("person", "region"):
        rtype = category_type(cat)
    return {
        "category": cat,
        "label_ko": label,
        "xyxy": [x1, y1, x2, y2],
        "conf": round(max(0.0, min(1.0, conf)), 3),
        "matches_prompt": _as_bool(match),
        "region_type": rtype,
        "source": "vlm",
    }


def detect_with_vlm(frame_bgr, phrases, api_key, model=VLM_MODEL,
                    timeout=30, query_raw="", include_all=True):
    """VLM으로 개인정보 항목을 탐지한다.

    phrases: CANDS 키 목록 (분석 힌트)
    include_all: True면 프롬프트 대상 외 추가 개인정보 후보까지 찾는다.
                 처리 단계(선택 항목만 렌더)에서는 False로 호출해 호출 비용을 줄인다.
    returns: [{category,label_ko,xyxy,conf,matches_prompt,region_type,source}]
    """
    if not api_key:
        return []
    client = _client(api_key, timeout)
    if client is None:
        return []
    try:
        h0, w0 = frame_bgr.shape[:2]
        b64 = _jpg_b64(frame_bgr, max_w=VLM_MAX_W)
        if not b64:
            return []
        scale = w0 / float(min(w0, VLM_MAX_W)) if w0 else 1.0

        hints = [TARGET_EN.get(p, str(p)) for p in (phrases or []) if p]
        hints_str = ", ".join(hints[:8]) if hints else "개인정보 관련 항목"
        if include_all:
            request = (
                "이 영상 프레임에서 두 종류를 모두 찾으세요.\n"
                "1) 사용자 요청이 지정한 대상(사람, 옷, 물건 등)\n"
                "2) 그 외 개인정보 보호를 위해 가릴 만한 것 "
                "(얼굴, 차량번호판, 주소/간판, 신분증/문서, 명찰, 문신, 휴대폰 화면 등)\n"
            )
            match_help = ("matches_prompt는 사용자가 요청한 대상이면 true, "
                          "추가로 발견한 개인정보면 false입니다.\n")
        else:
            request = (
                "이 영상 프레임에서 사용자 요청과 아래 후보에 해당하는 것만 찾으세요.\n"
            )
            match_help = ("matches_prompt는 사용자가 요청한 대상이면 true, "
                          "후보에만 해당하면 false입니다.\n")
        prompt = (
            f"사용자 요청: {query_raw or '(없음)'}\n"
            + request
            + f"참고 후보: {hints_str}\n"
            "이미지의 픽셀 좌표로 바운딩 박스를 답하세요.\n"
            "JSON 배열만 출력하세요. 형식: "
            "[{\"category\":\"face\",\"label_ko\":\"얼굴\",\"bbox\":[x1,y1,x2,y2],"
            "\"confidence\":0.95,\"matches_prompt\":true,\"region_type\":\"person\"}]\n"
            "category는 가능하면 face, license_plate, address, passenger, name_tag, "
            "id_card, tattoo, phone_screen 중 하나를 쓰고, 아니면 영문 snake_case로 쓰세요.\n"
            + match_help
            + "region_type은 사람에 붙어 있으면 person, 배경 영역이면 region입니다.\n"
            "찾지 못했으면 []만 출력하세요."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "당신은 영상 속 개인정보를 탐지하는 전문가입니다. JSON만 출력합니다."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ],
            max_tokens=800,
            temperature=0,
            timeout=timeout,
        )

        results = _parse_json_array(response.choices[0].message.content)
        out = []
        for item in results:
            try:
                norm = _normalize_item(item, w0, h0, scale, query_raw)
                if norm is not None:
                    out.append(norm)
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"VLM 탐지 실패: {e}")
        return []


def fallback_world(frame_bgr, phrases, conf=0.20, imgsz=640, query_raw="",
                   include_all=True):
    """YOLO-World 폴백. include_all이면 알려진 개인정보 카테고리까지 함께 찾는다.
    처리 단계에서는 선택된 카테고리만 검색해 속도를 아낀다.
    """
    from .world import category_for_phrase, get_world, phrases_for_categories

    prompt_set = prompt_categories(query_raw)
    cats = []
    for c in list(phrases or []) + list(prompt_set):
        if c and c not in cats:
            cats.append(c)
    if include_all:
        for c in CANDS:
            if c not in cats:
                cats.append(c)

    # 카테고리마다 첫 문구를 먼저 넣어 10개 제한에서도 종류가 골고루 잡히게 한다.
    english = phrases_for_categories(cats, limit=10)
    if not english:
        return []

    boxes = get_world().ground(frame_bgr, english, conf=conf, imgsz=imgsz)
    out = []
    for b in boxes:
        cat = category_for_phrase(b.get("phrase")) or canon_category(b.get("phrase"))
        out.append({
            "category": cat,
            "label_ko": label_ko(cat),
            "xyxy": b["xyxy"],
            "conf": b.get("conf", 0.5),
            "matches_prompt": cat in prompt_set,
            "region_type": category_type(cat),
            "source": "world",
        })
    return out
