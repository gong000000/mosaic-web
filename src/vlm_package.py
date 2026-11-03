"""VLM 기반 개인정보 탐지 모듈.

판정용 영상 프레임을 VLM에 전송하여 개인정보 항목(얼굴, 차량번호판, 간판,
명찰, 신분증, 휴대폰 화면 등)을 찾는다. VLM은 항상 사용하며 API 키가 필수다.
VLM은 박스/카테고리만 돌려주고, 실제 모자이크 픽셀은 render.py가 그린다.

- detect_with_vlm(): VLM으로 개인정보 항목 탐지 (API 키 필요)
"""
import base64
import json
import math
import threading

import cv2

from .config import (VLM_MODEL, canon_category, categories_in_text,
                     category_type, label_ko)

VLM_MAX_W = 1920
VLM_TIMEOUT = 90
JPEG_Q = 90

# CANDS 키 → VLM에 설명할 영문 이름
TARGET_EN = {
    "face": "human face",
    "license_plate": "vehicle license plate",
    "address": "address signboard or document text",
    "passenger": "passenger inside a vehicle",
    "name_tag": "name tag or employee badge",
    "id_card": "id card or document",
    "phone_screen": "smartphone screen",
}

PERSON_HINT = ("얼굴, 차량번호판, 주소/간판, 명찰, 신분증/문서, "
               "휴대폰 화면, 차량동승자")


class VLMError(RuntimeError):
    """VLM 키/호출/응답 문제. 빈 탐지 결과와 구분하기 위해 예외로 올린다."""


_CLIENTS = {}
_CLIENT_LOCK = threading.Lock()


def prompt_categories(query_raw):
    """프롬프트에 명시된 표준 카테고리 집합."""
    return categories_in_text(query_raw)


def _client(api_key, timeout, base_url=None):
    """API 키/타임아웃/제공자별 OpenAI 클라이언트 캐시.
    Gemini 등 OpenAI 호환 엔드포인트는 base_url만 바꿔서 같은 코드로 쓴다."""
    try:
        from openai import OpenAI
    except Exception:
        return None
    key = (str(api_key), float(timeout), str(base_url or ""))
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            kwargs = {"api_key": str(api_key), "timeout": timeout,
                      "max_retries": 1}
            if base_url:
                kwargs["base_url"] = str(base_url)
            client = OpenAI(**kwargs)
            _CLIENTS[key] = client
    return client


def _jpg_b64(img_bgr, max_w=0, quality=JPEG_Q):
    try:
        h, w = img_bgr.shape[:2]
        if max_w and w > max_w:
            nh = max(1, int(h * max_w / w))
            img_bgr = cv2.resize(img_bgr, (max_w, nh),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(bytes(buf)).decode("ascii")
    except Exception:
        return ""


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "예", "맞음")
    return bool(value)


def _salvage_objects(text):
    """잘리거나 깨진 JSON 배열에서 완전한 객체만 꺼낸다."""
    out = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    out.append(obj)
                start = -1
    return out


def _parse_json_array(content):
    """VLM 응답에서 JSON 배열을 뽑는다. 해석 불가면 None."""
    if content is None:
        return []
    text = str(content).strip()
    if not text:
        return []
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        return []
    if text == "[]":
        return []

    data = None
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        try:
            data = json.loads(text[i:j + 1])
        except Exception:
            data = None
    if data is None:
        try:
            data = json.loads(text)
        except Exception:
            data = None
    if data is None:
        # 출력 한도에서 배열이 잘린 경우 완전한 객체만 살린다.
        salvaged = _salvage_objects(text)
        return salvaged or None
    if isinstance(data, dict):
        if "bbox" in data or "box" in data or "box_2d" in data:
            return [data]
        data = (data.get("items") or data.get("results")
                or data.get("detections") or data.get("objects") or [])
    return data if isinstance(data, list) else None


def _normalize_item(item, w0, h0, scale, query_raw, coord_default="pixel"):
    """VLM 항목 1개를 앱 공통 형식으로 바꾼다. 실패하면 None.

    coord_default: 제공자가 좌표를 생략했고 값이 1보다 클 때의 해석.
                    Gemini는 0~1000 정규화를 자주 쓰므로 "gemini".
    """
    if not isinstance(item, dict):
        return None
    raw_cat = str(item.get("category") or item.get("label_ko")
                  or item.get("label") or "")
    if "tattoo" in raw_cat.lower() or "문신" in raw_cat:
        return None

    box_2d = item.get("box_2d")
    bbox = item.get("bbox") or item.get("box") or item.get("xyxy") or []
    is_box2d = False
    if len(bbox) != 4 and isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
        bbox = box_2d
        is_box2d = True
    if len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    if is_box2d:
        # Gemini box_2d: [ymin, xmin, ymax, xmax]
        vals = [vals[1], vals[0], vals[3], vals[2]]

    coord = str(item.get("coord") or item.get("coordinate") or "").lower()
    mx = max(vals)
    if is_box2d:
        if mx <= 1.001:
            sx, sy = w0, h0
        elif mx <= 1000:
            sx, sy = w0 / 1000.0, h0 / 1000.0
        else:
            sx = sy = scale
    elif "norm" in coord:
        if mx <= 1.001:
            sx, sy = w0, h0
        elif mx <= 1000:
            sx, sy = w0 / 1000.0, h0 / 1000.0
        else:
            sx = sy = scale
    elif "pixel" in coord or "abs" in coord:
        sx = sy = scale
    elif mx <= 1.001:
        sx, sy = w0, h0
    elif mx > 1000:
        sx = sy = scale
    elif coord_default == "gemini":
        # 좌표 표기가 없고 1~1000 사이면 Gemini의 0~1000 정규화로 본다.
        sx, sy = w0 / 1000.0, h0 / 1000.0
    else:
        sx = sy = scale

    x1, y1 = vals[0] * sx, vals[1] * sy
    x2, y2 = vals[2] * sx, vals[3] * sy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w0, int(round(x2))), min(h0, int(round(y2)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    cat = canon_category(raw_cat)
    if cat in ("tattoo", "문신"):
        return None
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


def _build_prompt(phrases, query_raw, include_all, focus):
    hints = [TARGET_EN.get(p, str(p)) for p in (phrases or []) if p]
    hints_str = ", ".join(hints[:10]) if hints else "개인정보 관련 항목"
    lines = []
    if query_raw:
        lines.append(f"사용자 요청: {query_raw}")
    else:
        lines.append("사용자 요청: (없음)")

    if focus == "license_plate":
        lines.append("이 프레임에서 차량 번호판만 찾으세요. "
                     "번호판이 작거나 야간/원거리여도 빠짐없이 찾으세요.")
    elif include_all:
        lines.append("이 영상 프레임에서 개인정보 보호를 위해 가릴 만한 것을 "
                     "모두 찾으세요.")
        lines.append(f"대상 예: {PERSON_HINT}.")
    else:
        lines.append("이 영상 프레임에서 아래 후보에 해당하는 개인정보를 "
                     "모두 찾으세요. 사용자 요청에 없다는 이유로 후보를 "
                     "빠뜨리지 마세요.")

    lines.append(f"참고 후보: {hints_str}")
    if include_all or focus == "license_plate" or "license_plate" in (phrases or []):
        lines.append("차량이 보이면 앞/뒤 번호판을 빠짐없이 찾으세요. "
                     "번호판이 작거나 야간이어도 찾고, 차량 전체가 아니라 "
                     "번호판 영역만 bbox로 표시하세요.")
    lines.append("bbox는 이미지 크기와 무관한 0~1 정규화 좌표 "
                 "[x1,y1,x2,y2]로 답하고 coord는 \"norm\"으로 쓰세요. "
                 "(x1,y1)=좌상단, (x2,y2)=우하단입니다.")
    lines.append("JSON 배열만 출력하세요. 형식: "
                 "[{\"category\":\"face\",\"label_ko\":\"얼굴\","
                 "\"bbox\":[0.1,0.2,0.4,0.5],\"coord\":\"norm\","
                 "\"confidence\":0.95,\"matches_prompt\":true,"
                 "\"region_type\":\"person\"}]")
    lines.append("category는 가능하면 face, license_plate, address, passenger, "
                 "name_tag, id_card, phone_screen 중 하나를 쓰고, 아니면 영문 "
                 "snake_case로 쓰세요.")
    if include_all:
        lines.append("matches_prompt는 사용자가 요청한 대상이면 true, "
                     "추가로 발견한 개인정보면 false입니다.")
    else:
        lines.append("matches_prompt는 사용자가 요청한 대상이면 true, "
                     "후보에만 해당하면 false입니다.")
    lines.append("region_type은 사람에 붙어 있으면 person, 배경 영역이면 "
                 "region입니다.")
    lines.append("찾지 못했으면 []만 출력하세요.")
    return "\n".join(lines)


def detect_with_vlm(frame_bgr, phrases, api_key, model=VLM_MODEL,
                    timeout=VLM_TIMEOUT, query_raw="", include_all=True,
                    base_url=None, focus=None):
    """VLM으로 개인정보 항목을 탐지한다.

    phrases: CANDS 키 목록 (분석 힌트)
    include_all: True면 프롬프트 대상 외 추가 개인정보 후보까지 찾는다.
                 처리 단계(선택 항목만 렌더)에서는 False로 호출한다.
    base_url: OpenAI 호환 엔드포인트(예: Gemini). None이면 OpenAI.
    focus: "license_plate"면 번호판만 집중 탐지한다(차량 크롭 보조 호출).
    returns: [{category,label_ko,xyxy,conf,matches_prompt,region_type,source}]
    raises: VLMError (키 없음/호출 실패/JSON 해석 실패)
    """
    api_key = str(api_key or "").strip()
    if not api_key:
        raise VLMError("VLM API 키가 없습니다. 사이드바에 키를 입력하세요.")
    client = _client(api_key, timeout, base_url)
    if client is None:
        raise VLMError("openai 패키지를 불러오지 못했습니다. "
                       "pip install -r requirements.txt 후 다시 실행하세요.")
    try:
        h0, w0 = frame_bgr.shape[:2]
        if h0 <= 0 or w0 <= 0:
            return []
        b64 = _jpg_b64(frame_bgr, max_w=VLM_MAX_W)
        if not b64:
            raise VLMError("프레임을 JPEG로 변환하지 못했습니다.")
        sent_w = min(w0, VLM_MAX_W)
        scale = w0 / float(sent_w) if sent_w else 1.0
        prompt = _build_prompt(phrases, query_raw, include_all, focus)

        image_url = {"url": f"data:image/jpeg;base64,{b64}"}
        if base_url is None:
            # OpenAI는 detail=high여야 작은 번호판/명찰을 더 잘 본다.
            image_url["detail"] = "high"
        extra = {}
        if base_url and "generativelanguage" in str(base_url):
            # Gemini 3는 내부 추론에 출력 한도를 먼저 쓰므로 추론량을 낮추고
            # 한도를 넉넉히 준다. 추론량이 크면 JSON이 중간에 잘린다.
            extra["reasoning_effort"] = "low"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "당신은 영상 속 개인정보를 탐지하는 전문가입니다. "
                            "JSON만 출력합니다."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url},
                ]}
            ],
            max_tokens=8192,
            temperature=0,
            timeout=timeout,
            **extra,
        )

        content = response.choices[0].message.content
        results = _parse_json_array(content)
        if results is None:
            snippet = str(content or "")[:200].replace("\n", " ")
            raise VLMError("VLM 응답을 JSON으로 해석하지 못했습니다: "
                           f"{snippet}")
        coord_default = ("gemini" if base_url and "generativelanguage" in
                         str(base_url) else "pixel")
        out = []
        for item in results:
            try:
                norm = _normalize_item(item, w0, h0, scale, query_raw,
                                       coord_default=coord_default)
                if norm is not None:
                    out.append(norm)
            except Exception:
                continue
        return out
    except VLMError:
        raise
    except Exception as e:
        raise VLMError(f"VLM 호출 실패: {e}") from e
