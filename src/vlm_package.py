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
import time

import cv2

from .config import (CANDS, GENERIC_CATEGORIES, VLM_MODEL, canon_category,
                     categories_in_text, category_type, label_ko,
                     plausible_box)
from .vlm_budget import VLMRetryPolicy, classify_vlm_error

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
                      # SDK 자체 재시도를 끈다. 429/인증 오류에 호출을 더
                      # 태우지 않고, 5xx만 detect_with_vlm이 직접 재시도한다.
                      "max_retries": 0}
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


def _resolve_category(item):
    """category와 label을 함께 보고 가장 구체적인 표준 카테고리를 고른다.

    제공자마다 ``{"category":"vehicle","label":"license plate"}``처럼
    포괄 category와 구체 label을 같이 주기도 한다. category만 믿으면
    번호판을 놓치므로, 표준 카테고리로 해석되는 label을 우선한다.
    """
    raw_values = [
        item.get("category"), item.get("label_ko"),
        item.get("label"), item.get("phrase"), item.get("name"),
    ]
    values = []
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    if not values:
        return "", "other"

    def _canon(text):
        return canon_category(text)

    cats = [_canon(text) for text in values]
    # 표준 키가 하나라도 있으면 그중 첫 번째를 쓴다. 단, category가
    # 포괄 표현('vehicle' 등)이고 label이 표준 키라면 label을 쓴다.
    for text, cat in zip(values, cats):
        if cat in CANDS and cat not in GENERIC_CATEGORIES:
            return text, cat
    for text, cat in zip(values, cats):
        if cat in CANDS:
            return text, cat
    # 자유 형식은 category를 우선하되, category가 비어 있으면 label을 쓴다.
    return values[0], cats[0]


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


def _content_text(content):
    """OpenAI 호환 응답 content를 문자열로 만든다.

    일부 제공자는 content를 [{"type":"text","text":"..."}] 형태로 준다.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _parse_json_array(content):
    """VLM 응답에서 JSON 배열을 뽑는다. 해석 불가면 None.

    빈 응답/None은 '탐지 없음'이 아니라 잘린/거부된 응답일 수 있으므로
    None을 돌려 호출부가 VLMError로 알리게 한다.
    """
    text = _content_text(content).strip()
    if not text:
        return None
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
    _, cat = _resolve_category(item)
    probe_cat = " ".join(str(item.get(k) or "") for k in
                         ("category", "label_ko", "label", "phrase"))
    if "tattoo" in probe_cat.lower() or "문신" in probe_cat:
        return None

    # Gemini는 box_2d([ymin,xmin,ymax,xmax], 0~1000)를 공식 형식으로
    # 쓰는 경우가 많다. bbox와 함께 와도 box_2d를 우선해야 스케일을
    # 잘못 해석해 화면 절반을 덮는 박스가 생기지 않는다.
    box_2d = item.get("box_2d")
    bbox = item.get("bbox") or item.get("box") or item.get("xyxy") or []
    is_box2d = False
    if isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
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

    coord = " ".join(str(item.get(k) or "") for k in
                     ("coord", "coordinate", "coordinates")).lower()
    mx = max(vals)
    normalized_label = any(k in coord for k in
                           ("norm", "relative", "ratio", "0-1", "0~1",
                            "0 to 1", "0..1"))
    norm1000_label = any(k in coord for k in ("1000", "0-1000", "0~1000"))
    pixel_label = any(k in coord for k in ("pixel", "abs", "image", "px"))
    if is_box2d or norm1000_label:
        if mx <= 1.001:
            sx, sy = w0, h0
        elif mx <= 1000:
            sx, sy = w0 / 1000.0, h0 / 1000.0
        else:
            sx = sy = scale
    elif normalized_label:
        if mx <= 1.001:
            sx, sy = w0, h0
        elif mx <= 1000:
            sx, sy = w0 / 1000.0, h0 / 1000.0
        else:
            sx = sy = scale
    elif pixel_label:
        sx = sy = scale
    elif mx <= 1.001:
        sx, sy = w0, h0
    elif mx > 1000:
        sx = sy = scale
    elif coord_default == "gemini":
        # 1~1000 사이인데 coord 표기가 없다. 0~1000 정규화로 보면
        # 프레임의 12% 이상을 덮고, 픽셀로 보면 작은 박스라면 픽셀로
        # 해석한다(거대한 오탐 박스 방지). 그 외에는 Gemini 기본값인
        # 0~1000 정규화를 따른다.
        norm_area = ((vals[2] - vals[0]) / 1000.0
                     * (vals[3] - vals[1]) / 1000.0)
        px_area = ((vals[2] - vals[0]) / max(1.0, w0)
                   * (vals[3] - vals[1]) / max(1.0, h0))
        if norm_area > 0.12 and px_area < 0.06:
            sx = sy = scale
        else:
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
    if not plausible_box(cat, x1, y1, x2, y2, w0, h0):
        # 좌표 스케일 오류로 생긴 거대 박스/찌그러진 박스는 버린다.
        return None

    if cat in ("tattoo", "문신"):
        return None
    if cat in CANDS:
        # 표준 항목은 UI 표기를 통일한다. VLM이 영어 label을 줘도 한국어로.
        label = label_ko(cat)
        rtype = category_type(cat)
    else:
        label = str(item.get("label_ko") or item.get("label")
                    or item.get("phrase") or label_ko(cat))
        rtype = item.get("region_type")
        if rtype not in ("person", "region"):
            rtype = category_type(cat)
    try:
        conf = float(item.get("confidence", item.get("conf", 0.5)))
    except (TypeError, ValueError):
        conf = 0.5
    if not math.isfinite(conf):
        conf = 0.5
    prompt_set = prompt_categories(query_raw)
    match = item.get("matches_prompt")
    if match is None:
        label_probe = " ".join(str(item.get(k) or "") for k in
                               ("category", "label_ko", "label", "phrase"))
        match = (cat in prompt_set
                 or canon_category(label_probe) in prompt_set
                 or bool(label and label.lower() in str(query_raw).lower()))
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
    hints = []
    for p in (phrases or []):
        if not p:
            continue
        hints.append(TARGET_EN.get(p, str(p).replace("_", " ")))
    hints_str = ", ".join(hints[:12]) if hints else "개인정보 관련 항목"
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
        lines.append("얼굴이 작거나 일부만 보여도, 번호판이 흐리거나 "
                     "원거리여도 찾으세요. 신분증·문서·휴대폰 화면처럼 "
                     "개인정보가 보이는 물체도 포함하세요.")
    else:
        lines.append("이 프레임에서 아래 후보에 해당하는 개인정보만 찾으세요. "
                     "후보에 없는 항목은 결과에 넣지 마세요.")
        lines.append("후보가 여러 개면 하나도 빠뜨리지 말고 각각 찾으세요.")

    lines.append(f"참고 후보: {hints_str}")
    lines.append("같은 종류가 여러 개 있으면 하나로 합치지 말고 각각 별도 "
                 "항목으로 bbox를 주세요.")
    if include_all or focus == "license_plate" or "license_plate" in (phrases or []):
        lines.append("차량이 보이면 앞/뒤 번호판을 빠짐없이 찾으세요. "
                     "승용차·승합차·화물차·오토바이 번호판을 모두 포함하고, "
                     "번호판이 작거나 야간이어도 찾으세요. 차량 전체가 아니라 "
                     "번호판 영역만 bbox로 표시하세요.")
    if focus != "license_plate":
        lines.append("얼굴 bbox는 이마만이 아니라 눈·코·입을 모두 포함한 "
                     "얼굴 전체를 감싸세요. 사람 몸 전체를 얼굴로 표시하지 "
                     "마세요. 명찰·신분증·간판은 글자가 읽히는 영역 전체를 "
                     "bbox로 표시하세요.")
    lines.append("좌표는 Gemini box_2d 형식 [ymin,xmin,ymax,xmax]의 0~1000 "
                 "정수로 답하세요. bbox를 쓸 경우에는 이미지 크기와 무관한 "
                 "0~1 정규화 좌표 [x1,y1,x2,y2]로 답하고 coord는 \"norm\"으로 "
                 "쓰세요. (box_2d/bbox 모두 좌상단이 시작점입니다.)")
    lines.append("JSON 배열만 출력하세요. 형식: "
                 "[{\"category\":\"face\",\"label_ko\":\"얼굴\","
                 "\"box_2d\":[200,100,500,400],\"confidence\":0.95,"
                 "\"matches_prompt\":true,\"region_type\":\"person\"}]")
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
                    base_url=None, focus=None,
                    max_retries: int = 3):
    """VLM으로 개인정보 항목을 탐지한다.

    phrases: CANDS 키 목록 (분석 힌트)
    include_all: True면 프롬프트 대상 외 추가 개인정보 후보까지 찾는다.
                 처리 단계(선택 항목만 렌더)에서는 False로 호출한다.
    base_url: OpenAI 호환 엔드포인트(현재는 Gemini 고정).
    focus: "license_plate"면 번호판만 집중 탐지한다(차량 크롭 보조 호출).
    max_retries: 서버 오류(5xx) 발생 시 최대 재시도 횟수.
                 한도 초과(429)나 인증 오류(401/403)는 재시도하지 않음.
    returns: [{category,label_ko,xyxy,conf,matches_prompt,region_type,source}]
    raises: VLMError (키 없음/호출 실패/JSON 해석 실패/한도 초과)
    """
    api_key = str(api_key or "").strip()
    if not api_key:
        raise VLMError("Gemini API 키가 없습니다. .env의 GEMINI_API_KEY 또는 "
                       "사이드바 입력을 확인하세요.")

    retry = VLMRetryPolicy(max_retries=max_retries)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            client = _client(api_key, timeout, base_url)
            if client is None:
                raise VLMError("openai 패키지를 불러오지 못했습니다. "
                               "pip install -r requirements.txt 후 다시 실행하세요.")
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

            try:
                choice = response.choices[0]
                content = choice.message.content
            except Exception as e:
                raise VLMError(f"VLM 응답 형식이 올바르지 않습니다: {e}") from e
            results = _parse_json_array(content)
            if results is None:
                snippet = _content_text(content)[:200].replace("\n", " ")
                finish = str(getattr(choice, "finish_reason", "") or "")
                extra_msg = f" (finish_reason={finish})" if finish else ""
                if not snippet:
                    raise VLMError("VLM이 빈 응답을 돌려줬습니다." + extra_msg)
                raise VLMError("VLM 응답을 JSON으로 해석하지 못했습니다: "
                               f"{snippet}{extra_msg}")
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

        except VLMError as e:
            last_error = e
            category = classify_vlm_error(e)
            # 한도 초과나 인증 오류는 재시도해도 소용없음.
            if category in ("quota_exceeded", "auth_error",
                            "model_not_found"):
                raise
            # 5xx 서버 오류 또는 파싱 오류는 재시도 대상.
            if not retry.should_retry(e, attempt):
                raise
            if attempt < max_retries:
                delay = retry.delay(attempt)
                time.sleep(delay)
                continue
            raise

        except Exception as e:
            last_error = e
            category = classify_vlm_error(e)
            if category in ("quota_exceeded", "auth_error",
                            "model_not_found"):
                raise VLMError(f"VLM 호출 실패: {e}") from e
            if not retry.should_retry(e, attempt):
                raise VLMError(f"VLM 호출 실패: {e}") from e
            if attempt < max_retries:
                delay = retry.delay(attempt)
                time.sleep(delay)
                continue
            raise VLMError(f"VLM 호출 실패(재시도 {max_retries}회 후): {e}") from e

    # 도달할 수 없지만, 타입 체커를 위해 예외 발생.
    raise last_error or VLMError("VLM 호출에 실패했습니다.")
