"""LLM 지시문 해석. 영상 프레임은 절대 보내지 않는다. 텍스트 한 줄만.
API 키 없으면 규칙 기반 폴백. 기본은 로컬 규칙(오프라인).
"""
import json
import os

SYSTEM = (
    "너는 영상 모자이크용 지시문 파서다. "
    "한국어 지시문을 보고 JSON 하나로만 답하라. "
    "키: targets(아래 후보 중 1개 이상 배열), invert(제외/빼고/말고/외에는/남기고 있으면 true), reason(짧게)."
)

_CANDS_HINT = (
    "green shirt, white sleeveless vest, white blazer, beige plaid shirt, "
    "striped sweater, red shirt, blue shirt, black shirt, dark navy uniform, "
    "light skin, dark skin, yellow skin"
)


def _call_openai(prompt_text, api_key, model="gpt-4o-mini", timeout=20):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    msg = (
        f"후보: {_CANDS_HINT}\n"
        f"지시문: {prompt_text}\n"
        '형식: {"targets":["..."],"invert":true/false,"reason":"..."}'
    )
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
        timeout=timeout,
    )
    return json.loads(r.choices[0].message.content)


_CACHE = {}


def _as_list(t, candidates):
    if isinstance(t, list):
        return [x for x in (str(v).strip() for v in t) if x in candidates]
    if isinstance(t, str) and t.strip() in candidates:
        return [t.strip()]
    return []


def parse_with_llm(prompt_text, candidates, api_key=None):
    """(targets[list], invert, note, used_llm) 반환. 실패하면 used_llm=False + 규칙 결과."""
    from .prompts import parse_korean_prompt as rule_parse
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    ck = (prompt_text, tuple(candidates), bool(key), key[-6:] if key else "")
    if ck in _CACHE:
        return _CACHE[ck]
    if not key:
        t, i, n = rule_parse(prompt_text)
        out = (t, i, n + " (로컬 규칙)", False)
    else:
        try:
            d = _call_openai(prompt_text, key)
            tl = _as_list(d.get("targets", d.get("target")), candidates)
            if not tl:
                t0, i0, n0 = rule_parse(prompt_text)
                out = (t0, bool(d.get("invert", i0)), f"LLM 응답 후보 밖. 규칙으로 대체. ({d.get('reason','')})", True)
            else:
                out = (tl, bool(d.get("invert", False)), f"LLM 해석: {d.get('reason','')}", True)
        except Exception as e:
            t, i, n = rule_parse(prompt_text)
            out = (t, i, f"LLM 실패라 규칙으로 대체 ({type(e).__name__})", False)
    _CACHE[ck] = out
    return out
