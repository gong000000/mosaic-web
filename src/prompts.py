"""한국어 프롬프트 파싱. 외부 전송 없음, 전부 로컬.
LLM 없음. 정규화+유사어 사전+퍼지 매칭으로 오타·붙여쓰기까지 처리.
반환: (target, invert, note)
"""
from .config import KOR2ENG, CANDS

INVERT_KEYS = ["제외", "빼고", "말고", "외에는", "외에", "빼고는", "남기고",
               "안쓴", "없는", "미착용"]

ABSTRACT_WORDS = [
    "수상한", "이상한", "위험한", "불량", "나쁜", "좋은", "예쁜", "멋진",
    "행동", "태도", "분위기", "느낌",
]

DOMAIN_PRESETS = {
    "건설 현장": "안전모 안 쓴 사람 모자이크해줘",
    "병원": "흰옷 입은 사람 제외하고 모두 가려줘",
    "바디캠/공공": "경찰 제복 입은 사람 제외하고 모두 가려줘",
    "크리에이터": "초록 셔츠 입은 사람 모자이크해줘",
}

EXAMPLES = [
    "초록 셔츠 입은 사람 모자이크해줘",
    "빨간 옷 입은 사람만 모자이크해줘",
    "흰옷 입은 사람 제외하고 모두 가려줘",
    "줄무늬 입은 사람 모자이크해줘",
]

GUIDE = (
    "색(초록/빨강/파랑/검정/흰)으로 쓰면 잘 맞는다. "
    "예: '파란 셔츠 입은 사람'. 모자·안경·우산·안전모도 된다. "
    "추상적인 말(수상한, 이상한 행동)은 눈으로 못 구분해서 안 된다."
)

# 타겟별 유사어. 띄어쓰기 없는 형태도 자동 생성해서 붙여쓰기 대응.
_SYN = {
    "green shirt": ["초록", "그린", "녹색", "연두", "청록", "그린셔츠", "초록셔츠"],
    "white sleeveless vest": ["흰", "하양", "하얀", "하얗", "화이트", "흰옷", "흰조끼", "민소매", "백의"],
    "white blazer": ["블레이저", "정장", "자켓", "양복", "흰자켓"],
    "beige plaid shirt": ["체크", "격자", "플래드", "격무늬", "체크셔츠"],
    "striped sweater": ["줄무늬", "스트라이프", "단가라", "줄무늬니트"],
    "red shirt": ["빨강", "빨간", "레드", "빨간옷", "적색"],
    "blue shirt": ["파랑", "파란", "블루", "청색", "남색"],
    "black shirt": ["검정", "까망", "블랙", "흑색", "검은"],
    "dark navy uniform": ["제복", "유니폼", "경찰", "군인", "작업복", "유니폼옷"],
    "light skin": ["백인", "흰피부", "하얀피부", "백피부", "화이트인"],
    "dark skin": ["흑인", "검은피부", "흑피부", "블랙인"],
    "yellow skin": ["황인", "황색피부", "노란피부", "동양인"],
}
# 안전모/헬멧/모자/안경/우산은 색이 아니라 월드 검출로 본다. red로 매핑하지 않는다.


def _norm(s: str) -> str:
    return (s or "").replace(" ", "").replace(" ", "")


def _expand_keywords():
    table = {}
    for target, keys in _SYN.items():
        bag = set()
        for k in keys:
            bag.add(k)
            bag.add(_norm(k))
        table[target] = sorted(bag, key=len, reverse=True)
    # 구 KOR2ENG도 합치기 (호환)
    for k, v in KOR2ENG.items():
        if v in table and k not in table[v]:
            table[v].append(k)
    return table


_SYN_TABLE = _expand_keywords()


def warn_abstract(text: str):
    hits = [w for w in ABSTRACT_WORDS if w in (text or "")]
    if hits:
        return f"추상적인 말({', '.join(hits)})은 영상으로 구분이 안 된다. 색·옷모양으로 바꿔봐라."
    return ""


def _best_target(ntext: str):
    best, best_kw, best_len = None, "", 0
    for target, kws in _SYN_TABLE.items():
        for kw in kws:
            if kw and kw in ntext and len(kw) > best_len:
                best, best_kw, best_len = target, kw, len(kw)
    return best, best_kw


def split_multi(text: str):
    """쉼표/와/과/랑/하고/및으로 나눈다."""
    import re
    parts = [s.strip() for s in re.split(r"[,/·]|와 |과 |랑 |하고 |및 ", text or "") if s.strip()]
    return parts if parts else [text]


def targets_from_text(text: str):
    """문장에서 타겟 목록 추출. [(target, keyword)]"""
    found = []
    for part in split_multi(text):
        best, kw = _best_target(_norm(part))
        if best is not None:
            found.append((best, kw))
    seen, out = set(), []
    for t, k in found:
        if t not in seen:
            seen.add(t)
            out.append((t, k))
    return out


def parse_korean_prompt(text: str):
    """(targets[list], invert, note) 반환."""
    raw = text or ""
    ntext = _norm(raw)
    invert = any(_norm(k) in ntext for k in INVERT_KEYS)
    found = targets_from_text(raw)
    if not found:
        try:
            from .world import open_phrases_for_raw
            ops = open_phrases_for_raw(raw)
        except Exception:
            ops = []
        if ops:
            return ["green shirt"], invert, f"'{','.join(ops)}'는 월드 검출로 본다. 색 기본값(초록)은 무시된다."
        return ["green shirt"], invert, "못 알아들었다. 아래 '가릴 옷'에서 직접 골라라."
    targets = [t for t, _ in found]
    for t in targets:
        if t not in CANDS:
            CANDS.append(t)
    kws = ",".join(k for _, k in found)
    if any(t in ("light skin", "dark skin", "yellow skin") for t, _ in found):
        note = f"'{kws}'를 피부색으로 이해함."
    else:
        note = f"'{kws}'를 {','.join(targets)}로 이해함."
    return targets, invert, note


class PromptParser:
    """지시문 파서. 규칙 사전+매칭 로직을 객체에 묶는다."""

    def __init__(self):
        self.invert_keys = list(INVERT_KEYS)
        self.abstract_words = list(ABSTRACT_WORDS)

    def normalize(self, text):
        return _norm(text)

    def split(self, text):
        return split_multi(text)

    def targets(self, text):
        return targets_from_text(text)

    def warn(self, text):
        hits = [w for w in self.abstract_words if w in (text or "")]
        if hits:
            return f"추상적인 말({', '.join(hits)})은 영상으로 구분이 안 된다. 색·옷모양으로 바꿔봐라."
        return ""

    def parse(self, text):
        return parse_korean_prompt(text)


def make_query(text, targets=None, invert=None, note="", used_llm=False):
    """원문 보존용 쿼리. VLM은 매핑 결과가 아니라 이 raw를 본다."""
    from .select import PromptQuery
    raw = text or ""
    if targets is None or invert is None:
        t, i, n = parse_korean_prompt(raw)
        targets = targets if targets is not None else t
        invert = invert if invert is not None else i
        note = note or n
    if isinstance(targets, str):
        targets = [targets]
    return PromptQuery(raw=raw, targets=list(targets), invert=bool(invert),
                       note=note, used_llm=bool(used_llm))
