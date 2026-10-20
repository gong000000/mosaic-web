"""개인정보 보호 모자이크 - 지시문 파싱 (개인정보 항목만)
외부 전송 없음, 전부 로컬."""
from .config import CANDS

INVERT_KEYS = ["제외", "빼고", "말고", "외에는", "외에", "빼고는", "남기고",
               "안쓴", "없는", "미탐지"]

ABSTRACT_WORDS = [
    "수상한", "이상한", "위험한", "불량", "나쁜", "좋은", "예쁜", "멋진",
    "행동", "태도", "분위기", "느낌",
]

DOMAIN_PRESETS = {
    "도로 CCTV": "차량번호판과 얼굴을 모자이크해줘",
    "건설 현장": "근로자 얼굴을 모자이크해줘",
    "병원": "환자 얼굴과 진료기록을 모자이크해줘",
    "주차장": "차량번호판과顔を 모자이크해줘",
}

EXAMPLES = [
    "얼굴과 차량번호판을 모자이크해줘",
    "얼굴만 가려줘",
    "주소가 적힌 간판을 가려줘",
    "차량동승자 얼굴을 모자이크해줘",
]

GUIDE = (
    "개인정보 항목을 선택해서 가립니다. "
    "예: '얼굴과 차량번호판을 모자이크해줘'. "
    "추상적인 말(수상한, 이상한 행동)은 눈으로 못 구분해서 안 된다."
)

_SYN = {
    "face": ["얼굴", "얼굴모자이크", "인물", "사람얼굴"],
    "license_plate": ["차량번호", "번호판", "차번호", "차량번호판"],
    "address": ["주소", "간판", "도로명", "건물번호", "주택번호"],
    "passenger": ["동승자", "탑승자", "차안사람", "차량동승자"],
}


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
    return table


_SYN_TABLE = _expand_keywords()


def warn_abstract(text: str):
    hits = [w for w in ABSTRACT_WORDS if w in (text or "")]
    if hits:
        return f"추상적인 말({', '.join(hits)})은 영상으로 구분이 안 된다. 개인정보 항목으로 바꿔봐라."
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
        return ["face"], invert, "못 알아들었다. 아래 '가릴 항목'에서 직접 골라라."
    targets = [t for t, _ in found]
    for t in targets:
        if t not in CANDS:
            CANDS.append(t)
    kws = ",".join(k for _, k in found)
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
            return f"추상적인 말({', '.join(hits)})은 영상으로 구분이 안 된다. 개인정보 항목으로 바꿔봐라."
        return ""

    def parse(self, text):
        return parse_korean_prompt(text)


def make_query(text, targets=None, invert=None, note="", used_llm=False):
    """원문 보존용 쿼리."""
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
