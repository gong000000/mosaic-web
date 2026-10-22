"""개인정보 보호 모자이크 - 전역 설정"""
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDS = [
    "face",           # 얼굴
    "license_plate",  # 차량번호판
    "address",        # 주소 (간판, 문서 등)
    "passenger",      # 차량동승자
    "name_tag",       # 명찰
    "id_card",        # 신분증/문서
    "tattoo",         # 문신
    "phone_screen",   # 휴대폰 화면
]

KO = {
    "face": "얼굴",
    "license_plate": "차량번호판",
    "address": "주소·간판",
    "passenger": "차량동승자",
    "name_tag": "명찰",
    "id_card": "신분증·문서",
    "tattoo": "문신",
    "phone_screen": "휴대폰 화면",
}

# 사람 트랙에 붙는 항목 / 배경 영역으로 처리하는 항목
CATEGORY_TYPE = {
    "face": "person",
    "passenger": "person",
    "name_tag": "person",
    "tattoo": "person",
    "license_plate": "region",
    "address": "region",
    "id_card": "region",
    "phone_screen": "region",
}

# 프롬프트에 나오는 한국어/영어 표현 → 표준 카테고리
CATEGORY_ALIASES = {
    "얼굴": "face", "face": "face",
    "차량번호판": "license_plate", "번호판": "license_plate",
    "차번": "license_plate",
    "license plate": "license_plate", "car plate": "license_plate",
    "주소": "address", "간판": "address", "signboard": "address",
    "현수막": "address", "안내판": "address", "address": "address",
    "동승자": "passenger", "탑승자": "passenger", "passenger": "passenger",
    "명찰": "name_tag", "명패": "name_tag", "name tag": "name_tag",
    "badge": "name_tag", "사원증": "name_tag",
    "신분증": "id_card", "문서": "id_card", "id card": "id_card",
    "identity card": "id_card", "id document": "id_card",
    "주민등록증": "id_card", "운전면허증": "id_card", "여권": "id_card",
    "document": "id_card", "출입증": "id_card",
    "문신": "tattoo", "tattoo": "tattoo",
    "휴대폰": "phone_screen", "핸드폰": "phone_screen", "휴대전화": "phone_screen",
    "phone screen": "phone_screen", "smartphone": "phone_screen",
    "cell phone": "phone_screen", "phone": "phone_screen",
}


def _alias_in(alias, *texts):
    """영문 별칭은 단어 경계로만 찾아 'surface'→face 같은 오검출을 막는다."""
    if any("\uac00" <= ch <= "\ud7a3" for ch in alias):
        return any(alias in t for t in texts)
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")
    return any(pattern.search(t) for t in texts)


def canon_category(value):
    """VLM/월드가 준 category 또는 label을 표준 키로 변환한다."""
    raw = str(value or "").strip()
    low = raw.lower().replace("-", "_")
    if low in CANDS:
        return low
    compact = low.replace(" ", "_")
    if compact in CANDS:
        return compact
    probe = low.replace("_", " ")
    for text, key in CATEGORY_ALIASES.items():
        if _alias_in(text, low, probe):
            return key
    return compact or "other"


def category_type(cat):
    return CATEGORY_TYPE.get(cat, "region")


def categories_in_text(text):
    """문장 안에 등장하는 표준 카테고리 집합. 영문은 단어 경계로 검사."""
    low = str(text or "").lower()
    probe = low.replace("_", " ")
    return {key for word, key in CATEGORY_ALIASES.items()
            if _alias_in(word, low, probe)}


def label_ko(cat):
    return KO.get(cat, cat.replace("_", " "))

PRESETS = {
    "얼굴 + 차량번호판": "얼굴과 차량번호판을 모자이크해줘",
    "얼굴만 가림": "얼굴만 모자이크해줘",
    "차량번호판만 가림": "차량번호판만 모자이크해줘",
    "주소·번호판 가림": "주소와 차량번호판을 모자이크해줘",
    "직접 입력": "",
}

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# VLM 설정 (VLM은 항상 사용. API 키가 없으면 로컬 YOLO-World로 폴백)
VLM_MODEL = "gpt-4o-mini"
