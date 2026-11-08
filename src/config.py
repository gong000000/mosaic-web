"""개인정보 보호 모자이크 - 전역 설정"""
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path=None):
    """프로젝트 루트의 .env를 읽는다. 이미 설정된 환경변수를 덮지 않는다.

    API 키를 코드/채팅에 남기지 않고 로컬 파일로만 쓰기 위한 최소 파서다.
    """
    if os.environ.get("MOSAIC_SKIP_DOTENV"):
        return
    env_path = path or os.path.join(ROOT_DIR, ".env")
    try:
        if not os.path.isfile(env_path):
            return
        # 메모장이 ANSI(cp949)로 저장해도 키 줄은 ASCII라 읽을 수 있게
        # 잘못된 바이트는 무시한다. UTF-8 BOM도 허용한다.
        with open(env_path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # 실제 값이 있는 환경변수는 덮지 않는다. 빈 값은 미설정으로 본다.
                if key and value and not os.environ.get(key):
                    os.environ[key] = value
    except Exception:
        # .env는 편의 기능이다. 읽기 실패가 앱 시작을 막으면 안 된다.
        pass


_load_dotenv()

# VLM이 찾는 개인정보 카테고리.
CANDS = [
    "face",           # 얼굴
    "license_plate",  # 차량번호판
    "address",        # 주소 (간판, 문서 등)
    "passenger",      # 차량동승자
    "name_tag",       # 명찰
    "id_card",        # 신분증/문서
    "phone_screen",   # 휴대폰 화면
]

KO = {
    "face": "얼굴",
    "license_plate": "차량번호판",
    "address": "주소·간판",
    "passenger": "차량동승자",
    "name_tag": "명찰",
    "id_card": "신분증·문서",
    "phone_screen": "휴대폰 화면",
}

# 사람 트랙에 붙는 항목 / 배경 영역으로 처리하는 항목
CATEGORY_TYPE = {
    "face": "person",
    "passenger": "person",
    "name_tag": "person",
    "license_plate": "region",
    "address": "region",
    "id_card": "region",
    "phone_screen": "region",
}

# 프롬프트에 나오는 한국어/영어 표현 → 표준 카테고리.
# 복수형/붙여쓰기/제공자마다 다른 표현도 같은 항목으로 묶는다.
CATEGORY_ALIASES = {
    "얼굴": "face", "얼굴들": "face", "face": "face", "faces": "face",
    "human face": "face", "human faces": "face", "face region": "face",
    "차량번호판": "license_plate", "차량 번호판": "license_plate",
    "자동차번호판": "license_plate", "자동차 번호판": "license_plate",
    "번호판": "license_plate", "번호 판": "license_plate",
    "차번": "license_plate", "차량 번호": "license_plate",
    "license plate": "license_plate", "license plates": "license_plate",
    "license_plates": "license_plate", "licenseplate": "license_plate",
    "number plate": "license_plate", "number plates": "license_plate",
    "numberplate": "license_plate", "car plate": "license_plate",
    "car plates": "license_plate", "vehicle plate": "license_plate",
    "vehicle plates": "license_plate", "plate": "license_plate",
    "plates": "license_plate", "license tag": "license_plate",
    "주소": "address", "주소지": "address", "간판": "address",
    "간판들": "address", "signboard": "address", "signboards": "address",
    "sign": "address", "signs": "address", "signage": "address",
    "street sign": "address", "store sign": "address",
    "building sign": "address", "billboard": "address",
    "현수막": "address", "안내판": "address", "표지판": "address",
    "address": "address", "addresses": "address",
    "동승자": "passenger", "탑승자": "passenger", "승객": "passenger",
    "passenger": "passenger", "passengers": "passenger",
    "car passenger": "passenger", "vehicle occupant": "passenger",
    "명찰": "name_tag", "명패": "name_tag", "이름표": "name_tag",
    "name tag": "name_tag", "name tags": "name_tag", "nametag": "name_tag",
    "name badge": "name_tag", "badge": "name_tag", "badges": "name_tag",
    "employee badge": "name_tag", "id badge": "name_tag",
    "사원증": "name_tag",
    "신분증": "id_card", "신분증명서": "id_card", "문서": "id_card",
    "서류": "id_card", "id card": "id_card", "id cards": "id_card",
    "identity card": "id_card", "identification": "id_card",
    "id document": "id_card", "identity document": "id_card",
    "주민등록증": "id_card", "운전면허증": "id_card", "여권": "id_card",
    "passport": "id_card", "driver license": "id_card",
    "driver's license": "id_card", "document": "id_card",
    "documents": "id_card", "document text": "id_card", "출입증": "id_card",
    "휴대폰": "phone_screen", "핸드폰": "phone_screen",
    "휴대전화": "phone_screen", "휴대폰 화면": "phone_screen",
    "핸드폰 화면": "phone_screen", "phone screen": "phone_screen",
    "smartphone screen": "phone_screen", "mobile phone screen": "phone_screen",
    "cell phone screen": "phone_screen", "smartphone": "phone_screen",
    "cell phone": "phone_screen", "cellphone": "phone_screen",
    "phone": "phone_screen",
}

# 'vehicle', 'other'처럼 개별 정보가 아닌 포괄 표현. label이 더 구체적이면
# label 쪽 카테고리를 우선한다.
GENERIC_CATEGORIES = {
    "other", "object", "item", "thing", "unknown", "vehicle", "car",
    "person", "people", "human", "text", "privacy", "pii",
}


def _alias_in(alias, *texts):
    """영문 별칭은 단어 경계로만 찾아 'surface'→face 같은 오검출을 막는다."""
    if any("\uac00" <= ch <= "\ud7a3" for ch in alias):
        return any(alias in t for t in texts)
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")
    return any(pattern.search(t) for t in texts)


def canon_category(value):
    """VLM이 준 category 또는 label을 표준 키로 변환한다."""
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

# VLM 설정. Gemini 하나만 사용한다. 모델명은 .env의 VLM_MODEL로 바꿀 수 있다.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VLM_MODEL = os.environ.get("VLM_MODEL", "gemini-3-flash-preview")

# 분석 화면의 박스에 그리는 ASCII 코드. OpenCV putText는 한글을 '?????'로
# 그리므로, 이미지에는 코드만 쓰고 한국어 이름은 체크리스트/캡션에 쓴다.
REGION_CODE = {
    "license_plate": "PLATE",
    "address": "SIGN",
    "id_card": "ID",
    "phone_screen": "PHONE",
    "name_tag": "NAME",
}


def region_code(cat):
    return REGION_CODE.get(cat, str(cat or "ITEM").upper()[:8])


def env_api_key():
    """Gemini API 키를 환경변수/.env에서 찾는다. 없으면 빈 문자열."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VLM_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


# VLM이 돌려준 박스가 실제 개인정보 영역으로 보이는지 확인하는 상한.
# 좌표 스케일을 잘못 해석하면 프레임 절반을 덮는 거대한 박스가 생기는데,
# 이런 박스는 모자이크하지 않는 편이 낫다.
_MAX_BOX_AREA = {
    "face": 0.16,
    "license_plate": 0.20,
    "address": 0.22,
    "passenger": 0.35,
    "name_tag": 0.10,
    "id_card": 0.18,
    "phone_screen": 0.14,
}
_MAX_BOX_ASPECT = {
    "face": 1.8,
    "license_plate": 10.0,
    "address": 14.0,
    "passenger": 4.0,
    "name_tag": 10.0,
    "id_card": 8.0,
    "phone_screen": 8.0,
}


def plausible_box(cat, x1, y1, x2, y2, frame_w, frame_h):
    """VLM 박스가 카테고리별로 현실적인 크기/종횡비인지 확인한다."""
    w = max(0, int(x2) - int(x1))
    h = max(0, int(y2) - int(y1))
    fw, fh = max(1, int(frame_w)), max(1, int(frame_h))
    if w < 4 or h < 4:
        return False
    area = (w * h) / float(fw * fh)
    if area > _MAX_BOX_AREA.get(cat, 0.35):
        return False
    aspect = w / float(max(1, h))
    return (1.0 / _MAX_BOX_ASPECT.get(cat, 14.0)
            <= aspect <= _MAX_BOX_ASPECT.get(cat, 14.0))
