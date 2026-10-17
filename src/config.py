"""프로젝트 전역 설정. UI 문구와 모델 기본값을 한 곳에서 관리."""

CANDS = [
    "green shirt",
    "white sleeveless vest",
    "white blazer",
    "beige plaid shirt",
    "striped sweater",
    "red shirt",
    "blue shirt",
    "black shirt",
    "light skin",
    "dark skin",
    "yellow skin",
]

PRESETS = {
    "직접 입력": ("초록 셔츠 입은 사람 모자이크해줘", None),
    "경찰제복 외 모두 가리기": ("경찰 제복 입은 사람 제외하고 모두 가려줘", "dark navy uniform"),
    "안전모 미착용자만": ("빨간 안전모 안 쓴 사람 모자이크해줘", None),
    "흰옷만 유지": ("흰옷 입은 사람 제외하고 모두 가려줘", "white sleeveless vest"),
}

KOR2ENG = {
    "초록": "green shirt", "그린": "green shirt", "녹색": "green shirt",
    "흰": "white sleeveless vest", "하양": "white sleeveless vest",
    "블레이저": "white blazer", "정장": "white blazer",
    "체크": "beige plaid shirt", "격자": "beige plaid shirt", "플래드": "beige plaid shirt",
    "줄무늬": "striped sweater", "스트라이프": "striped sweater",
    "빨강": "red shirt", "빨간": "red shirt", "레드": "red shirt",
    "파랑": "blue shirt", "파란": "blue shirt", "블루": "blue shirt",
    "검정": "black shirt", "까망": "black shirt", "블랙": "black shirt",
    "제복": "dark navy uniform", "유니폼": "dark navy uniform",
}

OUTPUT_DIR = "outputs"

# VLM 비활성 기본값. 활성화해도 첫 프레임 1회 판정만 사용한다.
USE_VLM = False
VLM_MODEL = ""
VLM_ONLY_FIRST_FRAME = True
