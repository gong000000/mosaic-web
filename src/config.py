"""개인정보 보호 모자이크 - 전역 설정"""
import os

CANDS = [
    "face",           # 얼굴
    "license_plate",  # 차량번호판
    "address",        # 주소 (간판, 문서 등)
    "passenger",      # 차량동승자
]

KO = {
    "face": "얼굴",
    "license_plate": "차량번호판",
    "address": "주소 (간판·문서 등)",
    "passenger": "차량동승자",
}

PRESETS = {
    "직접 입력": ("개인정보가 포함된 영상을 업로드하세요", None),
    "얼굴만 가림": ("얼굴만 모자이크해줘", ["face"]),
    "차량번호판만 가림": ("차량번호판만 모자이크해줘", ["license_plate"]),
    "주소·번호판 가림": ("주소와 차량번호판을 모자이크해줘", ["address", "license_plate"]),
}

OUTPUT_DIR = "outputs"

# VLM 설정
USE_VLM = False
VLM_MODEL = "gpt-4o-mini"
VLM_ONLY_FIRST_FRAME = True
