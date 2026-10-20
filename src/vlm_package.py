"""VLM 기반 개인정보 탐지 모듈.
첫 프레임을 VLM에 전송하여 개인정보 항목(얼굴, 차량번호판, 주소, 차량동승자 등)을 탐지한다.
외부 전송은 사용자 동의(API 키 입력) 시에만 수행된다.

- build_first_frame_package(): 첫 화면 포장 (전송 없음, 로컬만)
- detect_with_vlm(): VLM으로 개인정보 항목 탐지 (API 키 필요)
- fallback_world(): YOLO-World로 폴백 탐지
"""
import base64
import json
import os

import cv2

MAX_CROP_W = 320
JPEG_Q = 80


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


def build_first_frame_package(anno_bgr, persons, query_raw=""):
    """첫 화면 1장을 VLM 입력 형태로 묶는다. 전송 없음, 로컬 포장만."""
    items = []
    for p in persons or []:
        try:
            crop = p.get("crop")
            items.append({
                "bi": int(p.get("bi", len(items))),
                "bbox": [int(p.get("x1", 0)), int(p.get("y1", 0)),
                         int(p.get("x2", 0)), int(p.get("y2", 0))],
                "crop_jpg_b64": _jpg_b64(crop, max_w=MAX_CROP_W) if crop is not None else "",
                "heuristic_top1": str(p.get("top1", "-")),
                "heuristic_hits": [str(h) for h in p.get("hits", [])],
            })
        except Exception:
            continue
    return {
        "first_frame_only": True,
        "query_raw": query_raw or "",
        "annotated_jpg_b64": _jpg_b64(anno_bgr, max_w=1280),
        "persons": items,
        "note": "로컬 전용 패키지. 외부 전송을 하지 않는다.",
    }


def detect_with_vlm(frame_bgr, phrases, api_key, model="gpt-4o-mini", timeout=30):
    """VLM으로 개인정보 항목을 탐지한다.
    
    returns: [{xyxy:[x1,y1,x2,y2], conf, phrase, source:"vlm"}]
    """
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        b64 = _jpg_b64(frame_bgr, max_w=1280)
        if not b64:
            return []

        phrases_str = ", ".join(phrases)
        prompt = (
            f"이 이미지에서 다음 개인정보 항목들을 찾아주세요: {phrases_str}\n"
            "각 항목에 대해 바운딩 박스 [x1, y1, x2, y2]와 신뢰도(0-1)를 JSON 배열로 반환하세요.\n"
            "형식: [{\"phrase\": \"항목명\", \"bbox\": [x1, y1, x2, y2], \"confidence\": 0.95}]\n"
            "탐지되지 않은 항목은 포함하지 마세요."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 영상 속 개인정보를 탐지하는 전문가입니다."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ],
            max_tokens=500,
            timeout=timeout,
        )

        content = response.choices[0].message.content
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            results = []

        out = []
        for item in results:
            try:
                bbox = item.get("bbox", [])
                phrase = item.get("phrase", "")
                conf = float(item.get("confidence", 0.5))
                if len(bbox) == 4:
                    out.append({
                        "xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "conf": round(conf, 3),
                        "phrase": str(phrase),
                        "source": "vlm"
                    })
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"VLM 탐지 실패: {e}")
        return []


def fallback_world(frame_bgr, phrases, conf=0.20, imgsz=640):
    """YOLO-World로 폴백 탐지."""
    from .world import get_world
    return get_world().ground(frame_bgr, phrases, conf=conf, imgsz=imgsz)
