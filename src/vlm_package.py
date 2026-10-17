"""VLM용 첫 프레임 포장. 전송은 안 한다. 포장만.

- 입력: preview.analyze_first_frame()이 돌려주는 (anno_bgr, persons).
- 출력: JSON에 바로 넣을 수 있는 dict. JPEG→base64 문자열만 담는다.
- 원칙: 첫 프레임 1회만. 이 함수는 네트워크 호출을 절대 안 한다.
  나중에 VLM 붙일 때 이 dict 그대로 보내면 된다.
"""
import base64

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
