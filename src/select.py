"""VLM 기반 판정 인터페이스.

영상 프레임을 VLM에 전송해 개인정보 항목을 찾고, VLM이 찾은 사람 부착
항목(얼굴·동승자·명찰 등)을 사람 박스와 매칭한다. VLM은 필수이며 로컬
검출로 대신하지 않는다. VLM 박스가 없으면 가릴 항목도 없다.
"""
from .config import (CATEGORY_TYPE, VLM_MODEL, canon_category, category_type,
                     label_ko)


class VLMSelector:
    """VLM 기반 판정기. API 키가 없으면 생성할 수 없다."""

    name = "vlm"

    def __init__(self, query_raw="", targets=None, api_key=None,
                 vlm_model=VLM_MODEL, include_all=True, vlm_base_url=None):
        self.api_key = str(api_key or "").strip()
        if not self.api_key:
            raise ValueError("VLM API 키가 필요합니다.")
        self.query_raw = query_raw or ""
        self.targets = ([targets] if isinstance(targets, str)
                        else (list(targets) if targets else ["face"]))
        self.vlm_model = vlm_model
        self.vlm_base_url = str(vlm_base_url or "").strip() or None
        self.include_all = bool(include_all)

    def detect_boxes(self, frame_bgr):
        """프레임 1장에서 프롬프트 대상 + 추가 개인정보 후보 박스를 얻는다.
        VLM 호출 실패는 VLMError로 올라가며 빈 결과로 숨기지 않는다."""
        from .vlm_package import detect_with_vlm

        return detect_with_vlm(
            frame_bgr, self.targets, self.api_key,
            model=self.vlm_model,
            query_raw=self.query_raw,
            include_all=self.include_all,
            base_url=self.vlm_base_url,
        )

    def match_items(self, items, vlm_boxes):
        """이미 얻은 VLM 박스를 사람 아이템에 매칭한다.
        배경(region) 박스는 사람 카테고리 판정에서 제외한다.
        items: [{key, xyxy, ...}]. returns {key: (hit, detail)}.
        """
        out = {}
        if not items:
            return out

        from .detect import match_box
        for it in items:
            hits = []
            best = None
            best_iou = 0.0
            for box in vlm_boxes or []:
                cat = canon_category(box.get("category") or box.get("phrase"))
                rtype = box.get("region_type")
                if cat in CATEGORY_TYPE:
                    rtype = CATEGORY_TYPE[cat]     # 표준 카테고리는 설정이 우선
                elif rtype not in ("person", "region"):
                    rtype = category_type(cat)
                if rtype != "person":
                    continue
                hit, matched, iou = match_box(it["xyxy"], [box])
                if not hit:
                    continue
                if cat not in hits:
                    hits.append(cat)
                if best is None or (matched.get("conf", 0.5)
                                    > best.get("conf", 0.5)):
                    best = matched
                    best_iou = iou

            detail = {
                "selector": "vlm",
                "targets": list(self.targets),
                "hits": hits,
                "vlm_hit": bool(hits),
                "vlm_boxes": len(vlm_boxes or []),
            }
            if best is not None:
                cat = canon_category(best.get("category") or best.get("phrase"))
                detail["vlm_match"] = {
                    "phrase": best.get("phrase", ""),
                    "category": cat,
                    "label_ko": label_ko(cat),
                    "conf": best.get("conf", 0.5),
                    "iou": best_iou,
                    "source": "vlm",
                }
            out[it["key"]] = (bool(hits), detail)

        return out
