"""VLM 기반 판정 인터페이스.
영상 프레임을 VLM에 전송하여 개인정보 항목을 탐지하고, 사람 박스와 매칭한다.
API 키가 없거나 VLM 호출이 실패하면 로컬 YOLO-World로 폴백한다.
"""
from .config import (CATEGORY_TYPE, VLM_MODEL, canon_category, category_type,
                     label_ko)


class VLMSelector:
    """VLM 기반 판정기.

    API 키가 있으면 VLM(OpenAI)이 판단하고, 없거나 실패하면
    로컬 YOLO-World로 폴백한다. 두 결과 모두 사람 박스와 매칭해 사용한다.
    """

    name = "vlm"

    def __init__(self, query_raw="", targets=None, api_key=None,
                 vlm_model=VLM_MODEL, include_all=True):
        self.query_raw = query_raw or ""
        self.targets = [targets] if isinstance(targets, str) else (list(targets) if targets else ["face"])
        self.api_key = str(api_key or "").strip()
        self.vlm_model = vlm_model
        self.include_all = bool(include_all)
        self._vlm_available = bool(self.api_key)

    def detect_boxes(self, frame_bgr):
        """프레임 1장에서 프롬프트 대상 + 추가 개인정보 후보 박스를 모두 얻는다."""
        from .vlm_package import detect_with_vlm, fallback_world

        if self._vlm_available:
            boxes = detect_with_vlm(frame_bgr, self.targets, self.api_key,
                                    model=self.vlm_model,
                                    query_raw=self.query_raw,
                                    include_all=self.include_all)
            if not boxes:
                boxes = fallback_world(frame_bgr, self.targets,
                                       query_raw=self.query_raw,
                                       include_all=self.include_all)
        else:
            boxes = fallback_world(frame_bgr, self.targets,
                                   query_raw=self.query_raw,
                                   include_all=self.include_all)
        return boxes

    def match_items(self, items, world_boxes):
        """이미 얻은 VLM/월드 박스를 사람 아이템에 매칭한다.
        배경(region) 박스는 사람 카테고리 판정에서 제외한다.
        items: [{key, xyxy, ...}]. returns {key: (hit, detail)}.
        """
        out = {}
        if not items:
            return out

        from .world import match_box
        for it in items:
            hits = []
            best = None
            best_iou = 0.0
            for box in world_boxes or []:
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

            # 얼굴 타겟이면 다른 항목이 잡혔더라도 사람 얼굴 후보를 함께 올린다.
            # (seg 모델이 사람을 찾았으면 얼굴도 있다고 보는 보수적 폴백)
            if "face" in self.targets and "face" not in hits:
                hits.append("face")
                if best is None:
                    best = {"xyxy": list(it["xyxy"]), "conf": 1.0,
                            "phrase": "face", "category": "face",
                            "source": "fallback"}
                    best_iou = 1.0

            detail = {
                "selector": ("vlm" if (self._vlm_available and
                                       any(b.get("source") == "vlm"
                                           for b in (world_boxes or [])))
                             else "world"),
                "targets": list(self.targets),
                "hits": hits,
                "world_hit": bool(hits),
                "world_boxes": len(world_boxes or []),
            }
            if best is not None:
                cat = canon_category(best.get("category") or best.get("phrase"))
                detail["world_match"] = {
                    "phrase": best.get("phrase", ""),
                    "category": cat,
                    "label_ko": label_ko(cat),
                    "conf": best.get("conf", 0.5),
                    "iou": best_iou,
                    "source": best.get("source", "world"),
                }
            out[it["key"]] = (bool(hits), detail)

        return out

    def judge_frame(self, frame_bgr, items):
        """frame 1장에 대해 일괄 판정. items: [{key, xyxy}].
        returns {key: (hit: bool, detail: dict)}.
        """
        if not items:
            return {}
        return self.match_items(items, self.detect_boxes(frame_bgr))
