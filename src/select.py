"""개인정보 보호 판정 인터페이스. VLM(개인정보 탐지)이 중심이다.
- VLMSelector: OpenAI VLM 기반 개인정보 탐지. 첫 프레임 1회 또는 모든 프레임.
- WorldSelector: YOLO-World 폴백. 외부 전송 없음.
- HybridSelector: VLM 우선, 실패 시 월드 폴백.
판정 결과는 (hit: bool, detail: dict). detail은 JSON 직렬화 가능해야 함.
"""
from dataclasses import dataclass, field


@dataclass
class PromptQuery:
    """원문 보존용. VLM은 매핑된 CANDS가 아니라 raw를 본다."""
    raw: str = ""
    targets: list = field(default_factory=lambda: ["face"])
    invert: bool = False
    note: str = ""
    used_vlm: bool = False


class BaseSelector:
    name = "base"

    def judge(self, crop_bgr, mask_crop=None):
        """returns (hit: bool, detail: dict)."""
        raise NotImplementedError


class VLMSelector(BaseSelector):
    """VLM 기반 개인정보 판정. 첫 프레임 1회 또는 모든 프레임 탐지."""

    name = "vlm"
    FIRST_FRAME_ONLY = True

    def __init__(self, query_raw="", targets=None, api_key=None,
                 vlm_model="gpt-4o-mini", frame_stride=1):
        self.query_raw = query_raw or ""
        self.targets = [targets] if isinstance(targets, str) else (list(targets) if targets else ["face"])
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.frame_stride = max(1, int(frame_stride))
        self._calls = 0
        self._last_boxes = []
        self._vlm_available = bool(api_key)

    def judge(self, crop_bgr, mask_crop=None):
        raise NotImplementedError("VLMSelector는 judge_frame을 사용해야 합니다.")

    def select_first_frame(self, frame_bgr, items):
        """첫 프레임에서 VLM으로 개인정보 항목을 탐지한다.
        returns {key: (hit: bool, detail: dict)}
        """
        from .vlm_package import detect_with_vlm, fallback_world
        from .world import match_box

        out = {}
        if not items:
            return out

        self._calls += 1
        world_boxes = []
        if self._vlm_available and self.api_key:
            world_boxes = detect_with_vlm(frame_bgr, self.targets, self.api_key,
                                          model=self.vlm_model)
            if not world_boxes:
                world_boxes = fallback_world(frame_bgr, self.targets)
        else:
            world_boxes = fallback_world(frame_bgr, self.targets)

        self._last_boxes = world_boxes

        for it in items:
            key = it["key"]
            hit, best, iou = match_box(it["xyxy"], world_boxes)

            detail = {
                "selector": "vlm" if self._vlm_available else "world",
                "targets": list(self.targets),
                "hits": [],
                "world_hit": bool(hit),
                "world_boxes": len(world_boxes),
            }
            if best is not None:
                detail["world_match"] = {
                    "phrase": best["phrase"],
                    "conf": best["conf"],
                    "iou": iou,
                    "source": best.get("source", "world")
                }
                if hit:
                    detail["hits"].append(best["phrase"])

            out[key] = (bool(hit), detail)

        return out

    def judge_frame(self, frame_bgr, items, frame_idx=0):
        """frame 단위 판정. frame_stride마다 VLM 호출.
        returns {key: (hit, detail)}
        """
        out = {}
        if not items:
            return out

        self._calls += 1
        world_boxes = []
        if (self._calls - 1) % self.frame_stride == 0:
            if self._vlm_available and self.api_key:
                world_boxes = detect_with_vlm(frame_bgr, self.targets, self.api_key,
                                              model=self.vlm_model)
                if not world_boxes:
                    world_boxes = fallback_world(frame_bgr, self.targets)
            else:
                world_boxes = fallback_world(frame_bgr, self.targets)
            self._last_boxes = world_boxes
        else:
            world_boxes = self._last_boxes

        from .world import match_box
        for it in items:
            key = it["key"]
            hit, best, iou = match_box(it["xyxy"], world_boxes)

            detail = {
                "selector": "vlm" if self._vlm_available else "world",
                "targets": list(self.targets),
                "hits": [],
                "world_hit": bool(hit),
                "world_boxes": len(world_boxes),
            }
            if best is not None:
                detail["world_match"] = {
                    "phrase": best["phrase"],
                    "conf": best["conf"],
                    "iou": iou,
                    "source": best.get("source", "world")
                }
                if hit:
                    detail["hits"].append(best["phrase"])

            out[key] = (bool(hit), detail)

        return out


class WorldSelector(BaseSelector):
    """YOLO-World 기반 판정 (VLM 폴백용)."""

    name = "world"

    def __init__(self, targets=None, query_raw="", conf=0.20, imgsz=640):
        self.targets = [targets] if isinstance(targets, str) else (list(targets) if targets else ["face"])
        self.query_raw = query_raw or ""
        self.conf = conf
        self.imgsz = imgsz
        self._calls = 0
        self._last_boxes = []

    def judge(self, crop_bgr, mask_crop=None):
        raise NotImplementedError("WorldSelector는 judge_frame을 사용해야 합니다.")

    def select_first_frame(self, frame_bgr, items):
        from .vlm_package import fallback_world
        from .world import match_box

        out = {}
        if not items:
            return out

        self._calls += 1
        world_boxes = fallback_world(frame_bgr, self.targets, conf=self.conf, imgsz=self.imgsz)
        self._last_boxes = world_boxes

        for it in items:
            key = it["key"]
            hit, best, iou = match_box(it["xyxy"], world_boxes)

            detail = {
                "selector": "world",
                "targets": list(self.targets),
                "hits": [],
                "world_hit": bool(hit),
                "world_boxes": len(world_boxes),
            }
            if best is not None:
                detail["world_match"] = {
                    "phrase": best["phrase"],
                    "conf": best["conf"],
                    "iou": iou,
                    "source": best.get("source", "world")
                }
                if hit:
                    detail["hits"].append(best["phrase"])

            out[key] = (bool(hit), detail)

        return out

    def judge_frame(self, frame_bgr, items):
        return self.select_first_frame(frame_bgr, items)
