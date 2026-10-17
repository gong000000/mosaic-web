"""판정 인터페이스. 휴리스틱과 월드 검출이 같은 자리에서 교체되도록 설계.

- HeuristicSelector: HSV+직접지정(ref) 판정. 파이프라인·스트리밍·미리보기 공용.
- HybridSelector: 휴리스틱+월드. 색·피부는 HSV, 안경·우산·안전모·모자는 월드.
  자동 모드에서는 열린 말이 있을 때만 월드를 사용한다(키프레임 1회).
- VLMSelector: API형 VLM용 스텁.
- WorldStore/문구 매핑은 src/world.py. 로컬 실행, 외부 전송 없음.

판정 결과는 (hit, detail). detail은 JSON 직렬화 가능한 float/str/list만.
"""
from dataclasses import dataclass, field


@dataclass
class PromptQuery:
    """원문 보존용. VLM은 매핑된 CANDS가 아니라 raw가 필요하다."""
    raw: str = ""
    targets: list = field(default_factory=lambda: ["green shirt"])
    invert: bool = False
    note: str = ""
    used_llm: bool = False


class BaseSelector:
    name = "base"

    def judge(self, crop_bgr, mask_crop=None):
        """returns (hit: bool, detail: dict). hit=True면 '가릴 후보'."""
        raise NotImplementedError


class HeuristicSelector(BaseSelector):
    """현재 HSV 판정기. ref_bank 있으면 직접지정 경쟁 매칭, 없으면 HSV 점수."""

    name = "heuristic"

    def __init__(self, targets, margin=0.04, ref_bank=None, ref_thresh=0.82,
                 ref_margin=0.05, classifier=None):
        self.targets = [targets] if isinstance(targets, str) else [t for t in targets]
        self.margin = margin
        self.ref_bank = ref_bank
        self.ref_thresh = ref_thresh
        self.ref_margin = ref_margin
        if classifier is None:
            from .clothing import ClothingClassifier as _CC
            classifier = _CC(margin=margin)
        self.classifier = classifier

    @property
    def use_ref(self):
        return self.ref_bank is not None and len(self.ref_bank) > 0

    def judge(self, crop_bgr, mask_crop=None):
        if self.use_ref:
            return self._judge_ref(crop_bgr, mask_crop)
        return self._judge_hsv(crop_bgr, mask_crop)

    def _judge_hsv(self, crop_bgr, mask_crop=None):
        from .clothing import person_scores
        if not self.targets:
            return False, {"selector": "heuristic", "hits": [], "top": []}
        try:
            sc = person_scores(crop_bgr, mask_crop)
        except Exception:
            return False, {"selector": "heuristic", "hits": [], "error": "scores_fail"}
        hits = []
        for t in self.targets:
            try:
                ok, _ = self.classifier.decide(sc, t)
            except Exception:
                ok = False
            if ok:
                hits.append(t)
        top = sorted(sc.items(), key=lambda x: -x[1])[:3]
        detail = {
            "selector": "heuristic",
            "targets": list(self.targets),
            "hits": hits,
            "top": [(k, round(float(v), 3)) for k, v in top],
        }
        return len(hits) > 0, detail

    def _judge_ref(self, crop_bgr, mask_crop=None):
        import numpy as np
        from .clothing import torso_roi, hsv_hist, hist_corr
        try:
            roi, m2 = torso_roi(crop_bgr, mask_crop)
            if m2 is None:
                m2 = np.ones(roi.shape[:2], np.uint8) * 255
            hh = hsv_hist(roi, m2)
            corrs = [hist_corr(hh, rh) for rh, _ in self.ref_bank]
            order = sorted(range(len(corrs)), key=lambda k: -corrs[k])
            best = float(corrs[order[0]])
            second = float(corrs[order[1]]) if len(corrs) > 1 else -2.0
            _, is_tgt = self.ref_bank[order[0]]
            hit = bool(is_tgt and best >= self.ref_thresh and (best - second) >= self.ref_margin)
            detail = {
                "selector": "heuristic-ref",
                "targets": list(self.targets),
                "hits": [],
                "best": round(best, 3),
                "second": round(second, 3),
                "matched": int(order[0]),
                "matched_is_tgt": bool(is_tgt),
            }
            return hit, detail
        except Exception as e:
            return False, {"selector": "heuristic-ref", "hits": [], "error": type(e).__name__}


class VLMSelector(BaseSelector):
    """API형 VLM용 스텁.

    입력: vlm_package.build_first_frame_package() 결과 1건 (첫 프레임만).
    출력: {bi(int): bool} 번호별 가려짐 여부. 이후 프레임 추적은 ByteTrack이 이어받는다.
    """

    name = "vlm"
    FIRST_FRAME_ONLY = True

    def __init__(self, query_raw="", model=""):
        self.query_raw = query_raw
        self.model = model

    def judge(self, crop_bgr, mask_crop=None):
        raise NotImplementedError("VLMSelector is not implemented.")

    def select_first_frame(self, package):
        raise NotImplementedError("VLMSelector is not implemented.")


class HybridSelector(BaseSelector):
    """휴리스틱+월드 하이브리드. 직접지정(ref)이 있으면 ref만 본다(사용자 선택 우선).
    world_mode: auto(열린 말 있을 때만 월드) / heuristic(색 규칙만) / world(월드 우선+휴리스틱 폴백).
    프레임 단위 judge_frame()이 본체. judge()는 한 장 호환용."""

    name = "hybrid"

    def __init__(self, targets, query_raw="", margin=0.04, ref_bank=None,
                 ref_thresh=0.82, ref_margin=0.05, classifier=None,
                 world_mode="auto", world_conf=None, world_imgsz=None, world_stride=1):
        from .world import WORLD_CONF, WORLD_IMGSZ, needs_world, phrases_for_query
        self.targets = [targets] if isinstance(targets, str) else [t for t in targets]
        self.query_raw = query_raw or ""
        self.world_mode = world_mode if world_mode in ("auto", "heuristic", "world") else "auto"
        self.world_conf = WORLD_CONF if world_conf is None else world_conf
        self.world_imgsz = WORLD_IMGSZ if world_imgsz is None else world_imgsz
        self.world_stride = max(1, int(world_stride))
        h_targets = self._heuristic_targets(list(self.targets), self.query_raw,
                                              ref_bank, self.world_mode)
        self.heuristic = HeuristicSelector(h_targets, margin=margin,
                                           ref_bank=ref_bank, ref_thresh=ref_thresh,
                                           ref_margin=ref_margin, classifier=classifier)
        if self.heuristic.use_ref or self.world_mode == "heuristic":
            self.use_world = False
        elif self.world_mode == "world":
            self.use_world = True
        else:
            try:
                self.use_world = bool(needs_world(self.query_raw, self.targets))
            except Exception:
                self.use_world = False
        self.phrases = phrases_for_query(self.query_raw, self.targets, self.world_mode) \
            if self.use_world else []
        if self.world_mode == "world" and not self.phrases:
            self.phrases = phrases_for_query(self.query_raw, self.targets, "world")
        self._calls = 0
        self._last_boxes = []

    @staticmethod
    def _heuristic_targets(targets, query_raw, ref_bank, world_mode):
        """열린 말만 있고 색이 폴백(초록 기본값)이면 휴리스틱을 끈다([]).
        직접지정(ref)·heuristic 강제 모드에서는 그대로."""
        if ref_bank or world_mode == "heuristic":
            return targets
        from .world import open_phrases_for_raw, is_fallback_targets
        if is_fallback_targets(query_raw, targets) and open_phrases_for_raw(query_raw):
            return []
        return targets

    def judge(self, crop_bgr, mask_crop=None):
        return self.heuristic.judge(crop_bgr, mask_crop)

    def judge_frame(self, frame_bgr, items):
        """frame 1장에 대해 일괄 판정. items: [{key, xyxy, crop, mc}].
        returns {key: (hit, detail)}. 월드는 stride마다 1회만 돈다.
        API형 VLM Slow path를 붙일 자리도 여기다(키프레임 단위)."""
        out = {}
        if not items:
            return out
        world_boxes = []
        if self.use_world and self.phrases:
            self._calls += 1
            if (self._calls - 1) % self.world_stride == 0:
                try:
                    from .world import get_world
                    world_boxes = get_world().ground(frame_bgr, self.phrases,
                                                     conf=self.world_conf,
                                                     imgsz=self.world_imgsz)
                    self._last_boxes = world_boxes
                except Exception:
                    world_boxes = self._last_boxes
            else:
                world_boxes = self._last_boxes
        from .world import match_person, FACE_ITEMS
        face_boxes = [b for b in world_boxes if b.get("phrase") in FACE_ITEMS]
        body_boxes = [b for b in world_boxes if b.get("phrase") not in FACE_ITEMS]
        for it in items:
            key = it["key"]
            h_hit, h_detail = self.heuristic.judge(it.get("crop"), it.get("mc"))
            if not self.use_world or not self.phrases:
                out[key] = (h_hit, h_detail)
                continue
            try:
                x1c, y1c, x2c, y2c = it["xyxy"]
                face = [x1c, y1c, x2c, int(y1c + (y2c - y1c) * 0.42)]
                w_hit, best, iou, where = False, None, 0.0, ""
                if face_boxes:
                    fh, fb, fi = match_person(it["xyxy"], face_boxes,
                                              face_xyxy=face, face_only=True)
                    if fh:
                        w_hit, best, iou, where = fh, fb, fi, "face"
                if body_boxes:
                    bh, bb, bi = match_person(it["xyxy"], body_boxes)
                    if bh and (bb["conf"] >= (best["conf"] if best else -1.0)):
                        w_hit, best, iou, where = bh, bb, bi, "person"
            except Exception:
                w_hit, best, iou, where = False, None, 0.0, ""
            hit = bool(h_hit or w_hit)
            sel = "heuristic"
            if w_hit and not h_hit:
                sel = "world"
            elif w_hit and h_hit:
                sel = "hybrid"
            detail = dict(h_detail)
            detail["selector"] = sel
            detail["world_hit"] = bool(w_hit)
            detail["world_phrases"] = list(self.phrases)
            detail["world_boxes"] = len(world_boxes)
            if best is not None:
                detail["world_match"] = {"phrase": best["phrase"],
                                         "conf": best["conf"], "iou": iou,
                                         "where": where}
            out[key] = (hit, detail)
        return out
