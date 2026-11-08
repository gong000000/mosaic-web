"""분석 결과를 실제 모자이크 렌더 요청으로 바꾸는 작은 서비스 계층.

UI(app.py)는 체크리스트/세션 상태만 다루고, 어떤 항목을 어떤 priors로
넘길지·추적 캐시 서명이 무엇인지는 이 모듈이 담당한다. 렌더 결과 자체는
버튼을 누를 때마다 새로 만들고, 서명은 1패스 추적 캐시에만 쓴다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .config import CATEGORY_TYPE, category_type
from .pipeline import mosaic_video


def _int_boxes(boxes):
    out = []
    for box in boxes or []:
        try:
            coords = [int(float(v)) for v in box]
        except (TypeError, ValueError):
            continue
        if len(coords) == 4 and coords[2] > coords[0] and coords[3] > coords[1]:
            out.append(coords)
    return out


def _int_samples(samples):
    out = {}
    for key, boxes in (samples or {}).items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError):
            continue
        cleaned = _int_boxes(boxes)
        if cleaned:
            out[frame_idx] = cleaned
    return out


def _json_samples(samples):
    return {str(k): v for k, v in sorted((samples or {}).items())}


@dataclass(frozen=True)
class MosaicSettings:
    """사용자가 고른 렌더 효과. 값 자체가 렌더 캐시 키가 된다."""

    style: str = "pixel"
    blocks: int = 30
    blur: int = 61
    area_mode: str = "face_only"

    def as_dict(self):
        return {
            "style": self.style,
            "blocks": int(self.blocks),
            "blur": int(self.blur),
            "area_mode": self.area_mode,
        }


@dataclass
class RenderPlan:
    """선택된 체크리스트 항목을 pipeline 입력으로 변환한 결과."""

    targets: tuple[str, ...]
    region_priors: dict = field(default_factory=dict)
    person_priors: dict = field(default_factory=dict)
    mask_all_persons: bool = False
    settings: MosaicSettings = field(default_factory=MosaicSettings)

    @classmethod
    def from_items(cls, items, selected_cats, settings):
        selected = set(selected_cats or [])
        targets = []
        region_priors = {}
        person_priors = {}
        for item in items or []:
            cat = item.get("category")
            if cat not in selected:
                continue
            targets.append(cat)
            samples = _int_samples(item.get("samples") or {})
            if not samples:
                continue
            rtype = CATEGORY_TYPE.get(
                cat, item.get("type") or category_type(cat))
            if rtype == "region":
                region_priors[cat] = samples
            else:
                person_priors[cat] = samples
        # 얼굴은 VLM이 모든 프레임에서 찾지 못해도 사람 트랙 전체에
        # 적용한다(프라이버시 우선). 명찰/동승자는 찾은 위치만 쓴다.
        mask_all_persons = "face" in selected
        return cls(tuple(sorted(set(targets))), region_priors,
                   person_priors, mask_all_persons, settings)

    def pass1_signature(self, source_token, model_name, imgsz, stride, conf):
        payload = {
            "source": str(source_token),
            "targets": list(self.targets),
            "regions": {k: _json_samples(v)
                        for k, v in sorted(self.region_priors.items())},
            "persons": {k: _json_samples(v)
                        for k, v in sorted(self.person_priors.items())},
            "mask_all": self.mask_all_persons,
            "model": str(model_name),
            "imgsz": int(imgsz),
            "stride": int(stride),
            "conf": round(float(conf), 3),
        }
        raw = json.dumps(payload, sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def render_signature(self, source_token, model_name, imgsz, stride, conf):
        payload = {
            "pass1": self.pass1_signature(source_token, model_name,
                                           imgsz, stride, conf),
            "settings": self.settings.as_dict(),
        }
        raw = json.dumps(payload, sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def render(self, input_path, output_path, *, model_name, imgsz, stride,
               conf, preset, progress_fn=None, query_raw="", api_key=None,
               vlm_model=None, vlm_base_url=None, cache_path=None,
               use_cache=True, vlm_call_limit=0):
        """pipeline에 렌더를 위임한다. VLM 재호출은 기본 0회."""
        return mosaic_video(
            input_path, output_path,
            targets=list(self.targets),
            conf=conf,
            model_name=model_name,
            imgsz=imgsz,
            blocks=self.settings.blocks,
            style=self.settings.style,
            blur=self.settings.blur,
            ultra=False,
            area_mode=self.settings.area_mode,
            encode_preset=preset,
            keyframe_stride=stride,
            progress_fn=progress_fn,
            query_raw=query_raw,
            api_key=api_key,
            vlm_model=vlm_model,
            vlm_base_url=vlm_base_url,
            region_priors=self.region_priors,
            person_priors=self.person_priors,
            cache_path=cache_path,
            use_cache=use_cache,
            vlm_call_limit=vlm_call_limit,
            mask_all_persons=self.mask_all_persons,
        )
