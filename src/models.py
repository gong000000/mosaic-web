"""모델 로딩 캐시. weights/ 우선, 루트 fallback."""
import os
from ultralytics import YOLO

_WEIGHTS = ("weights", ".")
_STORE = None


class ModelStore:
    """YOLO 모델 보관함. 같은 이름은 한 번만 로드."""

    def __init__(self, search_dirs=("weights", ".")):
        self.search_dirs = search_dirs
        self._models = {}

    def _resolve(self, name):
        for d in self.search_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join(self.search_dirs[0], name)

    def get(self, name="yolo11m-seg.pt"):
        if name not in self._models:
            self._models[name] = YOLO(self._resolve(name))
        return self._models[name]


def get_model(name="yolo11m-seg.pt"):
    global _STORE
    if _STORE is None:
        _STORE = ModelStore()
    return _STORE.get(name)
