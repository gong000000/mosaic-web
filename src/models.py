"""모델 로딩 캐시. weights/ 우선, 프로젝트 루트 fallback.
여러 세션이 같은 모델/트래커를 동시에 만지지 않도록 공용 락도 제공한다."""
import os
import threading
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIGHT_DIRS = (os.path.join(ROOT_DIR, "weights"), ROOT_DIR)

# 추적/추론은 상태를 가진 모델을 쓰므로 프로세스 단위로 한 번에 하나만 실행한다.
MODEL_LOCK = threading.RLock()

_STORE = None


class ModelStore:
    """YOLO 모델 보관함. 같은 이름은 한 번만 로드."""

    def __init__(self, search_dirs=_WEIGHT_DIRS):
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
