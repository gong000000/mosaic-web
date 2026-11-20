"""모델 로딩 캐시. weights/ 우선, 프로젝트 루트도 검색.
여러 세션이 같은 모델/트래커를 동시에 만지지 않도록 공용 락도 제공한다."""
import logging
import os
import threading
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HOME_DIR = os.path.expanduser("~")
log = logging.getLogger("mosaic.models")
# MOSAIC_WEIGHTS_DIR(예: Modal 이미지 빌드 때 받아 둔 /opt/weights)를 먼저 본다.
_WEIGHT_DIRS = tuple(d for d in (
    os.environ.get("MOSAIC_WEIGHTS_DIR"),
    os.path.join(ROOT_DIR, "weights"),
    ROOT_DIR,
    os.path.join(_HOME_DIR, "weights"),
    _HOME_DIR,
) if d)
FALLBACK_SEG = "yolo11n-seg.pt"

# 추적/추론은 상태를 가진 모델을 쓰므로 프로세스 단위로 한 번에 하나만 실행한다.
MODEL_LOCK = threading.RLock()

_STORE = None
_STORE_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()


class ModelStore:
    """YOLO 모델 보관함. 같은 이름은 한 번만 로드.

    Streamlit은 세션마다 스레드를 쓰므로, 두 세션이 동시에 같은 모델을
    로드해 가중치를 두 번 읽는 일이 없도록 로드 구간을 잠근다.
    """

    def __init__(self, search_dirs=_WEIGHT_DIRS):
        self.search_dirs = search_dirs
        self._models = {}
        self._lock = threading.Lock()

    def _resolve(self, name):
        for d in self.search_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join(self.search_dirs[0], name)

    def get(self, name="yolo11n-seg.pt"):
        with self._lock:
            if name not in self._models:
                try:
                    self._models[name] = YOLO(self._resolve(name))
                except Exception:
                    # 큰 모델 가중치를 못 받으면 기본 nano 모델로 대체한다.
                    if name == FALLBACK_SEG or not name.endswith("-seg.pt"):
                        raise
                    log.exception("%s 로드 실패 - %s 로 대체", name,
                                  FALLBACK_SEG)
                    self._models[name] = YOLO(self._resolve(FALLBACK_SEG))
            return self._models[name]


def get_model(name="yolo11n-seg.pt"):
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ModelStore()
        store = _STORE
    return store.get(name)


def get_thread_model(name="yolo11n-seg.pt"):
    """작업 스레드마다 별도 YOLO 모델을 돌려준다.

    Ultralytics의 `track(persist=True)`는 모델 객체 안에 트래커 상태를
    보관한다. 여러 렌더 스레드가 같은 모델을 동시에 추적하면 트랙 ID가
    섞이므로, 렌더용 모델은 스레드 로컬 저장소에 따로 둔다.
    """
    store = getattr(_THREAD_LOCAL, "store", None)
    if store is None:
        store = ModelStore()
        _THREAD_LOCAL.store = store
    return store.get(name)
