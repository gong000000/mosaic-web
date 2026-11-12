"""보관 정책. 24시간이 지난 결과물을 자동 삭제한다.
영상 파일은 로컬에만 저장된다. (VLM 사용 시 판정용 프레임만 외부 전송)"""
import os
import threading
import time
try:
    from .config import OUTPUT_DIR
except ImportError:
    # src/retention.py를 직접 실행할 때를 위한 절대 import 처리
    from config import OUTPUT_DIR

_LOCK = threading.Lock()
_SKIP = {"jobs.json", "jobs.json.tmp"}


class RetentionPolicy:
    """결과물 보관 정책. 기준 시간 지난 파일을 지운다."""

    def __init__(self, directory=OUTPUT_DIR, max_hours=24):
        self.directory = directory
        self.max_hours = max_hours

    def purge_old(self, max_hours=None):
        limit = (self.max_hours if max_hours is None else max_hours) * 3600
        now = time.time()
        removed = 0
        with _LOCK:
            if not os.path.isdir(self.directory):
                return 0
            for f in os.listdir(self.directory):
                if f in _SKIP:
                    continue
                p = os.path.join(self.directory, f)
                if not os.path.isfile(p):
                    continue
                try:
                    if now - os.path.getmtime(p) > limit:
                        os.unlink(p)
                        removed += 1
                except Exception:
                    pass
        return removed


_POLICY = RetentionPolicy()
RETENTION_HOURS = _POLICY.max_hours
purge_old = _POLICY.purge_old
