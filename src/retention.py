"""보관 정책. 업로드 24시간 경과/다운로드 시 자동 삭제.
영상 프레임 외부 전송 없음. 전부 로컬 처리."""
import os
import time
from .config import OUTPUT_DIR


class RetentionPolicy:
    """결과물 보관 정책. 기준 시간 지난 파일을 지운다."""

    def __init__(self, directory=OUTPUT_DIR, max_hours=24):
        self.directory = directory
        self.max_hours = max_hours

    def purge_old(self, max_hours=None):
        limit = (self.max_hours if max_hours is None else max_hours) * 3600
        now = time.time()
        removed = 0
        if not os.path.isdir(self.directory):
            return 0
        for f in os.listdir(self.directory):
            p = os.path.join(self.directory, f)
            if not os.path.isfile(p) or f == "jobs.json":
                continue
            try:
                if now - os.path.getmtime(p) > limit:
                    os.unlink(p)
                    removed += 1
            except Exception:
                pass
        return removed

    def remove_on_download(self, path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
                return True
        except Exception:
            pass
        return False


_POLICY = RetentionPolicy()
RETENTION_HOURS = _POLICY.max_hours
purge_old = _POLICY.purge_old
remove_on_download = _POLICY.remove_on_download
