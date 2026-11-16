"""보관 정책. 오래된 결과물과 용량 초과분을 자동 삭제한다.

영상 파일은 로컬/볼륨에만 저장된다. (VLM 사용 시 판정용 프레임만 외부 전송)
jobs.db, jobs.json, WAL 파일은 절대 삭제하지 않는다.
"""
import os
import threading
import time
try:
    from .config import OUTPUT_DIR
except ImportError:
    # src/retention.py를 직접 실행할 때를 위한 절대 import 처리
    from config import OUTPUT_DIR

_LOCK = threading.Lock()
_SKIP_NAMES = {
    "jobs.json", "jobs.json.tmp", "jobs.json.migrated",
    "jobs.db", "jobs.db-wal", "jobs.db-shm", "jobs.db-journal",
}


def _env_int(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(0, int(default))


class RetentionPolicy:
    """결과물 보관 정책. 기준 시간/개수/용량을 넘으면 오래된 것부터 지운다."""

    def __init__(self, directory=OUTPUT_DIR, max_hours=24,
                 max_files=None, max_bytes=None):
        self.directory = directory
        self.max_hours = max(0.0, float(max_hours))
        self.max_files = (_env_int("MOSAIC_MAX_OUTPUT_FILES", 500)
                          if max_files is None else max(0, int(max_files)))
        if max_bytes is None:
            max_gb = _env_int("MOSAIC_MAX_OUTPUT_GB", 20)
            max_bytes = max_gb * 1024 ** 3
        self.max_bytes = max(0, int(max_bytes))

    def _result_files(self):
        """(path, name, size, mtime) 목록. DB/설정 파일은 제외한다."""
        out = []
        if not os.path.isdir(self.directory):
            return out
        for name in os.listdir(self.directory):
            if name in _SKIP_NAMES or name.startswith("jobs.db"):
                continue
            if not name.endswith(".mp4"):
                continue
            p = os.path.join(self.directory, name)
            try:
                if os.path.isfile(p):
                    st = os.stat(p)
                    out.append((p, name, int(st.st_size), float(st.st_mtime)))
            except OSError:
                pass
        return out

    def _unlink(self, path):
        try:
            os.unlink(path)
            return 1
        except OSError:
            return 0

    def purge_old(self, max_hours=None):
        limit = (self.max_hours if max_hours is None else max_hours) * 3600
        now = time.time()
        removed = 0
        with _LOCK:
            for p, _name, _size, mtime in self._result_files():
                if now - mtime > limit:
                    removed += self._unlink(p)
        return removed

    def usage(self):
        files = self._result_files()
        return {
            "files": len(files),
            "bytes": sum(f[2] for f in files),
            "max_files": self.max_files,
            "max_bytes": self.max_bytes,
        }

    def enforce(self, max_hours=None):
        """24시간 만료 → 개수 상한 → 용량 상한 순으로 정리한다."""
        removed = self.purge_old(max_hours=max_hours)
        with _LOCK:
            files = sorted(self._result_files(), key=lambda f: f[3])
            if self.max_files and len(files) > self.max_files:
                for p, _n, _s, _m in files[:len(files) - self.max_files]:
                    removed += self._unlink(p)
                files = sorted(self._result_files(), key=lambda f: f[3])
            if self.max_bytes:
                total = sum(f[2] for f in files)
                for p, _n, size, _m in files:
                    if total <= self.max_bytes:
                        break
                    removed += self._unlink(p)
                    total -= size
        return removed


_POLICY = RetentionPolicy()
RETENTION_HOURS = _POLICY.max_hours
purge_old = _POLICY.purge_old
enforce = _POLICY.enforce
usage = _POLICY.usage
