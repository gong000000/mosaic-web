"""작업 히스토리. outputs/ 폴더 + jobs.json. 로그인 없이 포트폴리오용 최소 기록."""
import json
import os
import threading
import time
from .config import OUTPUT_DIR

# 여러 세션이 동시에 저장/삭제해도 jobs.json이 깨지지 않게 한다.
_LOCK = threading.RLock()


class JobHistory:
    """결과물 기록 보관함. 저장 폴더를 상태로 가진다."""

    def __init__(self, directory=OUTPUT_DIR, max_rows=100):
        self.directory = directory
        self.max_rows = max_rows
        self.db = os.path.join(directory, "jobs.json")

    def _load(self):
        with _LOCK:
            try:
                with open(self.db, encoding="utf-8") as f:
                    rows = json.load(f)
                return rows if isinstance(rows, list) else []
            except Exception:
                return []

    def _save(self, rows):
        with _LOCK:
            os.makedirs(self.directory, exist_ok=True)
            # 임시 파일에 쓴 뒤 교체해, 저장 도중 앱이 멈춰도 기존 기록이
            # 반쯤 잘린 JSON으로 남지 않게 한다.
            tmp = self.db + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(rows[-self.max_rows:], f,
                              ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.db)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass

    def add(self, src_name, prompt, target, invert, out_name, seconds,
            query_raw=None, selector=None, summary=None):
        with _LOCK:
            rows = self._load()
            rows.append({
                "at": time.strftime("%m-%d %H:%M"),
                "src": src_name,
                "prompt": prompt,
                "query_raw": query_raw if query_raw is not None else prompt,
                "target": target,
                "invert": bool(invert),
                "file": out_name,
                "sec": round(float(seconds), 1),
                "selector": selector or "vlm",
                "summary": summary or "",
            })
            self._save(rows)

    def all_rows(self):
        return list(reversed(self._load()))

    def remove_missing(self):
        """파일이 이미 지워진(만료/수동 삭제) 기록을 정리한다."""
        with _LOCK:
            rows = self._load()
            keep = [r for r in rows
                    if r.get("file")
                    and os.path.exists(os.path.join(self.directory,
                                                    r["file"]))]
            if len(keep) != len(rows):
                self._save(keep)
            return len(rows) - len(keep)

    def delete(self, out_name):
        with _LOCK:
            self._save([r for r in self._load()
                        if r.get("file") != out_name])
            try:
                p = os.path.join(self.directory, out_name)
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


_HISTORY = JobHistory()
add = _HISTORY.add
all_rows = _HISTORY.all_rows
remove_missing = _HISTORY.remove_missing
delete = _HISTORY.delete
