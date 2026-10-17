"""작업 히스토리. outputs/ 폴더 + jobs.json. 로그인 없이 포트폴리오용 최소 기록."""
import json
import os
import time
from .config import OUTPUT_DIR


class JobHistory:
    """결과물 기록 보관함. 저장 폴더를 상태로 가진다."""

    def __init__(self, directory=OUTPUT_DIR, max_rows=100):
        self.directory = directory
        self.max_rows = max_rows
        self.db = os.path.join(directory, "jobs.json")

    def _load(self):
        try:
            with open(self.db, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, rows):
        os.makedirs(self.directory, exist_ok=True)
        with open(self.db, "w", encoding="utf-8") as f:
            json.dump(rows[-self.max_rows:], f, ensure_ascii=False, indent=2)

    def add(self, src_name, prompt, target, invert, out_name, seconds,
            query_raw=None, selector=None, summary=None):
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
            "selector": selector or "heuristic",
            "summary": summary or "",
        })
        self._save(rows)

    def all_rows(self):
        return list(reversed(self._load()))

    def delete(self, out_name):
        self._save([r for r in self._load() if r.get("file") != out_name])
        try:
            p = os.path.join(self.directory, out_name)
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


_HISTORY = JobHistory()
add = _HISTORY.add
all_rows = _HISTORY.all_rows
delete = _HISTORY.delete
