"""작업 히스토리. outputs/ 폴더 + jobs.json. 계정별 필터/보관은 사용자 단위로 관리한다."""
import json
import os
import threading
import time
from collections import defaultdict
from .config import OUTPUT_DIR

# 여러 세션이 동시에 저장/삭제해도 jobs.json이 깨지지 않게 한다.
_LOCK = threading.RLock()


def _owner_of(file_name):
    raw = str(file_name or "")
    return raw.split("--", 1)[0] if "--" in raw else ""


class JobHistory:
    """결과물 기록 보관함. 저장 폴더를 상태로 가진다.

    한 사용자가 작업을 많이 돌려도 다른 사용자의 기록이 밀려나지 않도록
    max_rows를 '사용자별' 상한으로 적용한다.
    """

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

    def _trim_per_user(self, rows):
        buckets = defaultdict(list)
        for idx, r in enumerate(rows):
            owner = r.get("user") or _owner_of(r.get("file"))
            buckets[owner].append((idx, r))
        keep_indices = set()
        for items in buckets.values():
            for idx, _r in items[-self.max_rows:]:
                keep_indices.add(idx)
        return [r for idx, r in enumerate(rows) if idx in keep_indices]

    def _save(self, rows):
        with _LOCK:
            os.makedirs(self.directory, exist_ok=True)
            trimmed = self._trim_per_user(rows)
            # 임시 파일에 쓴 뒤 교체해, 저장 도중 앱이 멈춰도 기존 기록이
            # 반쯤 잘린 JSON으로 남지 않게 한다.
            tmp = self.db + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(trimmed, f, ensure_ascii=False, indent=2)
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
            query_raw=None, selector=None, summary=None, user=None):
        with _LOCK:
            rows = self._load()
            owner = user or _owner_of(out_name)
            rows.append({
                "at": time.strftime("%m-%d %H:%M"),
                "user": owner,
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

    def all_rows(self, user=None):
        rows = list(reversed(self._load()))
        if not user:
            return rows
        prefix = f"{user}--"
        out = [r for r in rows
               if r.get("user") == user
               or str(r.get("file", "")).startswith(prefix)]
        known = {r.get("file") for r in out if r.get("file")}
        # jobs.json에서 누락됐지만 실제 outputs/ 폴더에 남아 있는 내 결과 파일도 복구한다.
        try:
            for fname in sorted(os.listdir(self.directory), reverse=True):
                if (fname.startswith(prefix) and fname.endswith(".mp4")
                        and not fname.endswith(".tmp.mp4")
                        and not fname.endswith(".noaudio.mp4")
                        and fname not in known):
                    p = os.path.join(self.directory, fname)
                    mtime = time.strftime(
                        "%m-%d %H:%M",
                        time.localtime(os.path.getmtime(p)))
                    out.append({
                        "at": mtime,
                        "user": user,
                        "src": fname[len(prefix):],
                        "prompt": "",
                        "query_raw": "",
                        "target": "",
                        "invert": False,
                        "file": fname,
                        "sec": 0.0,
                        "selector": "vlm",
                        "summary": "",
                    })
        except OSError:
            pass
        return out

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
