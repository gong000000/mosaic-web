"""작업 히스토리. outputs/ 폴더 + SQLite(jobs.db).

예전에는 jobs.json을 읽고 전체를 다시 쓰는 방식이라 동시 작업에서
덮어쓰기 위험이 있었다. 이제 작업 1건은 INSERT 1번, 사용자별 상한은
DELETE 1번으로 처리한다. 기존 jobs.json은 최초 실행 때 자동으로
jobs.db로 옮기고 `.migrated`로 보존한다.
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from .config import OUTPUT_DIR

_LOCK = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL DEFAULT '',
    user TEXT NOT NULL DEFAULT '',
    src TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    query_raw TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    invert INTEGER NOT NULL DEFAULT 0,
    file TEXT NOT NULL DEFAULT '',
    sec REAL NOT NULL DEFAULT 0,
    selector TEXT NOT NULL DEFAULT 'vlm',
    summary TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL DEFAULT 0
)
"""


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
        self.max_rows = max(1, int(max_rows))
        self.db = os.path.join(directory, "jobs.db")
        self.legacy = os.path.join(directory, "jobs.json")
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with _LOCK:
            os.makedirs(self.directory, exist_ok=True)
            with self._db() as c:
                c.execute(_SCHEMA)
                c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user "
                          "ON jobs(user, id)")
                n = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if n == 0 and os.path.exists(self.legacy):
                self._migrate_legacy()

    def _insert(self, c, r):
        file_name = str(r.get("file") or "")
        owner = str(r.get("user") or _owner_of(file_name))
        c.execute(
            "INSERT INTO jobs (at, user, src, prompt, query_raw, target, "
            "invert, file, sec, selector, summary, created) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(r.get("at") or time.strftime("%m-%d %H:%M")),
             owner, str(r.get("src") or ""), str(r.get("prompt") or ""),
             str(r.get("query_raw") or r.get("prompt") or ""),
             str(r.get("target") or ""), 1 if r.get("invert") else 0,
             file_name, float(r.get("sec") or 0),
             str(r.get("selector") or "vlm"),
             str(r.get("summary") or ""),
             float(r.get("created") or time.time())))

    def _migrate_legacy(self):
        try:
            with open(self.legacy, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            return
        if not isinstance(rows, list):
            return
        with _LOCK, self._db() as c:
            for r in rows:
                if isinstance(r, dict) and r.get("file"):
                    self._insert(c, r)
        try:
            os.replace(self.legacy, self.legacy + ".migrated")
        except OSError:
            pass

    def add(self, src_name, prompt, target, invert, out_name, seconds,
            query_raw=None, selector=None, summary=None, user=None):
        owner = user or _owner_of(out_name)
        row = {
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
            "created": time.time(),
        }
        with _LOCK, self._db() as c:
            self._insert(c, row)
            # 사용자별 최근 max_rows개만 남긴다.
            c.execute(
                "DELETE FROM jobs WHERE user=? AND id NOT IN "
                "(SELECT id FROM jobs WHERE user=? ORDER BY id DESC LIMIT ?)",
                (owner, owner, self.max_rows))

    def _rows(self):
        with _LOCK, self._db() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT at, user, src, prompt, query_raw, target, invert, "
                "file, sec, selector, summary FROM jobs "
                "ORDER BY id DESC").fetchall()]
        for r in rows:
            r["invert"] = bool(r.get("invert"))
        return rows

    def all_rows(self, user=None):
        rows = self._rows()
        if not user:
            return rows
        prefix = f"{user}--"
        out = [r for r in rows
               if r.get("user") == user
               or str(r.get("file", "")).startswith(prefix)]
        known = {r.get("file") for r in out if r.get("file")}
        # DB에서 누락됐지만 실제 outputs/ 폴더에 남아 있는 내 결과 파일도
        # 복구한다(이전 버전/수동 복사/마이그레이션 누락 대비).
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
        with _LOCK, self._db() as c:
            rows = c.execute("SELECT id, file FROM jobs").fetchall()
            gone = [(int(r["id"]),) for r in rows
                    if not os.path.exists(os.path.join(self.directory,
                                                       r["file"]))]
            c.executemany("DELETE FROM jobs WHERE id=?", gone)
            return len(gone)

    def delete(self, out_name):
        with _LOCK, self._db() as c:
            c.execute("DELETE FROM jobs WHERE file=?", (out_name,))
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
