"""지속형 렌더 작업 큐.

Streamlit 세션 스레드가 직접 렌더하면 창을 닫거나 컨테이너가 재시작될 때
작업이 사라진다. 이 모듈은 입력 영상을 볼륨에 복사하고 작업을 SQLite에
기록한 뒤, 백그라운드 워커 스레드가 하나씩 처리한다.

- `max_containers=1` 환경에서는 프로세스 재시작 후에도 queued 작업을 이어간다.
- 여러 컨테이너에서 동시에 실행되더라도 `running` 작업이 있으면 새 작업을
  잡지 않는다(간단한 전역 직렬화). SQLite는 WAL + busy_timeout을 쓴다.
- 원본 입력 파일은 작업 종료/실패 시 삭제한다.
"""
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from .config import (CATEGORY_TYPE, GEMINI_BASE_URL, OUTPUT_DIR, VLM_MODEL,
                     env_api_key)
from .errors import RenderCancelled
from .render import strip_audio
from .render_service import MosaicSettings, RenderPlan
from . import history as H
from . import chunk_render
from . import face_audit

log = logging.getLogger("mosaic.jobqueue")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS render_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    created REAL NOT NULL DEFAULT 0,
    started REAL NOT NULL DEFAULT 0,
    finished REAL NOT NULL DEFAULT 0,
    progress REAL NOT NULL DEFAULT 0,
    input_path TEXT NOT NULL DEFAULT '',
    output_path TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL DEFAULT 0,
    audit TEXT NOT NULL DEFAULT ''
)
"""


class JobCancelled(Exception):
    pass

_ITEM_KEYS = ("category", "label_ko", "type", "samples", "boxes",
              "region_type", "source", "count")


def _json_items(items):
    """numpy 썸네일 등을 제거하고 렌더에 필요한 항목만 직렬화한다."""
    out = []
    for it in items or []:
        row = {k: it.get(k) for k in _ITEM_KEYS if k in it}
        samples = {}
        for key, boxes in (it.get("samples") or {}).items():
            cleaned = []
            for b in boxes or []:
                try:
                    cleaned.append([float(v) for v in b[:4]])
                except (TypeError, ValueError):
                    pass
            if cleaned:
                samples[str(key)] = cleaned
        if samples:
            row["samples"] = samples
        boxes = []
        for b in it.get("boxes") or []:
            try:
                boxes.append([float(v) for v in b[:4]])
            except (TypeError, ValueError):
                pass
        if boxes:
            row["boxes"] = boxes
        out.append(row)
    return out


class RenderQueue:
    def __init__(self, db_path=None, input_dir=None, max_parallel=None):
        self.db = db_path or os.path.join(OUTPUT_DIR, "jobs.db")
        self.input_dir = input_dir or os.path.join(OUTPUT_DIR, "job_inputs")
        if max_parallel is None:
            try:
                max_parallel = int(os.environ.get("MOSAIC_MAX_PARALLEL", "3"))
            except (TypeError, ValueError):
                max_parallel = 3
        # 계정별로는 한 작업만, 서버 전체로는 최대 N개 계정 작업을 동시에
        # 처리한다. 다른 아이디의 작업이 내 대기 순번을 막지 않게 하기 위함.
        self.max_parallel = max(1, int(max_parallel))
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
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
        os.makedirs(os.path.dirname(self.db) or ".", exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        with self._db() as c:
            c.execute(_SCHEMA)
            cols = {r[1] for r in c.execute(
                "PRAGMA table_info(render_jobs)").fetchall()}
            if "cancel_requested" not in cols:
                c.execute("ALTER TABLE render_jobs ADD COLUMN "
                          "cancel_requested INTEGER NOT NULL DEFAULT 0")
            if "updated" not in cols:
                c.execute("ALTER TABLE render_jobs ADD COLUMN "
                          "updated REAL NOT NULL DEFAULT 0")
            if "audit" not in cols:
                c.execute("ALTER TABLE render_jobs ADD COLUMN "
                          "audit TEXT NOT NULL DEFAULT ''")
            c.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_status "
                      "ON render_jobs(status, id)")
            # 워커가 여러 프로세스/컨테이너에서 중복 실행되지 않도록
            # 짧은 리스를 둔다. heartbeat가 오래된 경우에만 새 워커가
            # 이어받는다.
            c.execute(
                "CREATE TABLE IF NOT EXISTS queue_worker ("
                " id INTEGER PRIMARY KEY CHECK (id=1),"
                " owner TEXT NOT NULL DEFAULT '',"
                " heartbeat REAL NOT NULL DEFAULT 0)")

    def enqueue(self, user, input_path, output_path, items, selected_cats,
                settings, model_name, imgsz, stride, conf, preset,
                query_raw, vlm_model=VLM_MODEL,
                vlm_base_url=GEMINI_BASE_URL, src_name="", preset_name="",
                person_total=0, strip_audio=False):
        token = uuid.uuid4().hex
        ext = os.path.splitext(input_path)[1] or ".mp4"
        stable = os.path.join(self.input_dir, token + ext)
        shutil.copyfile(input_path, stable)
        payload = {
            "items": _json_items(items),
            "selected_cats": list(selected_cats or []),
            "settings": settings.as_dict(),
            "model_name": model_name,
            "imgsz": int(imgsz),
            "stride": int(stride),
            "conf": float(conf),
            "preset": preset,
            "query_raw": query_raw,
            "vlm_model": vlm_model,
            "vlm_base_url": vlm_base_url,
            "src_name": src_name,
            "preset_name": preset_name,
            "person_total": int(person_total or 0),
            "strip_audio": bool(strip_audio),
        }
        try:
            with self._db() as c:
                cur = c.execute(
                    "INSERT INTO render_jobs (user, status, created, "
                    "progress, input_path, output_path, payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (user or "", "queued", time.time(), 0.0, stable,
                     output_path, json.dumps(payload, ensure_ascii=False)))
                return int(cur.lastrowid)
        except Exception:
            try:
                os.remove(stable)
            except OSError:
                pass
            raise

    def get(self, job_id):
        with self._db() as c:
            row = c.execute("SELECT * FROM render_jobs WHERE id=?",
                            (int(job_id),)).fetchone()
        return dict(row) if row else None

    def position(self, job_id, user=None):
        """같은 사용자의 앞선 queued/running 작업 수 + 1."""
        with self._db() as c:
            if user is None:
                n = c.execute(
                    "SELECT COUNT(*) FROM render_jobs WHERE id<? AND "
                    "status IN ('queued','running')",
                    (int(job_id),)).fetchone()[0]
            else:
                n = c.execute(
                    "SELECT COUNT(*) FROM render_jobs WHERE user=? AND id<? "
                    "AND status IN ('queued','running')",
                    (user or "", int(job_id))).fetchone()[0]
        return int(n) + 1

    def active_jobs(self, user):
        """해당 사용자의 queued/running 작업 목록."""
        with self._db() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT id, status, progress, created, started, output_path, "
                "cancel_requested FROM render_jobs WHERE user=? AND status IN "
                "('queued','running') ORDER BY id", (user or "",)).fetchall()]
        return rows

    def has_active_jobs(self):
        with self._db() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM render_jobs WHERE status IN "
                "('queued','running')").fetchone()[0]
        return int(n) > 0

    def status_snapshot(self):
        """관리자 상태 페이지용 큐/워커/보관 현황."""
        now = time.time()
        with self._db() as c:
            counts = {str(r["status"]): int(r["n"]) for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM render_jobs "
                "GROUP BY status")}
            active = [dict(r) for r in c.execute(
                "SELECT id,user,status,progress,created,started,updated,"
                "output_path,error,cancel_requested FROM render_jobs "
                "WHERE status IN ('queued','running') ORDER BY id").fetchall()]
            failures = [dict(r) for r in c.execute(
                "SELECT id,user,finished,error FROM render_jobs "
                "WHERE status='failed' ORDER BY id DESC LIMIT 10").fetchall()]
            worker = c.execute(
                "SELECT owner,heartbeat FROM queue_worker WHERE id=1"
            ).fetchone()
            oldest = c.execute(
                "SELECT MIN(created) FROM render_jobs WHERE status='queued'"
            ).fetchone()[0]
            stale = c.execute(
                "SELECT id,user,progress,updated FROM render_jobs "
                "WHERE status='running' AND updated>0 AND updated<? "
                "ORDER BY id LIMIT 1", (now - 900.0,)).fetchone()
        chunk_base = os.path.join(OUTPUT_DIR, ".chunks")
        chunk_bytes = 0
        chunk_dirs = 0
        if os.path.isdir(chunk_base):
            for name in os.listdir(chunk_base):
                path = os.path.join(chunk_base, name)
                if not os.path.isdir(path):
                    continue
                chunk_dirs += 1
                for root, _dirs, files in os.walk(path):
                    for fname in files:
                        try:
                            chunk_bytes += os.path.getsize(
                                os.path.join(root, fname))
                        except OSError:
                            pass
        try:
            disk = shutil.disk_usage(OUTPUT_DIR)
            disk_free = int(disk.free)
        except OSError:
            disk_free = 0
        heartbeat = float(worker["heartbeat"] or 0) if worker else 0.0
        return {
            "counts": counts,
            "active": active,
            "failures": failures,
            "max_parallel": int(self.max_parallel),
            "worker_owner": (worker["owner"] if worker else ""),
            "worker_age": (now - heartbeat if heartbeat else None),
            "oldest_queued_age": (now - float(oldest)) if oldest else None,
            "stale": dict(stale) if stale else None,
            "chunk_dirs": int(chunk_dirs),
            "chunk_bytes": int(chunk_bytes),
            "disk_free": int(disk_free),
        }

    def acquire_worker(self, owner, ttl=20.0):
        """워커 리스를 잡는다. 다른 워커가 살아 있으면 False."""
        now = time.time()
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT owner, heartbeat FROM queue_worker WHERE id=1"
            ).fetchone()
            if (row and row["owner"]
                    and float(row["heartbeat"] or 0) > now - float(ttl)
                    and str(row["owner"]) != str(owner)):
                return False
            c.execute(
                "INSERT INTO queue_worker (id, owner, heartbeat) "
                "VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "owner=excluded.owner, heartbeat=excluded.heartbeat",
                (str(owner), now))
        return True

    def heartbeat_worker(self, owner, ttl=20.0):
        """리스 소유자만 heartbeat를 갱신할 수 있다."""
        now = time.time()
        with self._db() as c:
            cur = c.execute(
                "UPDATE queue_worker SET heartbeat=? "
                "WHERE id=1 AND owner=?", (now, str(owner)))
            return cur.rowcount > 0

    def release_worker(self, owner):
        with self._db() as c:
            c.execute("UPDATE queue_worker SET heartbeat=0 "
                      "WHERE id=1 AND owner=?", (str(owner),))

    def cancel(self, job_id, user):
        """queued면 즉시 취소, running이면 취소 요청 플래그를 세운다."""
        with self._db() as c:
            row = c.execute(
                "SELECT id, user, status FROM render_jobs WHERE id=?",
                (int(job_id),)).fetchone()
        if row is None or str(row["user"]) != str(user or ""):
            return False
        return self._cancel_job(job_id)

    def cancel_admin(self, job_id):
        """관리자용. 소유자 확인 없이 작업을 취소한다."""
        return self._cancel_job(job_id)

    def _cancel_job(self, job_id):
        remove_path = ""
        with self._db() as c:
            row = c.execute(
                "SELECT id, user, status, input_path, cancel_requested "
                "FROM render_jobs WHERE id=?", (int(job_id),)).fetchone()
            if row is None:
                return False
            if row["status"] == "queued":
                remove_path = row["input_path"] or ""
                c.execute(
                    "UPDATE render_jobs SET status='cancelled', finished=?, "
                    "progress=0, error='관리자/사용자 취소', input_path='' "
                    "WHERE id=?", (time.time(), int(job_id)))
            elif row["status"] == "running":
                if row["cancel_requested"]:
                    return False
                c.execute(
                    "UPDATE render_jobs SET cancel_requested=1 WHERE id=?",
                    (int(job_id),))
            else:
                return False
        if remove_path:
            try:
                os.remove(remove_path)
            except OSError:
                pass
        return True

    def cancel_user_jobs(self, user):
        """해당 아이디의 queued/running 작업을 모두 취소한다."""
        with self._db() as c:
            rows = c.execute(
                "SELECT id FROM render_jobs WHERE user=? AND status IN "
                "('queued','running')", (str(user or ""),)).fetchall()
        return sum(1 for row in rows
                   if self._cancel_job(int(row["id"])))

    def cancel_all_active(self, except_user=None):
        """모든 활성 작업을 취소한다. except_user는 건너뛴다."""
        with self._db() as c:
            rows = c.execute(
                "SELECT id, user FROM render_jobs WHERE status IN "
                "('queued','running') ORDER BY id").fetchall()
        n = 0
        for row in rows:
            if except_user is not None and str(row["user"]) == str(except_user):
                continue
            if self._cancel_job(int(row["id"])):
                n += 1
        return n

    def is_cancel_requested(self, job_id):
        with self._db() as c:
            row = c.execute(
                "SELECT cancel_requested, status FROM render_jobs WHERE id=?",
                (int(job_id),)).fetchone()
        return bool(row and (row["cancel_requested"]
                             or row["status"] == "cancelled"))

    def claim(self):
        """queued 작업 1개를 원자적으로 running으로 바꾼다.

        같은 계정의 작업은 한 번에 하나만 실행하고, 서버 전체 동시 실행
        수는 `max_parallel`로 제한한다. 이렇게 하면 한 계정의 긴 작업이
        다른 계정의 첫 작업을 막지 않는다.
        """
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")
            running = c.execute(
                "SELECT COUNT(*) FROM render_jobs WHERE status='running'"
            ).fetchone()[0]
            if int(running) >= int(self.max_parallel):
                return None
            row = c.execute(
                "SELECT q.id FROM render_jobs q WHERE q.status='queued' "
                "AND NOT EXISTS (SELECT 1 FROM render_jobs r "
                "WHERE r.status='running' AND r.user=q.user) "
                "ORDER BY q.id LIMIT 1").fetchone()
            if row is None:
                return None
            job_id = int(row["id"])
            now = time.time()
            c.execute("UPDATE render_jobs SET status='running', started=?, "
                      "updated=?, progress=0, error='' WHERE id=?",
                      (now, now, job_id))
            c.commit()
            job = c.execute("SELECT * FROM render_jobs WHERE id=?",
                            (job_id,)).fetchone()
        return dict(job) if job else None

    def update_progress(self, job_id, frac):
        try:
            with self._db() as c:
                c.execute("UPDATE render_jobs SET progress=?, updated=? "
                          "WHERE id=?",
                          (min(1.0, max(0.0, float(frac))), time.time(),
                           int(job_id)))
        except Exception:
            pass

    def stale_running(self, max_silence=900.0):
        """진행률이 오랫동안 갱신되지 않은 running 작업을 돌려준다."""
        cutoff = time.time() - max(1.0, float(max_silence))
        with self._db() as c:
            row = c.execute(
                "SELECT id, user, progress, updated, started FROM render_jobs "
                "WHERE status='running' AND updated>0 AND updated<? "
                "ORDER BY id LIMIT 1", (cutoff,)).fetchone()
        return dict(row) if row else None

    def set_audit(self, job_id, audit):
        try:
            raw = json.dumps(audit or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = "{}"
        with self._db() as c:
            c.execute("UPDATE render_jobs SET audit=? WHERE id=?",
                      (raw, int(job_id)))

    def finish(self, job_id, status, summary="", error=""):
        with self._db() as c:
            c.execute(
                "UPDATE render_jobs SET status=?, finished=?, progress=?, "
                "summary=?, error=?, updated=? WHERE id=?",
                (status, time.time(), 1.0 if status == "done" else 0.0,
                 summary, error[:2000], time.time(), int(job_id)))

    def recover_running(self):
        """이전 프로세스가 남긴 running 작업을 다시 queued로 되돌린다."""
        stale_inputs = []
        with self._db() as c:
            rows = c.execute(
                "SELECT input_path FROM render_jobs WHERE status='running' "
                "AND cancel_requested=1").fetchall()
            stale_inputs = [r["input_path"] for r in rows if r["input_path"]]
            c.execute("UPDATE render_jobs SET status='cancelled', finished=?, "
                      "error='서버 재시작으로 중단됨', input_path='' "
                      "WHERE status='running' AND cancel_requested=1",
                      (time.time(),))
            c.execute("UPDATE render_jobs SET status='queued', started=0, "
                      "progress=0 WHERE status='running'")
        for path in stale_inputs:
            try:
                os.remove(path)
            except OSError:
                pass

    def run_job(self, job):
        payload = json.loads(job.get("payload") or "{}")
        job_id = int(job["id"])
        output_path = job.get("output_path") or ""
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            plan = RenderPlan.from_items(
                payload.get("items") or [],
                payload.get("selected_cats") or [],
                MosaicSettings(**(payload.get("settings") or {})))
            last = [0.0]

            def on_progress(frac, done, total, elapsed):
                now = time.time()
                if now - last[0] >= 1.0 or frac >= 1.0:
                    last[0] = now
                    self.update_progress(job_id, frac)

            if self.is_cancel_requested(job_id):
                raise RenderCancelled()
            stats = chunk_render.render_resumable(
                plan, job["input_path"], output_path,
                model_name=payload.get("model_name") or "yolo11m-seg.pt",
                imgsz=int(payload.get("imgsz") or 960),
                stride=int(payload.get("stride") or 15),
                conf=float(payload.get("conf") or 0.20),
                preset=payload.get("preset") or "veryfast",
                progress_fn=on_progress,
                query_raw=payload.get("query_raw") or "",
                api_key=env_api_key(),
                vlm_model=payload.get("vlm_model") or VLM_MODEL,
                vlm_base_url=payload.get("vlm_base_url") or GEMINI_BASE_URL,
                final_mux_audio=not bool(payload.get("strip_audio")),
                cancel_fn=lambda: self.is_cancel_requested(job_id))
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("결과 영상이 만들어지지 않았습니다.")
            selected = payload.get("selected_cats") or []
            if ("face" in selected
                    and os.environ.get("MOSAIC_FACE_AUDIT", "1") != "0"):
                self.update_progress(job_id, 0.99)
                try:
                    audit = face_audit.audit_faces(
                        job["input_path"], output_path,
                        cancel_fn=lambda: self.is_cancel_requested(job_id))
                    if (audit.get("fixable")
                            and os.environ.get(
                                "MOSAIC_FACE_AUDIT_FIX", "1") != "0"):
                        fixed = face_audit.fix_audited_faces(
                            output_path, audit.get("fixable") or [],
                            style=plan.settings.style,
                            blocks=plan.settings.blocks,
                            blur=plan.settings.blur,
                            preset=payload.get("preset") or "veryfast",
                            strip_audio=bool(payload.get("strip_audio")))
                        audit["fixed"] = int(fixed)
                    if audit.get("checked") or audit.get("error"):
                        self.set_audit(job_id, audit)
                except RenderCancelled:
                    raise
                except Exception as e:
                    log.exception("얼굴 마스크 자동 점검 실패")
                    try:
                        self.set_audit(job_id, {
                            "checked": 0, "suspicious": 0, "fixable": 0,
                            "error": f"{type(e).__name__}: {e}"})
                    except Exception:
                        pass
            if payload.get("strip_audio"):
                strip_audio(output_path)
            if self.is_cancel_requested(job_id):
                raise RenderCancelled()
            person_selected = any(CATEGORY_TYPE.get(c) == "person"
                                  for c in selected)
            face_count = (payload.get("person_total")
                          if "face" in selected else stats.get("_faces"))
            if face_count is None:
                face_count = sum(1 for key, value in stats.items()
                                 if isinstance(key, int) and value.get("mosaic"))
            summary = f"{len(selected)}개 항목"
            if person_selected:
                summary += f" · 사람 {int(face_count)}명"
            regions = int(stats.get("_regions", 0) or 0)
            if regions:
                summary += f" · 배경 {regions}곳"
            elapsed = max(0.0, time.time() - float(job.get("started") or 0))
            H.add(src_name=payload.get("src_name") or "",
                  prompt=payload.get("preset_name") or "",
                  target=",".join(selected), invert=False,
                  out_name=os.path.basename(output_path), seconds=elapsed,
                  query_raw=payload.get("query_raw") or "",
                  selector="vlm", summary=summary,
                  user=job.get("user") or "")
            self.finish(job_id, "done", summary=summary)
        except RenderCancelled:
            try:
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass
            self.finish(job_id, "cancelled", error="사용자 취소")
        except Exception as e:
            try:
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass
            self.finish(job_id, "failed",
                        error=f"{type(e).__name__}: {e}")
        finally:
            try:
                status = (self.get(job_id) or {}).get("status")
                if status in ("done", "cancelled"):
                    if job.get("input_path") and os.path.exists(
                            job["input_path"]):
                        os.remove(job["input_path"])
                    self.clear_input_path(job_id)
            except Exception:
                pass

    def cleanup_finished_inputs(self):
        """완료/취소된 작업의 남은 입력 파일을 정리한다. 실패는 재시도용으로 보관."""
        with self._db() as c:
            rows = c.execute(
                "SELECT id, input_path FROM render_jobs WHERE status IN "
                "('done','cancelled') AND input_path<>''").fetchall()
            for row in rows:
                path = row["input_path"]
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
                c.execute("UPDATE render_jobs SET input_path='' WHERE id=?",
                          (int(row["id"]),))
            return len(rows)

    def cleanup_failed_inputs(self, max_age=24 * 3600):
        """오래된 실패 작업의 입력 파일을 정리한다."""
        cutoff = time.time() - max(0, float(max_age))
        with self._db() as c:
            rows = c.execute(
                "SELECT id, input_path FROM render_jobs WHERE status='failed' "
                "AND input_path<>'' AND finished<?", (cutoff,)).fetchall()
            for row in rows:
                path = row["input_path"]
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
                c.execute("UPDATE render_jobs SET input_path='' WHERE id=?",
                          (int(row["id"]),))
            return len(rows)

    def clear_input_path(self, job_id):
        with self._db() as c:
            c.execute("UPDATE render_jobs SET input_path='' WHERE id=?",
                      (int(job_id),))

    def retry(self, job_id, user):
        """실패한 작업을 같은 입력/설정으로 다시 queued에 넣는다."""
        with self._db() as c:
            row = c.execute(
                "SELECT * FROM render_jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            if row is None or str(row["user"]) != str(user or ""):
                return None
            if row["status"] != "failed":
                return None
            input_path = row["input_path"] or ""
            if not input_path or not os.path.exists(input_path):
                return None
            cur = c.execute(
                "INSERT INTO render_jobs (user, status, created, progress, "
                "input_path, output_path, payload, cancel_requested) "
                "VALUES (?,?,?,?,?,?,?,0)",
                (row["user"], "queued", time.time(), 0.0, input_path,
                 row["output_path"] or "", row["payload"] or "{}"))
            c.execute("UPDATE render_jobs SET status='retried', "
                      "input_path='' WHERE id=?", (int(job_id),))
            return int(cur.lastrowid)

    def worker_loop(self):
        last_cleanup = time.time()
        while not _STOP.is_set():
            try:
                if time.time() - last_cleanup > 3600:
                    self.cleanup_finished_inputs()
                    self.cleanup_failed_inputs()
                    chunk_render.cleanup_stale_chunks()
                    last_cleanup = time.time()
                job = self.claim()
                if job is None:
                    _STOP.wait(1.5)
                    continue
                self.run_job(job)
            except Exception:
                log.exception("렌더 워커 루프 오류")
                _STOP.wait(5)


_QUEUE = RenderQueue()
_WORKERS = []
_SUPERVISOR = None
_WORKER_LOCK = threading.Lock()
_STOP = threading.Event()
_LEASE_OWNER = uuid.uuid4().hex
_LEASE_TTL = 20.0
_LAST_LEASE_FAIL = 0.0
try:
    _STALE_SEC = float(os.environ.get("MOSAIC_JOB_STALE_SEC", "900"))
except (TypeError, ValueError):
    _STALE_SEC = 900.0


def _worker_main():
    while not _STOP.is_set():
        try:
            _QUEUE.worker_loop()
        except Exception:
            log.exception("렌더 워커가 중단됨 - 5초 뒤 재시작")
            _STOP.wait(5)


def _new_worker(index):
    thread = threading.Thread(target=_worker_main, daemon=True,
                              name=f"mosaic-render-worker-{index}")
    thread.start()
    return thread


def _supervisor():
    """리스 heartbeat를 갱신하고 죽은 워커 스레드를 다시 띄운다."""
    while not _STOP.wait(5):
        try:
            if not _QUEUE.heartbeat_worker(_LEASE_OWNER, _LEASE_TTL):
                # 다른 프로세스가 리스를 가져갔다. 중복 렌더를 막기 위해
                # 현재 워커들은 다음 claim 전에 멈춘다.
                log.warning("렌더 큐 리스를 다른 워커가 가져감")
                _STOP.set()
                return
            if (_STALE_SEC > 0
                    and os.environ.get("MOSAIC_QUEUE_WORKER") == "external"):
                stale = _QUEUE.stale_running(_STALE_SEC)
                if stale:
                    # 스레드는 강제 종료할 수 없으므로 워커 프로세스를
                    # 재시작한다. 새 프로세스가 running 작업을 queued로
                    # 복구하고 다시 처리한다.
                    log.error(
                        "렌더 작업 #%s(%s)이 %.0f초 동안 진행되지 않아 "
                        "워커를 재시작합니다",
                        stale.get("id"), stale.get("user"), _STALE_SEC)
                    os._exit(1)
            with _WORKER_LOCK:
                for i, thread in enumerate(list(_WORKERS)):
                    if thread.is_alive():
                        continue
                    log.warning("렌더 워커 스레드 재시작: %s", i + 1)
                    _WORKERS[i] = _new_worker(i + 1)
        except Exception:
            log.exception("렌더 큐 감시 오류")


def _ensure_supervisor():
    global _SUPERVISOR
    if _SUPERVISOR is None or not _SUPERVISOR.is_alive():
        _SUPERVISOR = threading.Thread(target=_supervisor, daemon=True,
                                       name="mosaic-render-supervisor")
        _SUPERVISOR.start()
    return _SUPERVISOR


def start_worker(count=None):
    """워커 스레드를 시작한다. 다른 프로세스가 이미 실행 중이면 양보한다."""
    global _WORKERS, _LAST_LEASE_FAIL
    with _WORKER_LOCK:
        alive = [t for t in _WORKERS if t.is_alive()]
        if alive:
            # 감시 스레드만 죽은 경우 작업을 건드리지 않고 감시만 복구한다.
            _ensure_supervisor()
            return alive
        if not _QUEUE.acquire_worker(_LEASE_OWNER, _LEASE_TTL):
            now = time.time()
            if now - _LAST_LEASE_FAIL > 60:
                log.info("다른 프로세스가 렌더 큐 워커를 실행 중입니다")
                _LAST_LEASE_FAIL = now
            return []
        try:
            _QUEUE.cleanup_finished_inputs()
            _QUEUE.cleanup_failed_inputs()
        except Exception:
            log.exception("큐 입력 정리 실패")
        try:
            # 이전 프로세스가 남긴 running 작업을 되돌린다. 워커 리스를
            # 잡은 뒤에만 호출해야 다른 프로세스의 작업을 건드리지 않는다.
            _QUEUE.recover_running()
        except Exception:
            log.exception("시작 시 running 작업 복구 실패")
        _STOP.clear()
        n = max(1, int(count or _QUEUE.max_parallel))
        _WORKERS = [_new_worker(i + 1) for i in range(n)]
        _ensure_supervisor()
        log.info("렌더 큐 워커 %d개 시작 (최대 동시 %d개)", n,
                 _QUEUE.max_parallel)
        return list(_WORKERS)


def has_active_jobs():
    return _QUEUE.has_active_jobs()


def status_snapshot():
    return _QUEUE.status_snapshot()


def enqueue(**kwargs):
    return _QUEUE.enqueue(**kwargs)


def get(job_id):
    return _QUEUE.get(job_id)


def position(job_id, user=None):
    return _QUEUE.position(job_id, user)


def active_jobs(user):
    return _QUEUE.active_jobs(user)


def cancel(job_id, user):
    return _QUEUE.cancel(job_id, user)


def cancel_admin(job_id):
    return _QUEUE.cancel_admin(job_id)


def cancel_user_jobs(user):
    return _QUEUE.cancel_user_jobs(user)


def cancel_all_active(except_user=None):
    return _QUEUE.cancel_all_active(except_user)


def is_cancel_requested(job_id):
    return _QUEUE.is_cancel_requested(job_id)


def retry(job_id, user):
    return _QUEUE.retry(job_id, user)


def update_progress(job_id, frac):
    return _QUEUE.update_progress(job_id, frac)
