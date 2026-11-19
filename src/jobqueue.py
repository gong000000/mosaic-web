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
    cancel_requested INTEGER NOT NULL DEFAULT 0
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
    def __init__(self, db_path=None, input_dir=None):
        self.db = db_path or os.path.join(OUTPUT_DIR, "jobs.db")
        self.input_dir = input_dir or os.path.join(OUTPUT_DIR, "job_inputs")
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
            c.execute("CREATE INDEX IF NOT EXISTS idx_render_jobs_status "
                      "ON render_jobs(status, id)")

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

    def position(self, job_id):
        """앞에 있는 queued/running 작업 수 + 1."""
        with self._db() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM render_jobs WHERE id<? AND "
                "status IN ('queued','running')",
                (int(job_id),)).fetchone()[0]
        return int(n) + 1

    def active_jobs(self, user):
        """해당 사용자의 queued/running 작업 목록."""
        with self._db() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT id, status, progress, created, started, output_path, "
                "cancel_requested FROM render_jobs WHERE user=? AND status IN "
                "('queued','running') ORDER BY id", (user or "",)).fetchall()]
        return rows

    def cancel(self, job_id, user):
        """queued면 즉시 취소, running이면 취소 요청 플래그를 세운다."""
        with self._db() as c:
            row = c.execute(
                "SELECT id, user, status FROM render_jobs WHERE id=?",
                (int(job_id),)).fetchone()
            if row is None or str(row["user"]) != str(user or ""):
                return False
            if row["status"] == "queued":
                c.execute(
                    "UPDATE render_jobs SET status='cancelled', finished=?, "
                    "progress=0, error='사용자 취소' WHERE id=?",
                    (time.time(), int(job_id)))
            elif row["status"] == "running":
                c.execute(
                    "UPDATE render_jobs SET cancel_requested=1 WHERE id=?",
                    (int(job_id),))
            else:
                return False
        return True

    def is_cancel_requested(self, job_id):
        with self._db() as c:
            row = c.execute(
                "SELECT cancel_requested, status FROM render_jobs WHERE id=?",
                (int(job_id),)).fetchone()
        return bool(row and (row["cancel_requested"]
                             or row["status"] == "cancelled"))

    def claim(self):
        """queued 작업 1개를 원자적으로 running으로 바꾼다."""
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT id FROM render_jobs WHERE status='queued' "
                "AND NOT EXISTS (SELECT 1 FROM render_jobs "
                "WHERE status='running') ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            job_id = int(row["id"])
            c.execute("UPDATE render_jobs SET status='running', started=?, "
                      "progress=0, error='' WHERE id=?",
                      (time.time(), job_id))
            c.commit()
            job = c.execute("SELECT * FROM render_jobs WHERE id=?",
                            (job_id,)).fetchone()
        return dict(job) if job else None

    def update_progress(self, job_id, frac):
        try:
            with self._db() as c:
                c.execute("UPDATE render_jobs SET progress=? WHERE id=?",
                          (min(1.0, max(0.0, float(frac))), int(job_id)))
        except Exception:
            pass

    def finish(self, job_id, status, summary="", error=""):
        with self._db() as c:
            c.execute(
                "UPDATE render_jobs SET status=?, finished=?, progress=?, "
                "summary=?, error=? WHERE id=?",
                (status, time.time(), 1.0 if status == "done" else 0.0,
                 summary, error[:2000], int(job_id)))

    def recover_running(self):
        """이전 프로세스가 남긴 running 작업을 다시 queued로 되돌린다."""
        with self._db() as c:
            c.execute("UPDATE render_jobs SET status='cancelled', finished=?, "
                      "error='서버 재시작으로 중단됨' WHERE status='running' "
                      "AND cancel_requested=1", (time.time(),))
            c.execute("UPDATE render_jobs SET status='queued', started=0, "
                      "progress=0 WHERE status='running'")

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
            stats = plan.render(
                job["input_path"], output_path,
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
                use_cache=False, vlm_call_limit=0,
                cancel_fn=lambda: self.is_cancel_requested(job_id))
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("결과 영상이 만들어지지 않았습니다.")
            if payload.get("strip_audio"):
                strip_audio(output_path)
            selected = payload.get("selected_cats") or []
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
                if job.get("input_path") and os.path.exists(job["input_path"]):
                    os.remove(job["input_path"])
            except OSError:
                pass

    def worker_loop(self):
        self.recover_running()
        while True:
            job = self.claim()
            if job is None:
                time.sleep(1.5)
                continue
            self.run_job(job)


_QUEUE = RenderQueue()
_WORKER = None
_WORKER_LOCK = threading.Lock()


def start_worker():
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return _WORKER
        _WORKER = threading.Thread(target=_QUEUE.worker_loop, daemon=True,
                                   name="mosaic-render-worker")
        _WORKER.start()
        return _WORKER


def enqueue(**kwargs):
    return _QUEUE.enqueue(**kwargs)


def get(job_id):
    return _QUEUE.get(job_id)


def position(job_id):
    return _QUEUE.position(job_id)


def active_jobs(user):
    return _QUEUE.active_jobs(user)


def cancel(job_id, user):
    return _QUEUE.cancel(job_id, user)


def is_cancel_requested(job_id):
    return _QUEUE.is_cancel_requested(job_id)


def update_progress(job_id, frac):
    return _QUEUE.update_progress(job_id, frac)
