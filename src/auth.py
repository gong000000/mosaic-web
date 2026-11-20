"""로그인/회원가입 (Streamlit + SQLite).

- require_login(): 로그인된 아이디(str)를 반환한다. 아니면 로그인 화면을
  띄우고 st.stop().
- register(username, password): (ok, msg) 반환.
- 비밀번호는 PBKDF2-SHA256 + 사용자별 salt로 저장한다(평문 저장 안 함).
- 아이디는 영문/숫자/_ 3~20자만 허용한다. app.py가 결과 파일명을
  '아이디--영상이름.mp4'로 만들어 계정별 기록을 구분하므로, '--'나
  경로 문자가 아이디에 들어가면 안 된다.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing

import streamlit as st

from src.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "users.db")
USER_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")
ITERATIONS = 200_000
MAX_FAILS = 5          # 연속 실패 허용 횟수
LOCK_SECONDS = 60      # 초과 시 잠금 시간
ADMIN_USERS = {u.strip() for u in os.environ.get(
    "MOSAIC_ADMIN_USERS", "").split(",") if u.strip()}


def is_admin(username):
    return str(username or "") in ADMIN_USERS


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        " username TEXT PRIMARY KEY,"
        " salt TEXT NOT NULL,"
        " pw_hash TEXT NOT NULL,"
        " created_at TEXT NOT NULL)"
    )
    # 로그인 실패 횟수는 세션이 아니라 DB에 아이디 기준으로 저장한다.
    # (탭을 새로 열어도 잠금이 초기화되지 않게)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS login_attempts ("
        " username TEXT PRIMARY KEY,"
        " fails INTEGER NOT NULL DEFAULT 0,"
        " locked_until REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_controls ("
        " username TEXT PRIMARY KEY,"
        " kicked_until REAL NOT NULL DEFAULT 0,"
        " reason TEXT NOT NULL DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS site_control ("
        " id INTEGER PRIMARY KEY CHECK (id=1),"
        " locked_until REAL NOT NULL DEFAULT 0,"
        " reason TEXT NOT NULL DEFAULT '')"
    )
    return conn


def invite_required():
    """MOSAIC_INVITE_CODE 환경변수가 있으면 가입 시 초대 코드를 요구한다."""
    return bool(os.environ.get("MOSAIC_INVITE_CODE", "").strip())


def check_invite(code):
    want = os.environ.get("MOSAIC_INVITE_CODE", "").strip()
    if not want:
        return True
    return hmac.compare_digest((code or "").strip().encode("utf-8"),
                               want.encode("utf-8"))


def lock_remaining(username):
    """해당 아이디가 잠겨 있으면 남은 초, 아니면 0."""
    username = (username or "").strip()
    with closing(_conn()) as conn:
        row = conn.execute(
            "SELECT locked_until FROM login_attempts WHERE username = ?",
            (username,)).fetchone()
    if not row:
        return 0
    return max(0, int(row[0] - time.time()))


def kick(username, minutes=30, reason=""):
    """관리자 강퇴. 해당 아이디의 로그인/세션을 일정 시간 막는다."""
    username = (username or "").strip()
    if not username:
        return False
    until = time.time() + max(1, int(minutes)) * 60
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO user_controls (username, kicked_until, reason) "
            "VALUES (?, ?, ?) ON CONFLICT(username) DO UPDATE SET "
            "kicked_until=excluded.kicked_until, reason=excluded.reason",
            (username, until, str(reason or "")[:200]))
    return True


def unkick(username):
    username = (username or "").strip()
    with closing(_conn()) as conn, conn:
        conn.execute("UPDATE user_controls SET kicked_until=0, reason='' "
                     "WHERE username=?", (username,))


def kick_remaining(username):
    username = (username or "").strip()
    if not username:
        return 0
    with closing(_conn()) as conn:
        row = conn.execute(
            "SELECT kicked_until FROM user_controls WHERE username=?",
            (username,)).fetchone()
    return max(0, int((row[0] if row else 0) - time.time()))


def kicked_users():
    now = time.time()
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT username, kicked_until, reason FROM user_controls "
            "WHERE kicked_until>? ORDER BY kicked_until DESC",
            (now,)).fetchall()
    return [{"username": r[0], "remaining": int(r[1] - now),
             "reason": r[2]} for r in rows]


def all_users():
    with closing(_conn()) as conn:
        return [r[0] for r in conn.execute(
            "SELECT username FROM users ORDER BY username").fetchall()]


def lock_site(minutes=30, reason=""):
    until = time.time() + max(1, int(minutes)) * 60
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO site_control (id, locked_until, reason) "
            "VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "locked_until=excluded.locked_until, reason=excluded.reason",
            (until, str(reason or "")[:200]))
    return True


def unlock_site():
    with closing(_conn()) as conn, conn:
        conn.execute("UPDATE site_control SET locked_until=0, reason='' "
                     "WHERE id=1")


def site_lock_remaining():
    with closing(_conn()) as conn:
        row = conn.execute(
            "SELECT locked_until FROM site_control WHERE id=1").fetchone()
    return max(0, int((row[0] if row else 0) - time.time()))


def record_login_result(username, ok):
    """성공이면 실패 횟수 초기화, 실패면 증가시키고 한도 초과 시 잠근다."""
    username = (username or "").strip()[:64]
    with closing(_conn()) as conn, conn:
        if ok:
            conn.execute("DELETE FROM login_attempts WHERE username = ?",
                         (username,))
            return
        row = conn.execute(
            "SELECT fails FROM login_attempts WHERE username = ?",
            (username,)).fetchone()
        fails = (row[0] if row else 0) + 1
        locked = 0.0
        if fails >= MAX_FAILS:
            locked = time.time() + LOCK_SECONDS
            fails = 0
        conn.execute(
            "INSERT INTO login_attempts (username, fails, locked_until)"
            " VALUES (?, ?, ?) ON CONFLICT(username) DO UPDATE SET"
            " fails = excluded.fails, locked_until = excluded.locked_until",
            (username, fails, locked))


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
        ITERATIONS).hex()


def register(username, password):
    """회원가입. (성공 여부, 메시지)."""
    username = (username or "").strip()
    if not USER_RE.match(username):
        return False, "아이디는 영문/숫자/_ 3~20자로 입력하세요."
    if len(password or "") < 8:
        return False, "비밀번호는 8자 이상이어야 합니다."
    if site_lock_remaining():
        return False, "관리자에 의해 신규 가입이 잠시 제한되었습니다."
    salt = secrets.token_hex(16)
    try:
        # closing으로 연결을 닫아야 Windows에서 users.db 파일 잠금이
        # 남지 않는다. with conn은 트랜잭션만 커밋한다.
        with closing(_conn()) as conn, conn:
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?)",
                (username, salt, _hash(password, salt),
                 time.strftime("%Y-%m-%d %H:%M:%S")))
    except sqlite3.IntegrityError:
        return False, "이미 사용 중인 아이디입니다."
    return True, "가입 완료"


def verify(username, password):
    """아이디/비밀번호 확인."""
    username = (username or "").strip()
    with closing(_conn()) as conn:
        row = conn.execute(
            "SELECT salt, pw_hash FROM users WHERE username = ?",
            (username,)).fetchone()
    if not row:
        # 아이디 존재 여부로 시간 차이가 나지 않게 더미 계산
        _hash(password or "", "00" * 16)
        return False
    return hmac.compare_digest(_hash(password or "", row[0]), row[1])


def require_login():
    """로그인된 아이디를 반환한다. 로그인 전이면 화면을 그리고 멈춘다."""
    user = st.session_state.get("user")
    if user:
        remain = kick_remaining(user)
        if not is_admin(user):
            remain = max(remain, site_lock_remaining())
        if remain:
            st.session_state.clear()
            st.session_state["_kick_notice"] = (
                "관리자에 의해 로그아웃되었습니다. "
                f"약 {max(1, remain // 60)}분 후 다시 로그인할 수 있습니다.")
            st.rerun()
        return user

    st.markdown("## 🔒 개인정보 모자이크")
    st.caption("로그인하면 내 작업 기록이 계정별로 저장됩니다.")
    notice = st.session_state.pop("_kick_notice", "")
    if notice:
        st.warning(notice)
    tab_in, tab_up = st.tabs(["로그인", "회원가입"])

    with tab_in:
        with st.form("login_form"):
            uid = st.text_input("아이디")
            pw = st.text_input("비밀번호", type="password")
            go = st.form_submit_button("로그인")
        if go:
            wait = lock_remaining(uid)
            kick_wait = kick_remaining(uid)
            site_wait = 0 if is_admin(uid) else site_lock_remaining()
            if kick_wait or site_wait:
                minutes = max(1, max(kick_wait, site_wait) // 60)
                st.error("관리자에 의해 로그인이 제한되었습니다. "
                         f"약 {minutes}분 후 다시 시도하세요.")
            elif wait:
                st.error(f"로그인 시도가 너무 많습니다. {wait}초 후 다시 시도하세요.")
            elif verify(uid, pw):
                record_login_result(uid, True)
                st.session_state["user"] = uid.strip()
                st.rerun()
            else:
                record_login_result(uid, False)
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    with tab_up:
        with st.form("signup_form"):
            nu = st.text_input("새 아이디 (영문/숫자/_ 3~20자)")
            np1 = st.text_input("새 비밀번호 (8자 이상)", type="password")
            np2 = st.text_input("비밀번호 확인", type="password")
            invite = (st.text_input("초대 코드", type="password")
                      if invite_required() else "")
            go = st.form_submit_button("가입하기")
        if go:
            if not check_invite(invite):
                st.error("초대 코드가 올바르지 않습니다.")
            elif np1 != np2:
                st.error("비밀번호가 서로 다릅니다.")
            else:
                ok, msg = register(nu, np1)
                if ok:
                    st.success("가입 완료! '로그인' 탭에서 로그인하세요.")
                else:
                    st.error(msg)

    st.stop()
