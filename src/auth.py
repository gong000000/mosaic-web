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


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        " username TEXT PRIMARY KEY,"
        " salt TEXT NOT NULL,"
        " pw_hash TEXT NOT NULL,"
        " created_at TEXT NOT NULL)"
    )
    return conn


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


def _locked_seconds():
    until = st.session_state.get("login_locked_until", 0)
    return max(0, int(until - time.time()))


def require_login():
    """로그인된 아이디를 반환한다. 로그인 전이면 화면을 그리고 멈춘다."""
    user = st.session_state.get("user")
    if user:
        return user

    st.markdown("## 🔒 개인정보 모자이크")
    st.caption("로그인하면 내 작업 기록이 계정별로 저장됩니다.")
    tab_in, tab_up = st.tabs(["로그인", "회원가입"])

    with tab_in:
        with st.form("login_form"):
            uid = st.text_input("아이디")
            pw = st.text_input("비밀번호", type="password")
            go = st.form_submit_button("로그인")
        if go:
            wait = _locked_seconds()
            if wait:
                st.error(f"로그인 시도가 너무 많습니다. {wait}초 후 다시 시도하세요.")
            elif verify(uid, pw):
                st.session_state["user"] = uid.strip()
                st.session_state["login_fails"] = 0
                st.rerun()
            else:
                fails = st.session_state.get("login_fails", 0) + 1
                st.session_state["login_fails"] = fails
                if fails >= MAX_FAILS:
                    st.session_state["login_locked_until"] = (
                        time.time() + LOCK_SECONDS)
                    st.session_state["login_fails"] = 0
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    with tab_up:
        with st.form("signup_form"):
            nu = st.text_input("새 아이디 (영문/숫자/_ 3~20자)")
            np1 = st.text_input("새 비밀번호 (8자 이상)", type="password")
            np2 = st.text_input("비밀번호 확인", type="password")
            go = st.form_submit_button("가입하기")
        if go:
            if np1 != np2:
                st.error("비밀번호가 서로 다릅니다.")
            else:
                ok, msg = register(nu, np1)
                if ok:
                    st.success("가입 완료! '로그인' 탭에서 로그인하세요.")
                else:
                    st.error(msg)

    st.stop()