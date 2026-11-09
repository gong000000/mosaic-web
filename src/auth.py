"""간단한 회원가입/로그인 (SQLite + PBKDF2 해시)."""
import hashlib
import hmac
import os
import re
import sqlite3

import streamlit as st

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, salt BLOB NOT NULL, "
        "pw_hash BLOB NOT NULL)")
    return conn


def _hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 200_000)


def register(username, password):
    if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):
        return False, "아이디는 영문/숫자/_ 3~20자로 입력하세요."
    if len(password) < 8:
        return False, "비밀번호는 8자 이상이어야 합니다."
    salt = os.urandom(16)
    conn = _conn()
    try:
        conn.execute("INSERT INTO users VALUES (?, ?, ?)",
                     (username, salt, _hash(password, salt)))
        conn.commit()
    except sqlite3.IntegrityError:
        return False, "이미 사용 중인 아이디입니다."
    finally:
        conn.close()
    return True, ""


def verify(username, password):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT salt, pw_hash FROM users WHERE username=?",
            (username,)).fetchone()
    finally:
        conn.close()
    if not row:
        return False
    return hmac.compare_digest(_hash(password, row[0]), row[1])


def require_login():
    """로그인된 아이디를 돌려준다. 아니면 로그인 화면을 그리고 멈춘다."""
    user = st.session_state.get("user")
    if user:
        return user

    st.title("🔒 개인정보 모자이크")
    tab_in, tab_up = st.tabs(["로그인", "회원가입"])
    with tab_in:
        with st.form("login_form"):
            u = st.text_input("아이디")
            p = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인"):
                if verify(u.strip(), p):
                    st.session_state["user"] = u.strip()
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
    with tab_up:
        with st.form("signup_form"):
            nu = st.text_input("새 아이디")
            np1 = st.text_input("새 비밀번호 (8자 이상)", type="password")
            np2 = st.text_input("비밀번호 확인", type="password")
            if st.form_submit_button("가입하기"):
                if np1 != np2:
                    st.error("비밀번호가 서로 다릅니다.")
                else:
                    ok, msg = register(nu.strip(), np1)
                    if ok:
                        st.success("가입 완료! '로그인' 탭에서 로그인하세요.")
                    else:
                        st.error(msg)
    st.stop()