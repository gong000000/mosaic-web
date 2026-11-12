"""로그인/회원가입 검사.

실제 users.db를 건드리지 않도록 임시 DB로 바꿔서 검증한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\authtest.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import auth


def _with_temp_db(fn):
    root = tempfile.mkdtemp(prefix="mosaic_auth_")
    old = auth.DB_PATH
    auth.DB_PATH = os.path.join(root, "users.db")
    try:
        return fn()
    finally:
        auth.DB_PATH = old
        for name in os.listdir(root):
            try:
                os.unlink(os.path.join(root, name))
            except OSError:
                pass
        os.rmdir(root)


def _run():
    ok, _ = auth.register("ab", "12345678")
    assert not ok, "짧은 아이디가 통과됨"
    ok, _ = auth.register("tester", "short")
    assert not ok, "짧은 비밀번호가 통과됨"
    ok, _ = auth.register("tester", "pass1234")
    assert ok, "정상 가입 실패"
    ok, _ = auth.register("tester", "pass1234")
    assert not ok, "중복 아이디가 통과됨"
    assert auth.verify("tester", "pass1234"), "정상 로그인 실패"
    assert not auth.verify("tester", "wrongpass"), "잘못된 비밀번호 통과"
    assert not auth.verify("nobody", "pass1234"), "없는 아이디 통과"

    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(auth.DB_PATH)) as conn:
        row = conn.execute(
            "SELECT salt, pw_hash FROM users WHERE username='tester'"
        ).fetchone()
    assert row and row[1] != "pass1234", "비밀번호가 평문 저장됨"
    assert len(row[0]) == 32 and len(row[1]) == 64, "해시 형식 오류"


def _login_screen():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(
        os.path.join(os.path.dirname(__file__), "..", "app.py"),
        default_timeout=120)
    at.run()
    text = " ".join(str(x.value) for x in at.markdown)
    text += " ".join(str(x.value) for x in at.caption)
    assert "로그인" in text or "회원가입" in text, text
    assert not at.file_uploader, "로그인 전에 업로드 UI가 노출됨"
    assert not [e for e in at.exception], [str(e.value) for e in at.exception]


def main():
    _with_temp_db(_run)
    print("OK register/verify")
    _with_temp_db(_login_screen)
    print("OK login-screen")
    print("AUTHTEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
