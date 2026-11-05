"""설치/구성 스모크 테스트.

필수 패키지 import, 프로젝트 모듈 import, 카테고리 설정 일관성,
소스에 남은 merge conflict marker를 확인한다.
실행: .\\.venv\\Scripts\\python.exe scripts\\smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

mods = ["cv2", "numpy", "streamlit", "ultralytics", "torch",
        "imageio_ffmpeg", "openai"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"OK {m}")
    except Exception as e:
        bad.append(m)
        print(f"FAIL {m}: {e}")

project_mods = ["src.config", "src.detect", "src.render", "src.models",
                "src.select", "src.vlm_package", "src.preview",
                "src.pipeline", "src.history", "src.retention"]
for m in project_mods:
    try:
        __import__(m)
        print(f"OK {m}")
    except Exception as e:
        bad.append(m)
        print(f"FAIL {m}: {e}")

try:
    from src.config import CANDS, CATEGORY_TYPE, KO
    assert len(CANDS) == 7, CANDS
    assert set(CATEGORY_TYPE) == set(CANDS), CATEGORY_TYPE
    assert set(KO) == set(CANDS), KO
    print("OK category config")
except Exception as e:
    bad.append("category config")
    print(f"FAIL category config: {e}")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
markers = ("<<<<<<<", "=======", ">>>>>>>")
conflicted = []
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs
               if d not in (".git", ".venv", "__pycache__", "outputs")]
    for name in files:
        if not name.endswith((".py", ".md", ".txt", ".yaml", ".yml")):
            continue
        path = os.path.join(base, name)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if any(line.strip().startswith(marker)
               for line in text.splitlines() for marker in markers):
            conflicted.append(os.path.relpath(path, root))
if conflicted:
    bad.append("conflict markers")
    print("FAIL conflict markers: " + ", ".join(conflicted))
else:
    print("OK no conflict markers")

sys.exit(1 if bad else 0)
