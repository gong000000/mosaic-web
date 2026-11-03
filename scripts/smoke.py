"""설치 확인용 스모크 테스트. 필수 패키지가 모두 import되는지 본다.
실행: .\\.venv\\Scripts\\python.exe scripts\\smoke.py
"""
import sys

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
sys.exit(1 if bad else 0)
