import sys

mods = ["cv2", "numpy", "streamlit", "ultralytics", "imageio_ffmpeg"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"OK {m}")
    except Exception as e:
        bad.append(m)
        print(f"FAIL {m}: {e}")
sys.exit(1 if bad else 0)
