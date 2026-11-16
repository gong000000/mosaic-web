"""Modal 배포 설정 (Streamlit + GPU).

로컬 미리보기:
    modal serve modal_app.py

배포:
    modal deploy modal_app.py

- 웹 주소는 배포 출력에 찍히는 https://...modal.run 을 그대로 쓰면 된다.
- users.db / outputs/ 는 Modal Volume(mosaic-data)에 저장되어 재배포·재시작
  후에도 남는다.
- GEMINI_API_KEY 는 Modal Named Secret(mosaic-gemini)에서 읽는다.
  Secret은 워크스페이스에 있으므로 배포 PC마다 .env 를 복사할 필요가 없다.
  .env 자체는 이미지에 포함하지 않는다.
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

APP_NAME = "mosaic-web"
REMOTE_DIR = "/root/app"
DATA_DIR = "/data"
PORT = 8501

ROOT = Path(__file__).parent

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("mosaic-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ultralytics가 opencv-python(GUI)을 의존성으로 함께 설치하므로
    # libGL/libxcb가 필요하다. mediapipe 랜드마크는 libEGL을 쓴다.
    # FFmpeg는 imageio-ffmpeg가 정적 바이너리를 제공하므로 따로 설치하지 않는다.
    .apt_install("libgl1", "libglib2.0-0", "libxcb1", "libegl1", "libgles2")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    # 큰 seg 모델과 pose 모델 가중치를 이미지 빌드 때 받아 둔다(런타임 지연 방지).
    # 받기에 실패해도 빌드는 계속되고, 앱이 nano 모델로 대체한다.
    .run_commands(
        "mkdir -p /opt/weights",
        "python -c \"from ultralytics import YOLO; "
        "YOLO('/opt/weights/yolo11m-seg.pt'); "
        "YOLO('/opt/weights/yolo11s-seg.pt'); "
        "YOLO('/opt/weights/yolo11n-pose.pt')\"",
        # 얼굴 랜드마크 모델(약 3.7MB). 없어도 고정 얼굴형으로 동작한다.
        "python -c \"import urllib.request, os; "
        "p='/opt/weights/face_landmarker.task'; "
        "os.path.exists(p) or urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/"
        "face_landmarker.task', p)\"",
        # 번호판 전용 검출기(YOLO11n, 약 5.5MB, AGPL-3.0).
        # 없어도 VLM 앵커만으로 동작한다.
        "python -c \"import urllib.request, os; "
        "p='/opt/weights/license-plate-finetune-v1n.pt'; "
        "os.path.exists(p) or urllib.request.urlretrieve("
        "'https://huggingface.co/DavidDrill69/"
        "yolov11-license-plate-detection/resolve/main/"
        "license-plate-finetune-v1n.pt', p)\"",
        # 주소·간판 글자 정제(EAST, 약 96MB, Unlicense).
        # 없어도 VLM 박스를 그대로 쓴다.
        "python -c \"import urllib.request, os; "
        "p='/opt/weights/frozen_east_text_detection.pb'; "
        "os.path.exists(p) or urllib.request.urlretrieve("
        "'https://github.com/oyyd/frozen_east_text_detection.pb/"
        "raw/master/frozen_east_text_detection.pb', p)\"",
    )
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_DIR,
        ignore=[
            ".git",
            ".git/**",
            ".venv",
            ".venv/**",
            ".env",
            ".env.*",
            "users.db",
            "users.db-*",
            "outputs",
            "outputs/**",
            "docs",
            "docs/**",
            ".ruff_cache",
            ".ruff_cache/**",
            ".pytest_cache",
            ".pytest_cache/**",
            "__pycache__",
            "**/__pycache__",
            "**/__pycache__/**",
            "*.pyc",
            "*.pyo",
            "*.log",
        ],
    )
)


@app.function(
    image=image,
    # L4를 우선 쓰되, 용량이 없으면 T4로 자동 대체한다.
    # (L4만 고정하면 Modal 용량이 없을 때 앱이 계속 로딩 상태로 남는다.)
    gpu=["L4", "T4"],
    cpu=8.0,
    memory=16384,
    timeout=3600,
    scaledown_window=300,
    # SQLite(users.db)와 앱 내부 MODEL_LOCK 때문에 컨테이너는 하나만 쓴다.
    # 접속자는 이 한 컨테이너를 공유하고, 렌더는 앱이 순서대로 처리한다.
    max_containers=1,
    volumes={DATA_DIR: data_volume},
    # GEMINI_API_KEY 는 Modal Named Secret(mosaic-gemini) 에서 읽는다.
    secrets=[modal.Secret.from_name("mosaic-gemini")],
    env={
        "MOSAIC_DATA_DIR": DATA_DIR,
        "MOSAIC_SKIP_DOTENV": "1",
        "MOSAIC_WEIGHTS_DIR": "/opt/weights",
        "MOSAIC_VLM_RPM": "90",
        "PYTHONUNBUFFERED": "1",
    },
)
@modal.concurrent(max_inputs=100)
# label 을 지정하면 URL 이 https://<워크스페이스>--<label>.<리전>.modal.run
# 형태가 된다. (기본값은 <앱이름>-<함수이름>)
@modal.web_server(PORT, startup_timeout=600, label="mosaic")
def web():
    # Modal 리버스 프록시 뒤에서 Streamlit 파일 업로드가 403으로 막히지
    # 않도록 기본은 CORS/XSRF를 끄고, MOSAIC_ENABLE_XSRF=1 일 때만 켠다.
    enable_xsrf = os.environ.get("MOSAIC_ENABLE_XSRF") == "1"
    cmd = [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(PORT),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "true" if enable_xsrf else "false",
        "--server.enableXsrfProtection",
        "true" if enable_xsrf else "false",
        "--server.fileWatcherType",
        "none",
        "--server.maxUploadSize",
        "500",
    ]
    subprocess.Popen(cmd, cwd=REMOTE_DIR)

    # users.db / 결과 파일이 컨테이너 종료 때 유실되지 않도록 주기적으로
    # 볼륨을 commit 한다.
    def _commit_loop():
        while True:
            time.sleep(30)
            try:
                data_volume.commit()
            except Exception as e:
                print(f"volume commit 실패: {e}", flush=True)

    threading.Thread(target=_commit_loop, daemon=True).start()
