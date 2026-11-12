"""Modal 배포 설정 (Streamlit + GPU).

로컬 미리보기:
    modal serve modal_app.py

배포:
    modal deploy modal_app.py

- 웹 주소는 배포 출력에 찍히는 https://...modal.run 을 그대로 쓰면 된다.
- users.db / outputs/ 는 Modal Volume(mosaic-data)에 저장되어 재배포·재시작
  후에도 남는다.
- GEMINI_API_KEY 는 로컬 .env 에서 읽어 Modal Secret 으로 전달한다.
  .env 자체는 이미지에 포함하지 않는다.
"""
import subprocess
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
    # opencv-python 은 libGL/libglib 가 필요하다. FFmpeg 는 imageio-ffmpeg 가
    # 정적 바이너리를 제공하므로 따로 설치하지 않는다.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
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
    gpu="T4",
    cpu=4.0,
    memory=8192,
    timeout=3600,
    scaledown_window=300,
    # SQLite(users.db)와 앱 내부 MODEL_LOCK 때문에 컨테이너는 하나만 쓴다.
    # 접속자는 이 한 컨테이너를 공유하고, 렌더는 앱이 순서대로 처리한다.
    max_containers=1,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_dotenv(ROOT)],
    env={
        "MOSAIC_DATA_DIR": DATA_DIR,
        "MOSAIC_SKIP_DOTENV": "1",
        "PYTHONUNBUFFERED": "1",
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=600)
def web():
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
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--server.fileWatcherType",
        "none",
        "--server.maxUploadSize",
        "500",
    ]
    subprocess.Popen(cmd, cwd=REMOTE_DIR)
