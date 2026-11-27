"""Separate Modal GPU worker for SAM 2.1 / face parsing / OCR.

Deploy this file separately from the Streamlit app::

    modal deploy modal_worker.py

Then set ``MOSAIC_GPU_WORKER`` in the Streamlit deployment to the worker URL.
The main app keeps its MediaPipe/mesh fallback, so a worker outage never leaves
a face unmasked; it simply falls back to the local path.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "mosaic-refiner"
ROOT = Path(__file__).parent

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libxcb1", "libegl1", "libgles2",
                 "ffmpeg")
    .pip_install_from_requirements(str(ROOT / "requirements-worker.txt"))
    # SAM 2 is a research repo; install without dependencies so it cannot
    # replace the project's pinned OpenCV/Torch wheels.
    .run_commands(
        "pip install --no-deps "
        "'git+https://github.com/facebookresearch/sam2.git'",
        "mkdir -p /opt/weights /opt/hf",
        "python -c \"import urllib.request, os; "
        "p='/opt/weights/sam2.1_hiera_tiny.pt'; "
        "os.path.exists(p) or urllib.request.urlretrieve("
        "'https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/"
        "sam2.1_hiera_tiny.pt', p)\"",
        "HF_HOME=/opt/hf python -c \"from huggingface_hub import "
        "snapshot_download; snapshot_download('jonathandinu/face-parsing')\"",
    )
    .add_local_dir(
        str(ROOT / "src"),
        remote_path="/root/app/src",
        ignore=["**/__pycache__", "**/__pycache__/**", "*.pyc"],
    )
    .add_local_file(str(ROOT / "requirements-worker.txt"),
                    remote_path="/root/app/requirements-worker.txt")
)

_worker_env = {
    "MOSAIC_GPU_WORKER": "",
    "MOSAIC_SAM2_CKPT": "/opt/weights/sam2.1_hiera_tiny.pt",
    "MOSAIC_SAM2_CONFIG": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "MOSAIC_WEIGHTS_DIR": "/opt/weights",
    "HF_HOME": "/opt/hf",
    "PYTHONPATH": "/root/app",
    "PYTHONUNBUFFERED": "1",
}


@app.function(
    image=image,
    gpu=["L4", "T4", "A10G", "L40S", "A100"],
    cpu=4.0,
    memory=12288,
    timeout=1800,
    scaledown_window=300,
    max_containers=2,
    env=_worker_env,
)
@modal.asgi_app(label="refiner")
def refiner():
    from src.gpu_server import app as fastapi_app
    return fastapi_app


@app.function(image=modal.Image.debian_slim(), schedule=modal.Period(minutes=10),
              timeout=60, max_containers=1)
def keep_warm():
    """Optional: keep a warm worker while a long render is running.

    The queue worker itself is not modified; this function only pings the
    refiner health endpoint.  Set MOSAIC_REFINER_URL in the Modal secret/env
    before enabling it.
    """
    import urllib.request
    url = os.environ.get("MOSAIC_REFINER_URL", "").rstrip("/")
    if not url:
        return "no refiner url"
    with urllib.request.urlopen(url + "/health", timeout=20) as resp:
        return resp.read(200).decode("utf-8", "replace")
