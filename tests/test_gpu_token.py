# -*- coding: utf-8 -*-
import numpy as np
import pytest
from fastapi import HTTPException

from src import gpu_client
from src.gpu_server import _check_token


def test_worker_token_required(monkeypatch):
    monkeypatch.delenv("MOSAIC_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("MOSAIC_WORKER_ALLOW_ANON", raising=False)
    with pytest.raises(HTTPException) as exc:
        _check_token("")
    assert exc.value.status_code == 503

    monkeypatch.setenv("MOSAIC_WORKER_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        _check_token("wrong")
    assert exc.value.status_code == 401
    _check_token("secret-token")


def test_worker_token_anon_is_explicit(monkeypatch):
    monkeypatch.delenv("MOSAIC_WORKER_TOKEN", raising=False)
    monkeypatch.setenv("MOSAIC_WORKER_ALLOW_ANON", "1")
    _check_token("")


def test_gpu_client_records_disabled_fallback(monkeypatch):
    monkeypatch.setenv("MOSAIC_GPU_WORKER", "")
    gpu_client.reset_stats()
    frame = np.zeros((16, 16, 3), np.uint8)
    assert gpu_client.remote_face_masks(frame, [[0, 0, 8, 8]]) is None
    snap = gpu_client.snapshot()
    assert snap["enabled"] is False
    assert snap["fallback"] is True
    assert snap["disabled"] >= 1
