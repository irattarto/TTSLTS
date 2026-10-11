"""Shared helpers for creating ONNX Runtime sessions."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def providers(use_gpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    chosen = [p for p in _GPU_PROVIDERS if use_gpu and p in available]
    chosen.append("CPUExecutionProvider")
    return chosen


def make_session(path: str, use_gpu: bool):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(path, sess_options=opts, providers=providers(use_gpu))
    log.info("Loaded %s with %s", path, session.get_providers())
    return session
