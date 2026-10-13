"""Shared helpers for creating ONNX Runtime sessions."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ttslts import paths

log = logging.getLogger(__name__)

_GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
_notify: Callable[[str], None] | None = None


def set_fallback_notifier(callback: Callable[[str], None] | None) -> None:
    """Called with a message whenever a model gives up on the GPU and moves to the CPU."""
    global _notify
    _notify = callback


def _skip_file() -> Path:
    return paths.config_dir() / "gpu-skip.json"


def _skip_list() -> list[str]:
    """Model files known not to run on this machine's GPU (valid for this ONNX Runtime)."""
    import onnxruntime as ort

    try:
        data = json.loads(_skip_file().read_text(encoding="utf-8"))
        return list(data["models"]) if data.get("ort") == ort.__version__ else []
    except (OSError, ValueError, KeyError, AttributeError):
        return []


def _remember_gpu_failure(path: str) -> None:
    import onnxruntime as ort

    models = _skip_list()
    if path not in models:
        models.append(path)
    try:
        _skip_file().parent.mkdir(parents=True, exist_ok=True)
        _skip_file().write_text(
            json.dumps({"ort": ort.__version__, "models": models}), encoding="utf-8"
        )
    except OSError as exc:
        log.debug("Could not save the GPU skip list: %s", exc)


def providers(use_gpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    chosen = [p for p in _GPU_PROVIDERS if use_gpu and p in available]
    chosen.append("CPUExecutionProvider")
    return chosen


class ResilientSession:
    """A GPU session that moves to the CPU for good if a run fails on the GPU.

    GPU runs can fail while loading worked: the card may be busy with another job (such as
    training) or a driver may not support one operator at some input size. The failing call is
    repeated on the CPU, so the app slows down instead of showing an error.
    """

    def __init__(self, path: str, options, session):
        self._path = path
        self._options = options
        self._session = session
        self.on_cpu = False

    def run(self, *args, **kwargs):
        try:
            return self._session.run(*args, **kwargs)
        except Exception as exc:
            if self.on_cpu:
                raise
            self._fall_back(exc)
            return self._session.run(*args, **kwargs)

    def _fall_back(self, exc: Exception) -> None:
        import onnxruntime as ort

        log.warning("GPU run failed for %s (%s); using the CPU from now on", self._path, exc)
        self._session = ort.InferenceSession(
            self._path, sess_options=self._options, providers=["CPUExecutionProvider"]
        )
        self.on_cpu = True
        _remember_gpu_failure(self._path)  # next time it goes straight to the CPU
        if _notify is not None:
            _notify("A model could not run on the GPU and now runs on the CPU (slower)")

    def __getattr__(self, name: str):
        return getattr(self._session, name)


def make_session(path: str, use_gpu: bool):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if use_gpu and path in _skip_list():
        log.info("%s is known not to run on the GPU here; using the CPU", path)
        use_gpu = False
    chosen = providers(use_gpu)
    try:
        session = ort.InferenceSession(path, sess_options=opts, providers=chosen)
    except Exception as exc:  # no usable GPU / driver: run on the CPU instead of failing
        if chosen == ["CPUExecutionProvider"]:
            raise
        log.warning("GPU provider failed for %s (%s); using the CPU", path, exc)
        session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    log.info("Loaded %s with %s", path, session.get_providers())
    if session.get_providers()[0] != "CPUExecutionProvider":
        return ResilientSession(path, opts, session)
    return session
