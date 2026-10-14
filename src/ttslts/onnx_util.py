"""Shared helpers for creating ONNX Runtime sessions."""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
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


# Errors that say the card is busy or out of memory, not that the model cannot run there.
_TRANSIENT = (
    "out of memory",
    "outofmemory",
    "8007000e",
    "887a0005",
    "887a0006",
    "887a0007",
    "device removed",
    "device hung",
    "device reset",
    "busy",
)
CPU_BACKOFF_S = 30.0  # after a failed GPU run, stay on the CPU this long, then try the GPU again
SKIP_AFTER = 2  # a model is only given up on after the same kind of failure this many times


def is_transient(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(key in message for key in _TRANSIENT)


def _skip_file() -> Path:
    return paths.config_dir() / "gpu-skip.json"


def _load_skip() -> dict:
    """Failure counts per model file, valid for this ONNX Runtime version only."""
    import onnxruntime as ort

    try:
        data = json.loads(_skip_file().read_text(encoding="utf-8"))
        if data.get("ort") == ort.__version__ and isinstance(data.get("failures"), dict):
            return data["failures"]
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _skip_list() -> list[str]:
    """Model files that failed on the GPU often enough to run on the CPU from now on."""
    return [path for path, count in _load_skip().items() if count >= SKIP_AFTER]


def _record_gpu_failure(path: str) -> int:
    import onnxruntime as ort

    failures = _load_skip()
    failures[path] = failures.get(path, 0) + 1
    try:
        _skip_file().parent.mkdir(parents=True, exist_ok=True)
        _skip_file().write_text(
            json.dumps({"ort": ort.__version__, "failures": failures}), encoding="utf-8"
        )
    except OSError as exc:
        log.debug("Could not save the GPU failure counts: %s", exc)
    return failures[path]


def clear_gpu_failures() -> None:
    """Forget which models failed on the GPU, so every model tries it again."""
    from ttslts import gpu_probe

    _skip_file().unlink(missing_ok=True)
    gpu_probe.clear()


# Everything a GPU session does goes through this lock, one thread at a time: running, being
# created and being freed. Two threads running DirectML sessions at once crash the program
# with an access violation inside ONNX Runtime (reproduced). Re-entrant, because a session may
# be freed by the garbage collector in a thread that already holds it.
GPU_LOCK = threading.RLock()

# DirectML (ONNX Runtime 1.24.4) breaks every GPU session that was created *before* a session
# that is freed: their next run is an access violation, even on one thread and with nothing
# else running (reproduced; sessions created after the freed one are fine). Text to speech hit
# it when its voice model was released after a minute and Piper spoke again. So sessions are
# numbered as they are made, and one that is older than a freed one is made anew before it runs.
_made = itertools.count()
_freed_newest = -1  # number of the newest GPU session freed so far


def providers(use_gpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    chosen = [p for p in _GPU_PROVIDERS if use_gpu and p in available]
    chosen.append("CPUExecutionProvider")
    return chosen


class ResilientSession:
    """A GPU session that uses the CPU for a while if a run fails on the GPU.

    GPU runs can fail even though loading worked: the card may be busy or out of memory (for
    example while training), or a driver may not support one operator. The failing call is
    repeated on the CPU, and the GPU is tried again after ``CPU_BACKOFF_S`` seconds. A failure
    that is not about memory or a busy card is counted; once it has happened ``SKIP_AFTER`` times
    for a model, that model goes straight to the CPU from then on (until the counts are cleared).

    GPU runs are one at a time (``GPU_LOCK``), for every session of the program together, and
    a session that a freed newer one has broken is made anew first (see ``_freed_newest``).
    ``source`` is the model file or its bytes, when ``path`` is only a name for messages.
    """

    def __init__(self, path: str, options, session, source=None):
        self._path = path
        self._source = path if source is None else source
        self._options = options
        self._providers = session.get_providers()
        self._session = session
        self._number = next(_made)
        self._cpu = None
        self._cpu_until = 0.0

    def __del__(self):
        global _freed_newest
        session = self.__dict__.get("_session")
        if session is None:
            return
        try:
            with GPU_LOCK:  # freed while no other thread uses the card
                _freed_newest = max(_freed_newest, self._number)
                self.__dict__["_session"] = None
                del session
        except Exception:  # at interpreter exit the module may already be gone
            pass

    def _renew_if_broken(self) -> None:
        """Make the GPU session anew if a newer one was freed since it was made."""
        import onnxruntime as ort

        if self._number >= _freed_newest:
            return
        log.info("Making %s anew on the GPU (a newer session was freed)", self._path)
        fresh = ort.InferenceSession(self._source, self._options, providers=self._providers)
        self._session = fresh  # the broken one is older than the fresh one, so freeing it is safe
        self._number = next(_made)

    @property
    def on_cpu(self) -> bool:
        return time.monotonic() < self._cpu_until

    def _cpu_session(self):
        import onnxruntime as ort

        if self._cpu is None:
            self._cpu = ort.InferenceSession(
                self._source, sess_options=self._options, providers=["CPUExecutionProvider"]
            )
        return self._cpu

    def run(self, *args, **kwargs):
        if self.on_cpu:
            return self._cpu_session().run(*args, **kwargs)
        try:
            with GPU_LOCK:
                self._renew_if_broken()
                return self._session.run(*args, **kwargs)
        except Exception as exc:
            transient = is_transient(exc)
            log.warning("GPU run failed for %s (%s); trying the CPU", self._path, exc)
            result = self._cpu_session().run(*args, **kwargs)  # raises if the input is at fault
            count = 0 if transient else _record_gpu_failure(self._path)  # only the GPU failed
            self._cpu_until = time.monotonic() + CPU_BACKOFF_S
            log.warning("Using the CPU for %.0f s (GPU failure count %d)", CPU_BACKOFF_S, count)
            if _notify is not None:
                _notify(
                    "The graphics card is busy or out of memory: using the CPU for a moment"
                    if transient
                    else "A model could not run on the GPU: using the CPU (slower)"
                )
            return result

    def __getattr__(self, name: str):
        if name.startswith("_"):  # not set up (yet): no endless lookup through _session
            raise AttributeError(name)
        return getattr(self._session, name)


def session_options(chosen: list[str]):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if "DmlExecutionProvider" in chosen:
        # Required by the DirectML provider
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def make_session(path: str, use_gpu: bool):
    import onnxruntime as ort

    if use_gpu and path in _skip_list():
        log.info("%s is known not to run on the GPU here; using the CPU", path)
        use_gpu = False
    chosen = providers(use_gpu)
    if chosen != ["CPUExecutionProvider"]:
        from ttslts import gpu_probe

        found = gpu_probe.verdict(path, announce=_notify)  # once per model and graphics card
        if found and found["verdict"] == gpu_probe.INCOMPATIBLE:
            log.warning("%s does not work on this GPU (%s); using the CPU", path, found["reason"])
            chosen = ["CPUExecutionProvider"]
    opts = session_options(chosen)
    try:
        if chosen == ["CPUExecutionProvider"]:
            session = ort.InferenceSession(path, sess_options=opts, providers=chosen)
        else:
            with GPU_LOCK:  # creating one also works on the card (see GPU_LOCK)
                session = ort.InferenceSession(path, sess_options=opts, providers=chosen)
    except Exception as exc:  # no usable GPU / driver: run on the CPU instead of failing
        if chosen == ["CPUExecutionProvider"]:
            raise
        log.warning("GPU provider failed for %s (%s); using the CPU", path, exc)
        opts = session_options(["CPUExecutionProvider"])
        session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    log.info("Loaded %s with %s", path, session.get_providers())
    if session.get_providers()[0] != "CPUExecutionProvider":
        return ResilientSession(path, opts, session)
    return session
