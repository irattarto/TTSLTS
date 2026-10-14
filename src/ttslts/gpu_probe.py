"""Checks, once per model and graphics card, whether a model really runs on the GPU.

A model can load on a GPU and still fail when it runs (Kokoro did on DirectML), return garbage, or
even crash the whole process. So the check runs in a separate process: it runs the model on the
GPU and on the CPU with the same made-up inputs (a few sizes) and compares the results. The verdict
is saved together with the model file, the ONNX Runtime version and the graphics card and driver, so
it is repeated automatically after an update or a new card, and never when nothing changed.

Run as ``python -m ttslts.gpu_probe <model.onnx>`` (the packaged app: ``TTSLTS.exe --gpu-probe``),
it prints one JSON line with the verdict.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ttslts import paths

log = logging.getLogger(__name__)

LENGTHS = (24, 96, 3200)  # sizes for the unknown dimensions (frames, tokens, samples)
AUDIO_LENGTH = 13440  # tried only if none of the above ran: models that take raw audio
GARBAGE_ERROR = 3.0  # relative difference from the CPU result above which the output is garbage
TIMEOUT_S = 240
OK, INCOMPATIBLE, UNKNOWN = "ok", "incompatible", "unknown"


# --- the check itself (runs in its own process) -------------------------------------------------


def make_inputs(session, length: int, rng: np.random.Generator) -> dict:
    """Plausible inputs for any model: small integers, small floats, unknown sizes = ``length``."""
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else length for d in inp.shape]
        if "int64" in inp.type:
            feed[inp.name] = rng.integers(1, 8, shape).astype(np.int64)
        elif "int32" in inp.type:
            feed[inp.name] = rng.integers(1, 8, shape).astype(np.int32)
        elif "bool" in inp.type:
            feed[inp.name] = np.zeros(shape, dtype=bool)
        else:
            dtype = np.float16 if "float16" in inp.type else np.float32
            if int(np.prod(shape)) == 1:  # a scalar such as a speed: keep it sensible
                feed[inp.name] = rng.uniform(0.5, 1.5, shape).astype(dtype)
            else:
                feed[inp.name] = (rng.standard_normal(shape) * 0.1).astype(dtype)
    return feed


def compare(gpu: list, cpu: list) -> str:
    """Empty if the outputs agree well enough, otherwise what is wrong with the GPU's."""
    for g, c in zip(gpu, cpu, strict=False):
        g, c = np.asarray(g), np.asarray(c)
        if g.shape != c.shape:
            return f"output shape {g.shape} differs from the CPU's {c.shape}"
        if g.dtype.kind != "f":
            continue
        g, c = g.astype(np.float64), c.astype(np.float64)
        if not np.isfinite(g).all() and np.isfinite(c).all():
            return "the output contains NaN or infinity"
        scale = float(np.linalg.norm(c)) + 1e-8
        if float(np.linalg.norm(g - c)) / scale > GARBAGE_ERROR:
            return "the output is very different from the CPU's"
    return ""


def run_probe(model_path: str) -> dict:
    import onnxruntime as ort

    from ttslts import onnx_util

    started = time.monotonic()
    chosen = onnx_util.providers(True)
    if chosen == ["CPUExecutionProvider"]:
        return {"verdict": UNKNOWN, "reason": "no GPU provider is available"}
    cpu = ort.InferenceSession(
        model_path, onnx_util.session_options(["CPUExecutionProvider"]), ["CPUExecutionProvider"]
    )
    try:
        gpu = ort.InferenceSession(model_path, onnx_util.session_options(chosen), chosen)
    except Exception as exc:
        return {"verdict": INCOMPATIBLE, "reason": f"it does not load on the GPU: {exc}"[:300]}
    rng = np.random.default_rng(0)
    compared = 0

    def check(length: int) -> dict | None:
        """The verdict if this size shows a problem, ``None`` if fine or not comparable."""
        nonlocal compared
        feed = make_inputs(cpu, length, rng)
        try:
            expected = cpu.run(None, feed)
        except Exception:
            return None  # the made-up input does not suit this model: nothing to compare with
        try:
            got = gpu.run(None, feed)
        except Exception as exc:
            return {
                "verdict": INCOMPATIBLE,
                "reason": f"it fails on the GPU at size {length}: {exc}"[:300],
            }
        problem = compare(got, expected)
        if problem:
            return {"verdict": INCOMPATIBLE, "reason": f"{problem} (size {length})"}
        compared += 1
        return None

    for length in LENGTHS:
        if (found := check(length)) is not None:
            return found
    if not compared and (found := check(AUDIO_LENGTH)) is not None:  # models that take raw audio
        return found
    if not compared:
        return {"verdict": UNKNOWN, "reason": "the made-up inputs did not run on the CPU either"}
    return {"verdict": OK, "reason": "", "seconds": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    result = run_probe(argv[0])
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


# --- using the verdicts (in the app) --------------------------------------------------------------


def enabled() -> bool:
    return not os.environ.get("TTSLTS_NO_GPU_PROBE")


@functools.lru_cache(maxsize=1)
def gpu_signature() -> str:
    """Names and driver versions of the graphics cards, so a new card or driver is re-checked."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    commands = []
    if sys.platform == "win32":
        commands.append(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { $_.Name + ' ' + $_.DriverVersion }",
            ]
        )
    commands.append(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    for command in commands:
        try:
            out = subprocess.run(
                command, capture_output=True, text=True, timeout=30, creationflags=flags
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = " | ".join(sorted(line.strip() for line in out.stdout.splitlines() if line.strip()))
        if out.returncode == 0 and text:
            return text
    return "unknown"


def _cache_file() -> Path:
    return paths.config_dir() / "gpu-probe.json"


def _model_key(path: str) -> str:
    stat = Path(path).stat()
    return f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _load_cache() -> dict:
    import onnxruntime as ort

    try:
        data = json.loads(_cache_file().read_text(encoding="utf-8"))
        if data.get("ort") == ort.__version__ and data.get("gpu") == gpu_signature():
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return {"ort": ort.__version__, "gpu": gpu_signature(), "models": {}}


def _save_cache(data: dict) -> None:
    try:
        _cache_file().parent.mkdir(parents=True, exist_ok=True)
        _cache_file().write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError as exc:
        log.debug("Could not save the GPU check results: %s", exc)


def clear() -> None:
    """Forget all verdicts, so every model is checked again."""
    _cache_file().unlink(missing_ok=True)


def probe_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--gpu-probe"]
    return [sys.executable, "-m", "ttslts.gpu_probe"]


def probe_in_new_process(path: str) -> dict:
    """Run the check in a separate process; a crash there is a verdict, not a crash of the app."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        done = subprocess.run(
            [*probe_command(), path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            creationflags=flags,
        )
    except subprocess.TimeoutExpired:
        return {"verdict": UNKNOWN, "reason": "the check took too long"}
    except OSError as exc:
        return {"verdict": UNKNOWN, "reason": f"the check could not start: {exc}"}
    for line in reversed(done.stdout.splitlines()):
        try:
            result = json.loads(line)
        except ValueError:
            continue
        if isinstance(result, dict) and "verdict" in result:
            return result
    return {
        "verdict": INCOMPATIBLE,
        "reason": f"the GPU check crashed (exit code {done.returncode})",
    }


_VERDICT_LOCK = threading.Lock()


def verdict(path: str, runner=probe_in_new_process, announce=None) -> dict | None:
    """The saved verdict for a model, running the check first if there is none yet.

    ``announce`` is called with a message just before a check that takes a while is started.
    One check at a time: two at once checked the same model twice and the second save of the
    results file dropped what the first had added.
    """
    with _VERDICT_LOCK:
        return _verdict(path, runner, announce)


def _verdict(path: str, runner, announce) -> dict | None:
    if not enabled() or not Path(path).exists():
        return None
    cache = _load_cache()
    key = _model_key(path)
    if key in cache["models"]:
        return cache["models"][key]
    if announce is not None:
        announce(f"Checking {Path(path).name} on your graphics card (done once)…")
    result = runner(path)
    if result.get("verdict") != UNKNOWN:  # unknown is tried again next time
        cache["models"][key] = result
        _save_cache(cache)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
