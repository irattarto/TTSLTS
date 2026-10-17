"""Text to speech with an F5-TTS voice, read by ``f5_worker`` in the F5 environment.

F5 needs PyTorch, which the app itself does not have, so the voice lives in a separate process
that stays loaded between readings. The text is split into sentences here, with the same pauses
as Piper (``piper_engine.split_with_pauses``); the process reads one sentence at a time.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ttslts import paths

log = logging.getLogger(__name__)


class F5Error(RuntimeError):
    pass


class F5Voice:
    """One loaded F5 voice in its own process. ``command`` is how to start the worker's Python."""

    def __init__(self, command: list[str], env: dict[str, str], files: dict[str, Path], gpu: bool):
        self.files = files
        self.sample_rate = 24000
        self._lock = threading.Lock()
        self._tmp = Path(tempfile.mkdtemp(prefix="ttslts-f5-"))
        log_path = paths.config_dir() / "f5.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [*command, "-u", "-m", "ttslts.tts.f5_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            creationflags=flags,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "HF_HOME": str(paths.models_dir() / "hf"),
                **env,
            },
        )
        reply = self._ask(
            op="load",
            model=str(files["model"]),
            vocab=str(files["vocab"]),
            ref_audio=str(files["ref_audio"]),
            ref_text=str(files["ref_text"]),
            device="cuda" if gpu else "cpu",
            whisper_dir=str(paths.models_dir() / "whisper"),
        )
        self.device = reply.get("device", "cpu")

    def _ask(self, **msg) -> dict:
        proc = self._proc
        if proc.poll() is not None:
            raise F5Error(f"The F5 voice process has stopped. Details: {self._log.name}")
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise F5Error(f"The F5 voice process has stopped. Details: {self._log.name}")
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if reply.get("type") == "error":
                    raise F5Error(f"F5 voice: {reply.get('message')}. Details: {self._log.name}")
                return reply
        except OSError as exc:
            raise F5Error(f"The F5 voice process does not answer: {exc}") from exc

    def say(self, sentence: str, speed: float = 1.0, seed: int = 0) -> np.ndarray:
        out = self._tmp / f"s{seed}.npy"
        with self._lock:
            reply = self._ask(op="say", text=sentence, speed=speed, seed=seed, out=str(out))
        audio = np.load(reply["out"])
        out.unlink(missing_ok=True)
        self.sample_rate = int(reply["sr"])
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(
        self, text: str, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> tuple[np.ndarray, int]:
        from ttslts.tts.hu_text import normalize_hungarian
        from ttslts.tts.piper_engine import split_with_pauses

        parts: list[np.ndarray] = []
        pause = 0.0
        for n, (sentence, pause_after) in enumerate(split_with_pauses(normalize_hungarian(text))):
            if cancelled and cancelled():
                break
            audio = self.say(sentence, speed, seed=n)
            if not audio.size:
                continue
            if parts:
                parts.append(np.zeros(int(pause * self.sample_rate), dtype=np.float32))
            parts.append(audio)
            pause = pause_after
        if not parts:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        audio = np.concatenate(parts)
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio * (0.9 / peak)
        return audio.astype(np.float32), self.sample_rate

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
        self._log.close()
        for f in self._tmp.glob("*"):
            f.unlink(missing_ok=True)
        try:
            self._tmp.rmdir()
        except OSError:
            pass
