"""Write float audio to a 16-bit WAV file using only the standard library."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
