"""Write float audio to a 16-bit WAV file (standard library) or an MP3 file (soundfile)."""

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


def write_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Save as MP3 when the name ends in .mp3, otherwise as WAV."""
    if Path(path).suffix.lower() != ".mp3":
        write_wav(path, audio, sr)
        return
    import soundfile as sf  # libsndfile 1.1+ writes MP3 (LAME inside)

    data = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), data, int(sr), format="MP3")
