"""Turns recordings into training examples: clean, cut at pauses, then extract features.

Each example is a short clip of the voice with three files next to each other in the work
folder: the audio at the voice's sample rate, ContentVec features (what is said) and pitch (how
it is said), both in the 10 ms frames the RVC model uses. No PyTorch is needed for this part.
"""

from __future__ import annotations

import logging
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ttslts.audio.dsp import f0_to_coarse, resample, to_mono

log = logging.getLogger(__name__)

AUDIO_FILTER = "Audio (*.mp3 *.wav *.flac *.ogg *.m4a);;All files (*)"
CLIP_S = 3.7  # length of one training example
OVERLAP_S = 0.3
MIN_CLIP_S = 1.0
SILENCE_DB = -45.0  # quieter than this counts as a pause
MIN_PAUSE_S = 0.4
HIGHPASS_HZ = 48.0
HOP_16K = 160


class AudioError(ValueError):
    pass


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a recording (mp3, wav, flac, ogg) as mono float32 and its sample rate."""
    path = Path(path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise AudioError("Reading mp3 and other formats needs the soundfile package") from None
        data, sr = _read_wav(path)
    except Exception as exc:  # soundfile raises its own error types
        raise AudioError(f"Could not read {path.name}: {exc}") from exc
    return to_mono(data), int(sr)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise AudioError(f"{path.name}: only 16-bit WAV is supported without soundfile")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm.reshape(-1, ch), sr


def highpass(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(5, HIGHPASS_HZ, btype="high", fs=sr)
    return filtfilt(b, a, audio).astype(np.float32)


def split_at_pauses(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Cut the recording in the middle of every pause longer than ``MIN_PAUSE_S``."""
    frame = max(1, int(sr * 0.02))
    n = len(audio) // frame
    if n == 0:
        return [audio]
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    quiet = 20 * np.log10(rms) < SILENCE_DB
    cuts: list[int] = []
    start = None
    for i, q in enumerate([*quiet, False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * frame >= MIN_PAUSE_S * sr:
                cuts.append(((start + i) // 2) * frame)
            start = None
    edges = [0, *cuts, len(audio)]
    return [audio[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def clips(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Cut a pause-free stretch into overlapping clips of about ``CLIP_S`` seconds."""
    size, step = int(CLIP_S * sr), int((CLIP_S - OVERLAP_S) * sr)
    if len(audio) <= size + int(OVERLAP_S * sr):
        pieces = [audio]
    else:
        pieces = [audio[i : i + size] for i in range(0, len(audio) - int(OVERLAP_S * sr), step)]
    return [p for p in pieces if len(p) >= MIN_CLIP_S * sr]


def normalise(clip: np.ndarray) -> np.ndarray | None:
    """Level a clip the way RVC does; ``None`` if it clips badly (a damaged recording)."""
    peak = float(np.abs(clip).max())
    if peak > 2.5 or peak < 1e-4:
        return None
    return (clip / peak * 0.675 + 0.25 * clip).astype(np.float32)


def prepare_audio(
    files: list[str],
    sample_rate: int,
    work: Path,
    progress: Callable[[float], None] | None = None,
) -> int:
    """Write ``wavs/<n>.npy`` clips (float32 at ``sample_rate``); returns how many."""
    out = work / "wavs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npy"):
        old.unlink()
    count = 0
    for k, name in enumerate(files):
        audio, sr = load_audio(name)
        if sr != sample_rate:
            audio = resample(audio, sr, sample_rate)
        audio = highpass(audio, sample_rate)
        for stretch in split_at_pauses(audio, sample_rate):
            for clip in clips(stretch, sample_rate):
                clip = normalise(clip)
                if clip is not None:
                    np.save(out / f"{count}.npy", clip)
                    count += 1
        if progress:
            progress((k + 1) / len(files))
    if count == 0:
        raise AudioError(
            "No usable speech was found in the recordings (too short, silent, or distorted)."
        )
    return count


def extract_features(
    work: Path,
    sample_rate: int,
    contentvec: Callable[[np.ndarray], np.ndarray],
    pitch_fn: Callable[[np.ndarray], np.ndarray],
    progress: Callable[[float], None] | None = None,
) -> None:
    """Write ``feats/<n>.npy`` (T, 768, 50 fps) and ``f0/<n>.npy`` + ``f0nsf/<n>.npy`` (100 fps)."""
    wavs = sorted((work / "wavs").glob("*.npy"), key=lambda p: int(p.stem))
    for sub in ("feats", "f0", "f0nsf"):
        (work / sub).mkdir(exist_ok=True)
    for k, path in enumerate(wavs):
        audio16k = resample(np.load(path), sample_rate, 16000)
        feats = contentvec(audio16k)[0]
        f0 = np.asarray(pitch_fn(audio16k), dtype=np.float32)
        n = min(len(f0), len(audio16k) // HOP_16K + 1)
        np.save(work / "feats" / path.name, feats.astype(np.float32))
        np.save(work / "f0nsf" / path.name, f0[:n])
        np.save(work / "f0" / path.name, f0_to_coarse(f0[:n]))
        if progress:
            progress((k + 1) / len(wavs))
