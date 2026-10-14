"""Pitch extraction with the original RVC project's ``rmvpe.onnx``.

That export takes a log-mel spectrogram ``(1, 128, frames)`` and returns pitch salience
``(1, frames, 360)``. The mel front end and the salience-to-Hz decoding follow RVC's
``infer/lib/rmvpe.py``: 16 kHz audio, 1024-point Hann STFT, 10 ms hop, 128 HTK mel bands
between 30 Hz and 8 kHz, ``log(max(x, 1e-5))``.
"""

from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 1024
HOP = 160
N_MELS = 128
FMIN = 30.0
FMAX = 8000.0
CLAMP = 1e-5
BINS = 360
_CENTS = np.pad(20 * np.arange(BINS) + 1997.3794084376191, (4, 4))


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """HTK mel filterbank with Slaney area normalisation (``librosa.filters.mel(htk=True)``)."""
    fft_freqs = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : N_MELS + 2] - mel_f[:N_MELS]))[:, None]
    return weights.astype(np.float32)


_MEL = mel_filterbank()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def mel_spectrogram(audio16k: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram ``(128, 1 + len // 160)`` of 16 kHz mono audio."""
    audio = np.asarray(audio16k, dtype=np.float32)
    frames = 1 + len(audio) // HOP
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    magnitude = np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)).astype(np.float32)
    mel = _MEL @ magnitude.T
    return np.log(np.maximum(mel, CLAMP)).astype(np.float32)


def decode(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Salience ``(frames, 360)`` -> f0 in Hz per frame (0 where unvoiced)."""
    salience = np.asarray(salience, dtype=np.float32).reshape(-1, BINS)
    center = np.argmax(salience, axis=1) + 4
    padded = np.pad(salience, ((0, 0), (4, 4)))
    index = center[:, None] + np.arange(-4, 5)[None, :]
    weights = np.take_along_axis(padded, index, axis=1)
    total = weights.sum(axis=1)
    cents = np.divide(
        (weights * _CENTS[index]).sum(axis=1), total, out=np.zeros_like(total), where=total > 0
    )
    cents = np.where(padded.max(axis=1) <= threshold, 0.0, cents)
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    f0[f0 == 10.0] = 0.0
    return f0.astype(np.float32)


class RmvpeMelPitch:
    """Wraps the official ``rmvpe.onnx``; same call signature as ``rvc.RmvpePitch``."""

    def __init__(self, session, threshold: float = 0.03):
        self.session = session
        self.threshold = threshold
        self._name = session.get_inputs()[0].name

    def __call__(self, audio16k: np.ndarray) -> np.ndarray:
        mel = mel_spectrogram(audio16k)
        n = mel.shape[1]
        # The time axis must be a multiple of 32. Mirrored frames rather than zeros (which in
        # log-mel are a loud sound): at the end of a live window, where the newest frames are,
        # voiced/unvoiced then agrees with the whole recording twice as often (8.8 % instead
        # of 17.8 % disagreeing in the last 60 ms, measured).
        pad = 32 * ((n - 1) // 32 + 1) - n
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="reflect" if n > 1 else "edge")
        out = self.session.run(None, {self._name: mel[None].astype(np.float32)})[0]
        return decode(np.asarray(out).reshape(-1, BINS)[:n], self.threshold)


def is_mel_model(session) -> bool:
    """True for the official mel-input export, False for waveform-input exports."""
    shape = session.get_inputs()[0].shape
    return len(shape) == 3 and shape[1] == N_MELS
