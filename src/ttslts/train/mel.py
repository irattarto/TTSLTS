"""Spectrograms for training, matching the original RVC code (``mel_processing``).

The mel filter bank is built here instead of with librosa (Slaney scale and normalisation, the
same numbers as ``librosa.filters.mel``) to keep the dependencies small.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = math.log(6.4) / 27.0


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    mel = hz / _F_SP
    high = hz >= _MIN_LOG_HZ
    mel[high] = _MIN_LOG_MEL + np.log(hz[high] / _MIN_LOG_HZ) / _LOG_STEP
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    hz = _F_SP * mel
    high = mel >= _MIN_LOG_MEL
    hz[high] = _MIN_LOG_HZ * np.exp(_LOG_STEP * (mel[high] - _MIN_LOG_MEL))
    return hz


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax=None) -> np.ndarray:
    fmax = sr / 2 if fmax is None else fmax
    mel_f = _mel_to_hz(
        np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    )
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


_windows: dict[str, torch.Tensor] = {}
_banks: dict[str, torch.Tensor] = {}


def spectrogram(y: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """(B, T) waveform -> (B, n_fft // 2 + 1, T // hop) linear magnitude spectrogram."""
    key = f"{n_fft}_{y.dtype}_{y.device}"
    if key not in _windows:
        _windows[key] = torch.hann_window(n_fft).to(dtype=y.dtype, device=y.device)
    pad = int((n_fft - hop) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=_windows[key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spec_to_mel(spec: torch.Tensor, n_fft: int, n_mels: int, sr: int) -> torch.Tensor:
    key = f"{sr}_{n_fft}_{n_mels}_{spec.dtype}_{spec.device}"
    if key not in _banks:
        bank = mel_filterbank(sr, n_fft, n_mels)
        _banks[key] = torch.from_numpy(bank).to(dtype=spec.dtype, device=spec.device)
    return torch.log(torch.clamp(torch.matmul(_banks[key], spec), min=1e-5))


def mel_spectrogram(y: torch.Tensor, n_fft: int, n_mels: int, sr: int, hop: int) -> torch.Tensor:
    return spec_to_mel(spectrogram(y, n_fft, hop), n_fft, n_mels, sr)
