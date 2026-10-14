"""Small signal-processing helpers used by the voice changer and TTS paths."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

# RVC's coarse pitch quantisation range (see RVC's pipeline: f0_min / f0_max).
F0_MIN = 50.0
F0_MAX = 1100.0
_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)
HIGHPASS_HZ = 48.0  # rumble filter used on training audio and on the microphone


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_from == sr_to or x.size == 0:
        return x
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def highpass_filter(sr: int, hz: float = HIGHPASS_HZ):
    """A streaming high-pass filter: call it block after block, it keeps its own state."""
    from scipy.signal import butter, sosfilt, sosfilt_zi

    sos = butter(5, hz, btype="high", fs=sr, output="sos")
    state = sosfilt_zi(sos) * 0.0

    def run(block: np.ndarray) -> np.ndarray:
        nonlocal state
        out, state = sosfilt(sos, np.asarray(block, dtype=np.float64), zi=state)
        return out.astype(np.float32)

    return run


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad the end of ``x`` to exactly ``n`` samples."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))


def crossfade(prev_tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Crossfade from ``prev_tail`` into ``head`` (same length) whose gains add up to one,
    which keeps the level of two signals that are alike (equal-gain)."""
    n = len(head)
    if n == 0:
        return head
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    return prev_tail * np.cos(t) ** 2 + head * np.sin(t) ** 2


CROSSFADE_BANDS_HZ = (1500.0, 4000.0)  # edges of the bands that band_crossfade treats apart


def band_crossfade(prev_tail: np.ndarray, head: np.ndarray, sr: int) -> np.ndarray:
    """Crossfade that keeps the level in every frequency band, however alike the two are.

    An equal-gain crossfade is right for two versions of a sound that line up, but where they
    are unrelated (the hiss of an "sz", the highest harmonics) it is 3 dB too quiet halfway
    through. So the two are split into bands; in each band the gains are ``cos``/``sin``
    divided by ``sqrt(1 + 2 rho cos sin)``, where ``rho`` is how alike the two are there:
    equal-gain for ``rho = 1``, equal-power for ``rho = 0``. At both ends every band has the
    same gain, so the split itself leaves no trace.
    """
    n = len(head)
    if n < 8:
        return crossfade(prev_tail, head)
    a = np.asarray(prev_tail, dtype=np.float64)
    b = np.asarray(head, dtype=np.float64)
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = [0.0, *CROSSFADE_BANDS_HZ, sr / 2 + 1]
    t = np.linspace(0.0, np.pi / 2, n)
    cos, sin = np.cos(t), np.sin(t)
    out = np.zeros(n)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        band_a = np.fft.irfft(np.where(mask, fa, 0), n)
        band_b = np.fft.irfft(np.where(mask, fb, 0), n)
        energy = np.sqrt(np.dot(band_a, band_a) * np.dot(band_b, band_b))
        rho = float(np.clip(np.dot(band_a, band_b) / energy, 0.0, 1.0)) if energy > 1e-12 else 1.0
        gain = 1.0 / np.sqrt(1.0 + 2.0 * rho * cos * sin)
        out += (band_a * cos + band_b * sin) * gain
    return out.astype(np.float32)


def f0_to_coarse(f0: np.ndarray) -> np.ndarray:
    """Quantise f0 (Hz, 0 = unvoiced) to RVC's 1..255 coarse pitch bins."""
    f0 = np.asarray(f0, dtype=np.float64)
    mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - _MEL_MIN) * 254.0 / (_MEL_MAX - _MEL_MIN) + 1.0
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64)


def estimate_f0(
    x: np.ndarray,
    sr: int = 16000,
    hop: int = 160,
    window: int = 1024,
    fmin: float = F0_MIN,
    fmax: float = F0_MAX,
    threshold: float = 0.15,
    silence_db: float = -60.0,
) -> np.ndarray:
    """YIN pitch estimate, one value per ``hop`` samples (0 where unvoiced).

    Returns ``len(x) // hop + 1`` frames, matching RVC's framing at 10 ms hops.
    Used when no RMVPE model is configured.
    """
    x = np.asarray(x, dtype=np.float32)
    n_frames = len(x) // hop + 1
    half = window // 2
    padded = np.pad(x, (half, half + window))
    starts = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)[starts]

    tau_min = max(2, int(sr / fmax))
    tau_max = min(half - 1, int(sr / fmin))

    # Difference function d(tau) = sum_{j<half} (x_j - x_{j+tau})^2 via FFT.
    nfft = 1 << int(np.ceil(np.log2(2 * window)))
    spec_full = np.fft.rfft(frames, nfft, axis=1)
    spec_head = np.fft.rfft(frames[:, :half], nfft, axis=1)
    cross = np.fft.irfft(np.conj(spec_head) * spec_full, nfft, axis=1)[:, : half + 1]
    sq = np.square(frames, dtype=np.float64)
    csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(sq, axis=1)], axis=1)
    taus = np.arange(half + 1)
    energy_head = csum[:, half][:, None]
    energy_shift = csum[:, taus + half] - csum[:, taus]
    diff = energy_head + energy_shift - 2.0 * cross
    diff[:, 0] = 0.0

    # Cumulative mean normalised difference.
    running = np.cumsum(diff[:, 1:], axis=1)
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] * taus[1:] / np.maximum(running, 1e-12)

    f0 = np.zeros(n_frames, dtype=np.float32)
    frame_db = 10.0 * np.log10(np.maximum(sq[:, :half].mean(axis=1), 1e-12))
    for i in range(n_frames):
        if frame_db[i] < silence_db:
            continue
        c = cmnd[i, tau_min : tau_max + 1]
        below = np.nonzero(c < threshold)[0]
        if below.size:
            k = below[0]
            while k + 1 < len(c) and c[k + 1] < c[k]:
                k += 1
        else:
            k = int(np.argmin(c))
            if c[k] > 2 * threshold:
                continue
        tau = k + tau_min
        if 1 <= tau < half:
            a, b, d = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
            denom = a - 2 * b + d
            shift = 0.5 * (a - d) / denom if abs(denom) > 1e-12 else 0.0
            tau = tau + float(np.clip(shift, -1, 1))
        f0[i] = sr / tau
    return f0


def split_at_quiet(x: np.ndarray, sr: int, target_s: float = 6.0, search_s: float = 1.5) -> list:
    """Cut ``x`` into pieces of about ``target_s`` seconds, each cut at the quietest moment
    within ``search_s`` seconds before the target, so no word is split."""
    x = np.asarray(x, dtype=np.float32)
    target, search, frame = int(target_s * sr), int(search_s * sr), max(1, int(0.02 * sr))
    pieces, start = [], 0
    while len(x) - start > target + search // 2:
        lo, hi = start + target - search, start + target + search // 2
        seg = x[lo:hi]
        n = len(seg) // frame
        level = np.sqrt((seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        quiet = level <= level.min() * 1.2 + 1e-5
        edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
        runs = list(zip(edges[::2], edges[1::2], strict=True))  # [first, end) of each quiet stretch
        first, end = max(runs, key=lambda r: r[1] - r[0])
        cut = lo + ((first + end) // 2) * frame  # the middle of the longest pause
        pieces.append(x[start:cut])
        start = cut
    pieces.append(x[start:])
    return pieces
