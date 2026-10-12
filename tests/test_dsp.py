import numpy as np
import pytest

from ttslts.audio.dsp import crossfade, estimate_f0, f0_to_coarse, fit_length, resample, rms_db


def sine(freq, sr=16000, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq", [90, 220, 440, 900])
def test_estimate_f0_tracks_sine(freq):
    f0 = estimate_f0(sine(freq))
    assert len(f0) == 16000 // 160 + 1
    assert np.median(f0[5:-5]) == pytest.approx(freq, rel=0.02)


def test_estimate_f0_silence_is_unvoiced():
    assert not estimate_f0(np.zeros(8000, dtype=np.float32)).any()


def test_f0_to_coarse_range():
    coarse = f0_to_coarse(np.array([0.0, 50.0, 100.0, 220.0, 1100.0, 5000.0]))
    assert coarse.dtype == np.int64
    assert coarse[0] == 1 and coarse[-1] == 255 and coarse[-2] == 255
    assert np.all(np.diff(coarse[1:]) >= 0)


def test_resample_length_and_pitch():
    y = resample(sine(440, sr=48000), 48000, 16000)
    assert len(y) == 16000
    assert np.median(estimate_f0(y)[5:-5]) == pytest.approx(440, rel=0.02)


def test_fit_length():
    x = np.arange(5, dtype=np.float32)
    assert len(fit_length(x, 3)) == 3
    assert fit_length(x, 7).tolist() == [0, 1, 2, 3, 4, 0, 0]


def test_crossfade_endpoints():
    a, b = np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32)
    y = crossfade(a, b)
    assert y[0] == pytest.approx(1) and y[-1] == pytest.approx(0, abs=1e-6)
    # Equal-power for identical signals keeps the level constant.
    assert np.allclose(crossfade(a, a), 1)


def test_rms_db():
    assert rms_db(np.zeros(10)) <= -100
    assert rms_db(np.ones(10)) == pytest.approx(0)
