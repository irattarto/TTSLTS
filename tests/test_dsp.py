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


def test_split_at_quiet_cuts_in_a_pause_and_keeps_every_sample():
    from ttslts.audio.dsp import split_at_quiet

    sr = 16000
    rng = np.random.default_rng(0)
    speech = lambda s: (0.3 * rng.standard_normal(int(s * sr))).astype(np.float32)  # noqa: E731
    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
    x = np.concatenate([speech(5.5), pause, speech(5.2), pause, speech(5.2), pause, speech(3)])
    pieces = split_at_quiet(x, sr, target_s=6.0, search_s=1.5)
    assert sum(len(p) for p in pieces) == len(x) and np.array_equal(np.concatenate(pieces), x)
    assert all(len(p) <= 8 * sr for p in pieces[:-1]) and len(pieces) >= 3
    for p in pieces[:-1]:  # every cut lands inside a pause, so no word is split
        assert abs(p[-int(0.05 * sr) :]).max() == 0 or abs(p[-int(0.05 * sr) :]).mean() < 0.05


def test_split_at_quiet_leaves_short_audio_alone():
    from ttslts.audio.dsp import split_at_quiet

    x = np.ones(16000 * 4, dtype=np.float32)
    assert len(split_at_quiet(x, 16000)) == 1


def test_band_crossfade_keeps_alike_signals_unchanged():
    from ttslts.audio.dsp import band_crossfade

    a = sine(300, sr=48000, seconds=0.04) + sine(6000, sr=48000, seconds=0.04, amp=0.1)
    y = band_crossfade(a, a, 48000)
    assert np.abs(y - a).max() < 1e-4


def test_band_crossfade_keeps_the_level_of_unrelated_noise():
    """Hiss (an "sz") in two unrelated versions: equal-gain loses 3 dB halfway, this does not."""
    from ttslts.audio.dsp import band_crossfade

    rng = np.random.default_rng(0)
    n, trials = 1920, 200
    mid = slice(n // 2 - 200, n // 2 + 200)
    plain = fixed = 0.0
    for _ in range(trials):
        a, b = rng.standard_normal((2, n)).astype(np.float32)
        plain += float(np.mean(crossfade(a, b)[mid] ** 2))
        fixed += float(np.mean(band_crossfade(a, b, 48000)[mid] ** 2))
    assert 10 * np.log10(plain / trials) < -2.5
    assert abs(10 * np.log10(fixed / trials)) < 0.5


def test_band_crossfade_ends_on_the_two_signals():
    from ttslts.audio.dsp import band_crossfade

    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 960)).astype(np.float32)
    y = band_crossfade(a, b, 48000)
    assert y[0] == pytest.approx(a[0], abs=1e-4) and y[-1] == pytest.approx(b[-1], abs=1e-4)


def test_highpass_filter_streams_like_one_pass():
    from scipy.signal import butter, sosfilt

    from ttslts.audio.dsp import HIGHPASS_HZ, highpass_filter

    x = np.random.default_rng(2).standard_normal(48000).astype(np.float32)
    run = highpass_filter(48000)
    streamed = np.concatenate([run(x[i : i + 4800]) for i in range(0, len(x), 4800)])
    whole = sosfilt(butter(5, HIGHPASS_HZ, btype="high", fs=48000, output="sos"), x)
    assert np.abs(streamed - whole).max() < 1e-4
