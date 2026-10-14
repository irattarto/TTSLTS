import numpy as np
import pytest

from ttslts.audio.dsp import highpass_filter, resample
from ttslts.vc.realtime import OutputBuffer, StreamConverter


def passthrough_to(model_sr):
    from ttslts.audio.dsp import resample

    return lambda x16k, keep, pos: resample(x16k, 16000, model_sr)


def test_stream_reproduces_input_with_crossfade_delay():
    in_sr = out_sr = 48000
    conv = StreamConverter(
        passthrough_to(40000),
        40000,
        in_sr,
        out_sr,
        chunk_ms=100,
        context_ms=300,
        crossfade_ms=20,
        silence_db=-90,
    )
    t = np.arange(in_sr * 2) / in_sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = np.concatenate(
        [conv.process(x[i : i + conv.chunk_in]) for i in range(0, len(x), conv.chunk_in)]
    )
    assert len(out) == len(x)
    heard = highpass_filter(in_sr)(x)  # the microphone goes through the rumble filter
    a = out[in_sr // 2 :]
    best = max(
        np.corrcoef(a, heard[in_sr // 2 - d : len(x) - d])[0, 1]
        for d in range(conv.xfade_out, conv.xfade_out + conv.sola_out + 1)
    )
    assert best > 0.99


def test_window_has_no_glitches_at_block_edges():
    """The microphone audio reaches the model as one smooth signal, not block by block."""
    in_sr = 48000
    conv = StreamConverter(lambda x, keep, pos: x, 16000, in_sr, 16000, chunk_ms=100)
    t = np.arange(in_sr * 2) / in_sr
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    for i in range(0, len(x), conv.chunk_in):
        window = conv._window(x[i : i + conv.chunk_in])
    end = len(x) // 3
    expected = resample(highpass_filter(in_sr)(x), in_sr, 16000)[end - conv.window_16k : end]
    inner = slice(0, conv.window_16k - 40)  # the newest few samples have no future yet
    assert np.abs(window[inner] - expected[inner]).max() < 1e-3


def test_crossfade_cannot_be_longer_than_a_chunk():
    conv = StreamConverter(
        lambda x, keep, pos: np.zeros(keep, dtype=np.float32), 16000, 16000, 16000,
        chunk_ms=50, crossfade_ms=200, silence_db=-90,
    )  # fmt: skip
    assert len(conv.process(np.full(conv.chunk_in, 0.1, dtype=np.float32))) == conv.chunk_out


def test_stream_position_moves_one_chunk_per_block():
    positions = []

    def convert(x, keep, pos):
        positions.append(pos)
        return np.zeros(keep, dtype=np.float32)

    conv = StreamConverter(convert, 16000, 48000, 16000, chunk_ms=100, silence_db=-90)
    for _ in range(3):
        conv.process(np.full(conv.chunk_in, 0.1, dtype=np.float32))
    assert np.diff(positions).tolist() == [conv.chunk_16k] * 2
    assert positions[0] + conv.window_16k == conv.chunk_16k  # the window ends at "now"


def test_silence_gate_outputs_zeros():
    calls = []

    def convert(x, keep, pos):
        calls.append(x)
        return x

    conv = StreamConverter(convert, 16000, 16000, 16000, chunk_ms=100, silence_db=-40)
    out = conv.process(np.full(conv.chunk_in, 1e-4, dtype=np.float32))
    assert not calls and not out.any() and len(out) == conv.chunk_out


def test_stream_handles_short_model_output():
    conv = StreamConverter(
        lambda x, keep, pos: np.zeros(10, dtype=np.float32),
        40000,
        48000,
        44100,
        chunk_ms=200,
        silence_db=-90,
    )
    out = conv.process(np.random.default_rng(0).normal(0, 0.1, conv.chunk_in))
    assert len(out) == 44100 * 200 // 1000


def test_output_buffer_pads_and_bounds():
    buf = OutputBuffer(max_samples=10)
    buf.push(np.arange(15, dtype=np.float32))
    assert buf.pull(4).tolist() == [5, 6, 7, 8]
    assert buf.pull(10).tolist() == pytest.approx([9, 10, 11, 12, 13, 14, 0, 0, 0, 0])


def joined_amplitude(sola_ms):
    """Smallest short-term level of a steady tone whose phase restarts for every window."""
    rng = np.random.default_rng(3)
    sr = 16000

    def convert(x16k, keep, pos):  # like RVC: the same tone, but a new phase for every call
        t = np.arange(keep) / sr
        return (0.3 * np.sin(2 * np.pi * 150 * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)

    conv = StreamConverter(
        convert, sr, sr, sr, chunk_ms=100, context_ms=200, crossfade_ms=20, silence_db=-90,
        sola_ms=sola_ms,
    )  # fmt: skip
    block = np.full(conv.chunk_in, 0.1, dtype=np.float32)
    out = np.concatenate([conv.process(block) for _ in range(40)])[sr // 2 :]
    frames = out[: len(out) // 80 * 80].reshape(-1, 80)  # 5 ms
    level = np.sqrt((frames**2).mean(axis=1))
    return float(level.min() / level.max())


def test_sola_hides_the_joins_between_chunks():
    plain = joined_amplitude(0)
    aligned = joined_amplitude(12)
    assert plain < 0.6  # tones with unrelated phases cancel each other while crossfading
    assert aligned > 0.8  # SOLA lines the phase up first, so the level stays even


def test_sola_keeps_the_block_length():
    conv = StreamConverter(
        lambda x, keep, pos: np.zeros(keep, dtype=np.float32), 16000, 16000, 16000, chunk_ms=100,
        silence_db=-90, sola_ms=12,
    )  # fmt: skip
    out = conv.process(np.full(conv.chunk_in, 0.1, dtype=np.float32))
    assert len(out) == conv.chunk_out


def play(buf, pushes, pull_ms=10, sr=48000, seconds=8.0):
    """Simulated real time: ``pushes`` is a list of (time_s, samples); the speaker pulls
    ``pull_ms`` of sound at a steady rate. Returns what was played."""
    out, step, t, k = [], sr * pull_ms // 1000, 0.0, 0
    while t < seconds:
        while k < len(pushes) and pushes[k][0] <= t:
            buf.push(pushes[k][1])
            k += 1
        out.append(buf.pull(step))
        t += pull_ms / 1000
    return np.concatenate(out)


def test_output_margin_absorbs_slower_chunks():
    sr, chunk = 48000, 9600  # 200 ms chunks
    rng = np.random.default_rng(0)
    tone = (0.3 * np.sin(2 * np.pi * 227 * np.arange(sr * 9) / sr)).astype(np.float32)
    pushes = [  # every chunk takes 60-100 ms to convert after its audio is in
        (0.2 * (i + 1) + rng.uniform(0.06, 0.10), tone[i * chunk : (i + 1) * chunk])
        for i in range(40)
    ]
    plain = OutputBuffer(max_samples=chunk * 4)
    heard = play(plain, pushes)
    start = np.flatnonzero(heard)[0]
    assert plain.underruns >= 1  # played as soon as it arrives: the sound breaks off
    assert np.abs(np.diff(heard[start:])).max() > 0.1  # with a click

    buf = OutputBuffer(max_samples=chunk * 4, margin=sr * 40 // 1000, fade=240)
    played = play(buf, pushes)
    assert buf.underruns == 0
    start = np.flatnonzero(played)[0]
    assert np.abs(np.diff(played[start : start + 30 * chunk])).max() < 0.05  # no clicks


def test_output_margin_grows_after_running_dry_and_fades():
    sr, chunk = 48000, 9600
    tone = (0.3 * np.sin(2 * np.pi * 227 * np.arange(sr * 9) / sr)).astype(np.float32)
    late = {10: 0.12}  # one chunk that takes 120 ms longer than the others
    pushes = [(0.2 * (i + 1) + 0.05 + late.get(i, 0.0), tone[i * chunk : (i + 1) * chunk])
              for i in range(40)]  # fmt: skip
    buf = OutputBuffer(
        max_samples=chunk * 4, margin=960, margin_step=960, max_margin=chunk, fade=240
    )
    played = play(buf, pushes)
    assert buf.underruns >= 1 and buf.margin > 960
    assert np.abs(np.diff(played)).max() < 0.05  # it faded out and back in instead of a click


def test_output_buffer_trims_a_creeping_delay():
    chunk = 9600
    pushes = [(0.2 * (i + 1) * 0.99, np.full(chunk, 0.1, dtype=np.float32)) for i in range(40)]
    buf = OutputBuffer(max_samples=chunk * 4, margin=1920, fade=240, watch=10)
    play(buf, pushes)  # the microphone's clock runs 1 % fast: 2 ms more every chunk
    assert len(buf._data) < 1920 + chunk + 960


def test_gate_keeps_a_quiet_word_ending_and_a_short_pause():
    calls = []

    def convert(x, keep, pos):
        calls.append(pos)
        return np.full(keep, 0.1, dtype=np.float32)

    conv = StreamConverter(convert, 16000, 16000, 16000, chunk_ms=100, silence_db=-40)
    loud = np.full(conv.chunk_in, 0.1, dtype=np.float32)
    ending = np.zeros(conv.chunk_in, dtype=np.float32)
    ending[:300] = 0.05  # a soft consonant in an otherwise silent block: the block is quiet
    silent = np.zeros(conv.chunk_in, dtype=np.float32)
    for block in (loud, ending, silent, silent, silent, silent):
        conv.process(block)
    # the soft ending is converted, then the gate holds 200 ms (2 blocks) and closes
    assert len(calls) == 4


def test_pacer_waits_about_as_long_as_most_conversions_take():
    from ttslts.vc.realtime import Pacer

    pacer = Pacer(max_s=0.18)
    assert pacer.delay == 0.0
    for t in [0.10] * 30 + [0.12]:
        pacer.note(t)
    assert 0.10 < pacer.delay < 0.13
    for _ in range(50):
        pacer.note(0.5)  # far too slow: never more than the limit
    assert pacer.delay == 0.18


def test_model_keeps_running_now_and_then_during_silence():
    calls = []

    def convert(x, keep, pos):
        calls.append(pos)
        return np.zeros(keep, dtype=np.float32)

    conv = StreamConverter(convert, 16000, 16000, 16000, chunk_ms=200, silence_db=-40)
    silent = np.zeros(conv.chunk_in, dtype=np.float32)
    for _ in range(12):  # 2.4 s of silence
        assert not conv.process(silent).any()
    assert len(calls) == 3  # every 800 ms, so the graphics card does not slow down
    assert not conv.converted


def test_warm_up_runs_the_model_on_the_live_window_size():
    sizes = []

    def convert(x, keep, pos):
        sizes.append(len(x))
        return np.zeros(keep, dtype=np.float32)

    conv = StreamConverter(convert, 16000, 48000, 16000)
    assert conv.warm_up() >= 0.0
    assert sizes == [conv.window_16k] * 3
