import numpy as np
import pytest

from ttslts.vc.realtime import OutputBuffer, StreamConverter


def passthrough_to(model_sr):
    from ttslts.audio.dsp import resample

    return lambda x16k: resample(x16k, 16000, model_sr)


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
    delay = conv.xfade_out
    a, b = out[in_sr // 2 :], x[in_sr // 2 - delay : len(x) - delay]
    assert np.corrcoef(a, b)[0, 1] > 0.99


def test_silence_gate_outputs_zeros():
    calls = []

    def convert(x):
        calls.append(x)
        return x

    conv = StreamConverter(convert, 16000, 16000, 16000, chunk_ms=100, silence_db=-40)
    out = conv.process(np.full(conv.chunk_in, 1e-4, dtype=np.float32))
    assert not calls and not out.any() and len(out) == conv.chunk_out


def test_stream_handles_short_model_output():
    conv = StreamConverter(
        lambda x: np.zeros(10, dtype=np.float32), 40000, 48000, 44100, chunk_ms=200, silence_db=-90
    )
    out = conv.process(np.random.default_rng(0).normal(0, 0.1, conv.chunk_in))
    assert len(out) == 44100 * 200 // 1000


def test_output_buffer_pads_and_bounds():
    buf = OutputBuffer(max_samples=10)
    buf.push(np.arange(15, dtype=np.float32))
    assert buf.pull(4).tolist() == [5, 6, 7, 8]
    assert buf.pull(10).tolist() == pytest.approx([9, 10, 11, 12, 13, 14, 0, 0, 0, 0])
