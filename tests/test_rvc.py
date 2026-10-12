import json

import numpy as np
import pytest
from fakes import fake_contentvec, rvc_session

from ttslts.vc.rvc import ContentVec, RvcModel


def model(names, **kw):
    return RvcModel(
        rvc_session(names, **kw),
        ContentVec(fake_contentvec()),
        pitch_fn=lambda a: np.full(len(a) // 160 + 1, 200.0, dtype=np.float32),
    )


WOKADA = ["feats", "p_len", "pitch", "pitchf", "sid"]
WEBUI = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]


@pytest.mark.parametrize("names", [WOKADA, WEBUI])
def test_feed_matches_exporter_inputs(names):
    m = model(names)
    feed = m.build_feed(np.zeros(16000, dtype=np.float32), pitch_shift=12, speaker_id=3)
    assert set(feed) == set(names)
    feats = feed[names[0]]
    n = feats.shape[1]
    assert feats.shape == (1, n, 768) and n == 100  # 50 ContentVec frames doubled
    assert feed[names[1]].tolist() == [n]
    assert feed["pitch"].shape == (1, n) and feed["pitch"].dtype == np.int64
    assert np.allclose(feed["pitchf"], 400.0)  # +12 semitones doubles f0
    assert feed[names[4]].tolist() == [3]
    if "rnd" in feed:
        assert feed["rnd"].shape == (1, 192, n)


def test_half_precision_models_get_float16():
    m = model(WOKADA, dtype="tensor(float16)")
    feed = m.build_feed(np.zeros(16000, dtype=np.float32))
    assert feed["feats"].dtype == np.float16 and feed["pitchf"].dtype == np.float16


def test_model_without_f0():
    m = model(["feats", "p_len", "sid"])
    assert not m.info.uses_f0
    assert set(m.build_feed(np.zeros(16000, dtype=np.float32))) == {"feats", "p_len", "sid"}


def test_sample_rate_from_metadata_or_fallback():
    meta = {"metadata": json.dumps({"samplingRate": 48000})}
    assert model(WOKADA, meta=meta).sample_rate == 48000
    assert model(WOKADA).sample_rate == 40000


def test_convert_output_length():
    out = model(WOKADA, sr_hop=400).convert(np.zeros(16000, dtype=np.float32))
    assert out.ndim == 1 and len(out) == 40000


def test_unknown_inputs_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        model(["feats", "mystery"])
