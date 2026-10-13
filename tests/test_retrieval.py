import numpy as np
import pytest
from fakes import fake_contentvec, rvc_session

from ttslts.settings import Settings
from ttslts.vc import retrieval
from ttslts.vc.retrieval import RetrievalIndex


def test_index_path_sits_next_to_the_voice(tmp_path):
    assert (
        retrieval.index_path(tmp_path / "anna" / "anna.onnx")
        == tmp_path / "anna" / "anna.index.npy"
    )
    assert retrieval.index_path(tmp_path / "anna.pth").name == "anna.index.npy"


def test_search_returns_the_closest_real_frames():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((500, 768)).astype(np.float32)
    index = RetrievalIndex(vectors)
    exact = index.search(vectors[:5], k=4)
    assert np.allclose(exact, vectors[:5], atol=1e-3)  # a frame that is in the index maps to itself
    noisy = vectors[7] + 0.01 * rng.standard_normal(768).astype(np.float32)
    assert np.abs(index.search(noisy[None])[0] - vectors[7]).max() < 0.05
    assert index.search(vectors[:3]).shape == (3, 768)


def test_search_with_fewer_frames_than_neighbours():
    index = RetrievalIndex(np.eye(3, 768, dtype=np.float32))
    assert index.search(np.eye(2, 768, dtype=np.float32)).shape == (2, 768)


def test_build_caps_and_roundtrips(tmp_path):
    feats = tmp_path / "feats"
    feats.mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        np.save(feats / f"{i}.npy", rng.standard_normal((400, 768)).astype(np.float32))
    out = tmp_path / "v" / "v.index.npy"
    assert retrieval.build(feats, out, max_vectors=500) == 500
    assert len(RetrievalIndex.load(out)) == 500
    assert retrieval.build(feats, out) == 1200  # under the cap: everything is kept
    with pytest.raises(FileNotFoundError):
        retrieval.build(tmp_path / "nothing", out)


def make_model(index=None, rate=0.0, protect=0.33):
    from ttslts.vc.rvc import ContentVec, RvcModel

    session = rvc_session(["feats", "p_len", "pitch", "pitchf", "sid"])
    model = RvcModel(
        session,
        ContentVec(fake_contentvec()),
        lambda a: np.where(np.arange(len(a) // 160 + 1) < 10, 0.0, 200.0).astype(np.float32),
    )
    model.index, model.index_rate, model.protect = index, rate, protect
    return model


def feed_feats(model):
    audio = np.random.default_rng(2).standard_normal(16000).astype(np.float32) * 0.1
    return next(v for k, v in model.build_feed(audio).items() if k == "feats")


def test_index_rate_zero_changes_nothing():
    plain = feed_feats(make_model())
    index = RetrievalIndex(np.full((50, 768), 5.0, dtype=np.float32))
    assert np.allclose(feed_feats(make_model(index, 0.0)), plain)


def test_index_pulls_voiced_frames_to_the_voice_and_protect_spares_unvoiced_ones():
    index = RetrievalIndex(np.full((50, 768), 5.0, dtype=np.float32))
    plain = feed_feats(make_model())
    mixed = feed_feats(make_model(index, 1.0, protect=0.0))
    voiced = slice(15, None)  # the stub pitch tracker reports no pitch for the first 10 frames
    assert np.allclose(mixed[0, voiced], 5.0)  # fully the index
    assert np.allclose(mixed[0, :10], plain[0, :10])  # protect 0: unvoiced frames stay as spoken
    half = feed_feats(make_model(index, 1.0, protect=0.33))
    assert np.allclose(half[0, :10], 0.33 * 5.0 + 0.67 * plain[0, :10], atol=1e-4)


def test_settings_have_defaults():
    s = Settings()
    assert 0 < s.rvc_index_rate < 1 and 0 < s.rvc_protect < 0.5
