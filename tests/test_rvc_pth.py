import numpy as np
import onnxruntime as ort
import pytest
from fakes import fake_contentvec
from rvc_fixture import TINY_CONFIG, feed, for_session, random_voice
from torch_fixture import write_checkpoint

from ttslts.vc import rvc_pth
from ttslts.vc.rvc import ContentVec, RvcModel
from ttslts.vc.torch_zip import load_object


def _session(model):
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _save(tmp_path, voice, name="voice.pth"):
    path = tmp_path / name
    extra = {k: v for k, v in voice.items() if k != "weight"}
    write_checkpoint(path, voice["weight"], extra=extra)
    return str(path)


def test_graph_runs_and_has_expected_shape():
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg))
    out = session.run(None, feed())[0]
    assert out.shape == (1, 1, 9 * 8)  # frames * product of upsample rates
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0


def test_works_for_short_and_long_inputs():
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg))
    for frames in (1, 3, 25):
        out = session.run(None, feed(frames=frames))[0]
        assert out.shape[-1] == frames * 8


def test_noise_free_build_is_deterministic():
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg, noise=False))
    assert "nsf_noise" not in [i.name for i in session.get_inputs()]
    a = session.run(None, for_session(session, feed()))[0]
    b = session.run(None, for_session(session, feed()))[0]
    assert np.array_equal(a, b)


def test_noise_comes_from_the_input():
    """Same noise in, same audio out: overlapping windows of a stream can then agree."""
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg))
    a = session.run(None, feed())[0]
    assert np.array_equal(a, session.run(None, feed())[0])
    other = {**feed(), "nsf_noise": feed(seed=7)["nsf_noise"]}
    assert not np.allclose(a, session.run(None, other)[0])


def test_skip_uses_the_noise_of_the_decoded_frames():
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg))
    inputs = {**feed(), "skip": np.array([4], dtype=np.int64)}
    base = session.run(None, inputs)[0]

    def changed(frame):
        noise = inputs["nsf_noise"].copy()
        noise[..., frame] += 1.0
        return session.run(None, {**inputs, "nsf_noise": noise})[0]

    assert np.array_equal(changed(2), base)  # a frame that is not decoded
    assert not np.array_equal(changed(6), base)


def test_second_resblock_type_builds():
    config = list(TINY_CONFIG)
    config[9], config[11] = "2", [[1, 3], [1, 3]]
    state, cfg = rvc_pth.validate(random_voice(config))
    out = _session(rvc_pth.build_model(state, cfg)).run(None, feed(config))[0]
    assert out.shape == (1, 1, 72)


def test_checkpoint_roundtrip_and_sample_rate(tmp_path):
    path = _save(tmp_path, random_voice())
    loaded = load_object(path)
    assert loaded["version"] == "v2" and loaded["config"][-1] == 16000
    state, cfg = rvc_pth.validate(loaded)
    assert cfg.sample_rate == 16000 and cfg.upp == 8
    assert state["emb_g.weight"].shape == (3, 8)


def test_string_sample_rate_names():
    voice = random_voice(sr="40k")
    voice["config"][-1] = "40k"
    assert rvc_pth.validate(voice)[1].sample_rate == 40000


@pytest.mark.parametrize(
    "change, message",
    [
        ({"version": "v1"}, "v1"),
        ({"f0": 0}, "pitch"),
    ],
)
def test_unsupported_voices_are_explained(change, message):
    with pytest.raises(rvc_pth.UnsupportedVoice, match=message):
        rvc_pth.validate(random_voice(**change))


def test_non_voice_checkpoint_is_rejected():
    with pytest.raises(rvc_pth.UnsupportedVoice):
        rvc_pth.validate({"model": {}})


def test_ensure_onnx_converts_once_and_feeds_the_voice_model(tmp_path):
    path = _save(tmp_path, random_voice())
    cache = tmp_path / "cache"
    first = rvc_pth.ensure_onnx(path, str(cache))
    stamp = (cache / first.rsplit("/", 1)[-1]).stat().st_mtime_ns
    assert rvc_pth.ensure_onnx(path, str(cache)) == first
    assert (cache / first.rsplit("/", 1)[-1]).stat().st_mtime_ns == stamp  # reused

    session = ort.InferenceSession(first, providers=["CPUExecutionProvider"])
    model = RvcModel(session, ContentVec(fake_contentvec()))
    assert model.sample_rate == 16000 and model.info.uses_f0
    audio = model.convert(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)
    assert audio.size > 0 and np.isfinite(audio).all()


def test_weight_norm_forms_agree():
    g = np.array([[[2.0]], [[0.5]]], dtype=np.float32)
    v = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    old = rvc_pth._conv_weight({"c.weight_g": g, "c.weight_v": v}, "c")
    new = rvc_pth._conv_weight(
        {"c.parametrizations.weight.original0": g, "c.parametrizations.weight.original1": v}, "c"
    )
    assert np.allclose(old, new)
    assert np.allclose(np.sqrt((old**2).sum(axis=(1, 2))), [2.0, 0.5], atol=1e-5)


def test_skip_decodes_only_the_newest_frames():
    state, cfg = rvc_pth.validate(random_voice())
    session = _session(rvc_pth.build_model(state, cfg, noise=False))
    assert "skip" in [i.name for i in session.get_inputs()]
    inputs = for_session(session, feed())
    full = session.run(None, inputs)[0]  # skip = 0 decodes everything
    assert full.shape[-1] == 72
    part = session.run(None, {**inputs, "skip": np.array([4], dtype=np.int64)})[0]
    assert part.shape[-1] == (9 - 4) * 8 and np.isfinite(part).all()


def test_convert_returns_just_the_newest_audio_when_asked(tmp_path):
    path = _save(tmp_path, random_voice())
    session = ort.InferenceSession(
        rvc_pth.ensure_onnx(path, str(tmp_path / "cache")), providers=["CPUExecutionProvider"]
    )
    model = RvcModel(session, ContentVec(fake_contentvec()))
    assert model.can_skip
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    full = model.convert(audio)
    part = model.convert(audio, keep_16k=400)
    assert len(part) == 400 < len(full) and np.isfinite(part).all()
    assert model.build_feed(audio, keep_16k=400)["skip"][0] > 0  # older frames not decoded


def test_models_without_skip_are_trimmed_to_the_newest_audio():
    from fakes import rvc_session

    model = RvcModel(
        rvc_session(["feats", "p_len", "pitch", "pitchf", "sid"]), ContentVec(fake_contentvec())
    )
    assert not model.can_skip
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    assert len(model.convert(audio, keep_16k=3200)) == round(3200 * 40000 / 16000)
