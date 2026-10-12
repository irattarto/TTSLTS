import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from tests.torch_fixture import write_checkpoint
from ttslts.vc import hubert_onnx, rmvpe_mel, rvc_base
from ttslts.vc.torch_zip import load_state_dict

# --- torch-free checkpoint reader ------------------------------------------------------------


def test_reads_tensors_of_several_dtypes(tmp_path):
    tensors = {
        "a.weight": np.arange(24, dtype=np.float16).reshape(2, 3, 4),
        "a.bias": np.linspace(-1, 1, 5, dtype=np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
    }
    path = tmp_path / "m.bin"
    write_checkpoint(path, tensors)
    loaded = load_state_dict(str(path))
    assert list(loaded) == list(tensors)
    for name, array in tensors.items():
        assert loaded[name].dtype == array.dtype
        np.testing.assert_array_equal(loaded[name], array)


def test_unwraps_a_model_entry(tmp_path):
    path = tmp_path / "m.pt"
    write_checkpoint(path, {"w": np.ones((2, 2), dtype=np.float32)}, wrap="model")
    assert list(load_state_dict(str(path))) == ["w"]


def test_refuses_unexpected_globals(tmp_path):
    path = tmp_path / "evil.bin"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(Path))  # any global outside the allow-list
    with pytest.raises(pickle.UnpicklingError, match="Refusing"):
        load_state_dict(str(path))


def test_rejects_non_checkpoints(tmp_path):
    path = tmp_path / "plain.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(ValueError):
        load_state_dict(str(path))


# --- ContentVec graph against an independent NumPy implementation ------------------------------


def _conv1d(x, w, bias=None, stride=1, pad=0, groups=1):
    x = np.pad(x, ((0, 0), (pad, pad)))
    out_ch, in_per_group, k = w.shape
    frames = (x.shape[1] - k) // stride + 1
    out = np.zeros((out_ch, frames), dtype=np.float32)
    per_group = out_ch // groups
    for g in range(groups):
        xs = x[g * in_per_group : (g + 1) * in_per_group]
        cols = np.stack([xs[:, t * stride : t * stride + k].reshape(-1) for t in range(frames)], 1)
        out[g * per_group : (g + 1) * per_group] = (
            w[g * per_group : (g + 1) * per_group].reshape(per_group, -1) @ cols
        )
    if bias is not None:
        out += bias[:, None]
    return out


def _gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _reference(state, cfg, audio):
    x = audio[None]  # (1, samples)
    for i, s in enumerate(cfg.conv_stride):
        x = _conv1d(x, state[f"feature_extractor.conv_layers.{i}.conv.weight"], stride=s)
        if i == 0:
            p = "feature_extractor.conv_layers.0.layer_norm"
            x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + cfg.eps)
            x = x * state[f"{p}.weight"][:, None] + state[f"{p}.bias"][:, None]
        x = _gelu(x)
    x = x.T  # (frames, channels)
    x = _layer_norm(
        x,
        state["feature_projection.layer_norm.weight"],
        state["feature_projection.layer_norm.bias"],
    )
    x = (
        x @ state["feature_projection.projection.weight"].T
        + state["feature_projection.projection.bias"]
    )

    g = state["encoder.pos_conv_embed.conv.weight_g"]
    v = state["encoder.pos_conv_embed.conv.weight_v"]
    w = g * v / np.sqrt((v**2).sum(axis=(0, 1), keepdims=True))
    pos = _conv1d(
        x.T,
        w,
        state["encoder.pos_conv_embed.conv.bias"],
        pad=cfg.pos_kernel // 2,
        groups=cfg.pos_groups,
    )
    pos = _gelu(pos[:, :-1] if cfg.pos_kernel % 2 == 0 else pos)
    x = _layer_norm(x + pos.T, state["encoder.layer_norm.weight"], state["encoder.layer_norm.bias"])

    heads, dim = cfg.num_heads, cfg.hidden_size // cfg.num_heads
    for i in range(cfg.num_layers):
        p = f"encoder.layers.{i}"

        def proj(name, p=p, x=x):
            return x @ state[f"{p}.attention.{name}.weight"].T + state[f"{p}.attention.{name}.bias"]

        q, k, v_ = (
            proj(n).reshape(-1, heads, dim).transpose(1, 0, 2)
            for n in ("q_proj", "k_proj", "v_proj")
        )
        scores = q @ k.transpose(0, 2, 1) * dim**-0.5
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        ctx = (scores @ v_).transpose(1, 0, 2).reshape(-1, cfg.hidden_size)
        attn = (
            ctx @ state[f"{p}.attention.out_proj.weight"].T + state[f"{p}.attention.out_proj.bias"]
        )
        x = _layer_norm(x + attn, state[f"{p}.layer_norm.weight"], state[f"{p}.layer_norm.bias"])
        hidden = _gelu(
            x @ state[f"{p}.feed_forward.intermediate_dense.weight"].T
            + state[f"{p}.feed_forward.intermediate_dense.bias"]
        )
        ff = (
            hidden @ state[f"{p}.feed_forward.output_dense.weight"].T
            + state[f"{p}.feed_forward.output_dense.bias"]
        )
        x = _layer_norm(
            x + ff, state[f"{p}.final_layer_norm.weight"], state[f"{p}.final_layer_norm.bias"]
        )
    return x


def _session(model):
    import onnxruntime as ort

    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("samples", [3200, 4017])
def test_onnx_graph_matches_numpy_reference(samples):
    cfg = hubert_onnx.tiny_config()
    state = hubert_onnx.random_state(cfg, seed=3)
    audio = np.random.default_rng(5).standard_normal(samples).astype(np.float32)
    got = _session(hubert_onnx.build_model(state, cfg)).run(None, {"source": audio[None]})[0]
    expected = _reference(state, cfg, audio)
    assert got.shape == (1, *expected.shape)
    np.testing.assert_allclose(got[0], expected, atol=2e-4, rtol=2e-3)


def test_graph_accepts_parametrized_weight_norm_names():
    cfg = hubert_onnx.tiny_config()
    state = hubert_onnx.random_state(cfg)
    renamed = dict(state)
    renamed["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = renamed.pop(
        "encoder.pos_conv_embed.conv.weight_g"
    )
    renamed["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = renamed.pop(
        "encoder.pos_conv_embed.conv.weight_v"
    )
    audio = np.random.default_rng(1).standard_normal((1, 3000)).astype(np.float32)
    a = _session(hubert_onnx.build_model(state, cfg)).run(None, {"source": audio})[0]
    b = _session(hubert_onnx.build_model(renamed, cfg)).run(None, {"source": audio})[0]
    np.testing.assert_array_equal(a, b)


def test_selftest_runs():
    assert "ok" in hubert_onnx.selftest()


def test_rejects_other_hubert_variants(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"do_stable_layer_norm": True}))
    with pytest.raises(ValueError):
        hubert_onnx.HubertConfig.from_json(str(path))


# --- RMVPE front end and decoder ---------------------------------------------------------------


def test_mel_spectrogram_shape_and_pitch_band():
    sr = rmvpe_mel.SR
    t = np.arange(sr) / sr
    mel = rmvpe_mel.mel_spectrogram(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert mel.shape == (rmvpe_mel.N_MELS, 1 + sr // rmvpe_mel.HOP)
    assert np.isfinite(mel).all() and mel.min() >= np.log(rmvpe_mel.CLAMP) - 1e-6
    centers = rmvpe_mel._mel_to_hz(
        np.linspace(rmvpe_mel._hz_to_mel(rmvpe_mel.FMIN), rmvpe_mel._hz_to_mel(rmvpe_mel.FMAX), 130)
    )[1:-1]
    assert abs(centers[int(mel[:, 50].argmax())] - 1000) < 100


def test_decode_salience_to_hz():
    salience = np.zeros((4, rmvpe_mel.BINS), dtype=np.float32)
    salience[0, 100] = 0.9
    salience[1, 200], salience[1, 201] = 0.8, 0.4
    salience[2, 5] = 0.01  # below the threshold: unvoiced
    f0 = rmvpe_mel.decode(salience)
    assert f0[0] == pytest.approx(10 * 2 ** ((1997.3794084376191 + 20 * 100) / 1200), rel=1e-5)
    assert f0[1] > 10 * 2 ** ((1997.3794084376191 + 20 * 200) / 1200)  # pulled toward bin 201
    assert f0[2] == 0 and f0[3] == 0


def test_rmvpe_wrapper_pads_time_and_trims_output():
    seen = {}

    class Input:
        name = "input"
        shape = [1, 128, "time"]

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _names, feed):
            mel = feed["input"]
            seen["shape"] = mel.shape
            out = np.zeros((1, mel.shape[2], rmvpe_mel.BINS), dtype=np.float32)
            out[0, :, 120] = 0.9
            return [out]

    audio = np.zeros(16000 + 123, dtype=np.float32)
    f0 = rmvpe_mel.RmvpeMelPitch(Session())(audio)
    assert seen["shape"][0:2] == (1, 128) and seen["shape"][2] % 32 == 0
    assert f0.shape == (len(audio) // rmvpe_mel.HOP + 1,)
    assert np.all(f0 > 0)
    assert rmvpe_mel.is_mel_model(Session())


# --- download and conversion flow --------------------------------------------------------------


def _fake_fetch(cfg, state, calls):
    def fetch(name, target, progress):
        calls.append(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == rvc_base.LICENSE_FILE:
            target.write_text("MIT License\nCopyright (c) 2022 lj1995\n", encoding="utf-8")
        elif name == rvc_base.CONFIG_FILE:
            target.write_text(
                json.dumps(
                    {
                        "conv_dim": list(cfg.conv_dim),
                        "conv_kernel": list(cfg.conv_kernel),
                        "conv_stride": list(cfg.conv_stride),
                        "hidden_size": cfg.hidden_size,
                        "num_hidden_layers": cfg.num_layers,
                        "num_attention_heads": cfg.num_heads,
                        "intermediate_size": cfg.intermediate_size,
                        "num_conv_pos_embeddings": cfg.pos_kernel,
                        "num_conv_pos_embedding_groups": cfg.pos_groups,
                        "layer_norm_eps": cfg.eps,
                        "do_stable_layer_norm": False,
                        "feat_extract_norm": "group",
                        "conv_bias": False,
                    }
                )
            )
        elif name == rvc_base.WEIGHTS_FILE:
            half = {k: v.astype(np.float16) for k, v in state.items()}
            write_checkpoint(target, half)
        else:
            target.write_bytes(b"onnx")

    return fetch


def test_download_and_convert(tmp_path, monkeypatch):
    cfg = hubert_onnx.tiny_config()
    state = hubert_onnx.random_state(cfg)
    calls: list[str] = []
    monkeypatch.setattr(rvc_base, "_fetch", _fake_fetch(cfg, state, calls))

    contentvec, rmvpe = rvc_base.download_and_convert(dest=tmp_path)

    assert rmvpe.read_bytes() == b"onnx"
    out = _session_from_file(contentvec).run(None, {"source": np.zeros((1, 3200), np.float32)})[0]
    assert out.shape[0] == 1 and out.shape[2] == cfg.hidden_size
    assert (tmp_path / "LICENSE-RVC.txt").read_text(encoding="utf-8").startswith("MIT License")
    assert not (tmp_path / "pytorch_model.bin").exists()  # replaced by the ONNX file
    assert set(calls) == {
        rvc_base.LICENSE_FILE,
        rvc_base.RMVPE_FILE,
        rvc_base.WEIGHTS_FILE,
        rvc_base.CONFIG_FILE,
    }

    calls.clear()
    rvc_base.download_and_convert(dest=tmp_path)
    assert calls == []  # everything is reused


def _session_from_file(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_downloads_come_from_the_pinned_original_repo():
    url = rvc_base._url(rvc_base.WEIGHTS_FILE)
    assert url == (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
        f"{rvc_base.REVISION}/hubert_base/pytorch_model.bin"
    )
    assert "%E4%BD%BF" in rvc_base._url(rvc_base.LICENSE_FILE)  # non-ASCII names are quoted


def test_consent_text_states_the_research_use_clause():
    text = rvc_base.CONSENT_TEXT
    assert "仅供研究使用" in text and "research use only" in text
    assert "huggingface.co/lj1995/VoiceConversionWebUI" in text


def test_load_rvc_picks_the_matching_rmvpe_wrapper(monkeypatch):
    from ttslts.vc import rvc

    class Input:
        def __init__(self, name, shape):
            self.name, self.shape, self.type = name, shape, "tensor(float)"

    class Session:
        def __init__(self, inputs):
            self._inputs = inputs

        def get_inputs(self):
            return self._inputs

        def get_modelmeta(self):
            class Meta:
                custom_metadata_map = {}

            return Meta()

    sessions = {
        "voice.onnx": Session([Input("feats", [1, "t", 768]), Input("p_len", [1])]),
        "cv.onnx": Session([Input("source", [1, "n"])]),
        "mel.onnx": Session([Input("input", [1, 128, "time"])]),
        "wave.onnx": Session([Input("waveform", [1, "n"]), Input("threshold", [1])]),
    }
    monkeypatch.setattr("ttslts.onnx_util.make_session", lambda path, gpu: sessions[path])

    assert isinstance(
        rvc.load_rvc("voice.onnx", "cv.onnx", "mel.onnx").pitch_fn, rmvpe_mel.RmvpeMelPitch
    )
    assert isinstance(rvc.load_rvc("voice.onnx", "cv.onnx", "wave.onnx").pitch_fn, rvc.RmvpePitch)
    assert rvc.load_rvc("voice.onnx", "cv.onnx").pitch_fn is rvc.estimate_f0  # built-in YIN
