import numpy as np
import pytest

from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.tts import piper_download
from ttslts.tts.piper_engine import SENTENCE_PAUSE_S, PiperVoice, split_sentences

ID_MAP = {
    "_": [0],
    "^": [1],
    "$": [2],
    " ": [3],
    "a": [10],
    "b": [11],
    "ː": [12],
}


class Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, inputs=("input", "input_lengths", "scales"), samples=2000):
        self._inputs = [Input(n) for n in inputs]
        self.samples = samples
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def run(self, _names, feed):
        self.calls.append(feed)
        rng = np.random.default_rng(len(self.calls))
        return [rng.standard_normal((1, 1, 1, self.samples)).astype(np.float32) * 0.2]


def make_voice(session=None, **overrides):
    config = {
        "audio": {"sample_rate": 22050},
        "espeak": {"voice": "hu"},
        "inference": {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.7},
        "num_speakers": 1,
        "phoneme_type": "espeak",
        "phoneme_id_map": ID_MAP,
    }
    config.update(overrides)
    return PiperVoice(session or FakeSession(), config)


def test_phoneme_ids_follow_pipers_layout():
    voice = make_voice()
    # BOS, then every phoneme followed by PAD, then EOS; unknown phonemes are skipped
    assert voice.phoneme_ids(["a", "b", " ", "?"]) == [1, 10, 0, 11, 0, 3, 0, 2]
    assert voice.missing_phonemes(["a", "?", "b"]) == {"?"}


def test_scales_speed_and_lengths():
    session = FakeSession()
    voice = make_voice(session)
    audio, sr = voice.synthesize("ab", speed=2.0)
    feed = session.calls[0]
    assert sr == 22050 and audio.dtype == np.float32
    np.testing.assert_allclose(feed["scales"], [0.5, 0.5, 0.7])  # length_scale / speed
    assert feed["input"].dtype == np.int64 and feed["input_lengths"][0] == feed["input"].shape[1]
    assert "sid" not in feed


def test_multi_speaker_voices_get_a_speaker_id():
    session = FakeSession(inputs=("input", "input_lengths", "scales", "sid"))
    voice = make_voice(session, num_speakers=3)
    voice.synthesize("ab", speaker_id=2)
    assert session.calls[0]["sid"].tolist() == [2]


def test_sentences_are_joined_with_a_pause_and_normalised():
    session = FakeSession(samples=1000)
    voice = make_voice(session)
    audio, sr = voice.synthesize("Ab. Ba!")
    pause = int(SENTENCE_PAUSE_S * sr)
    assert len(session.calls) == 2
    assert len(audio) == 1000 + pause + 1000
    assert float(np.abs(audio).max()) == pytest.approx(0.9, rel=1e-4)


def test_empty_text_makes_no_sound():
    session = FakeSession()
    audio, sr = make_voice(session).synthesize("   ")
    assert audio.size == 0 and sr == 22050 and not session.calls


def test_split_sentences():
    assert split_sentences("Szia! Hogy vagy? Jól.\nÉs te?") == [
        "Szia!",
        "Hogy vagy?",
        "Jól.",
        "És te?",
    ]


def test_hungarian_is_phonemised_by_espeak():
    voice = make_voice()
    phonemes = voice.phonemize("Gyors tűz.")
    assert "ɟ" in phonemes and "y" in phonemes  # gy and ű are real Hungarian sounds
    assert phonemes[-1] == "."  # punctuation is kept


def test_only_espeak_voices_are_supported():
    with pytest.raises(ValueError):
        make_voice(phoneme_type="text")


# --- download and installation -----------------------------------------------------------------


def _fake_fetch(calls):
    def fetch(url, target, expected, progress):
        calls.append((url, target.name, expected))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    return fetch


def test_download_installs_all_voices_from_the_pinned_repo(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(piper_download, "_fetch", _fake_fetch(calls))

    installed = piper_download.download_voices(dest=tmp_path)

    assert installed == list(piper_download.VOICES)
    assert len(calls) == 9  # model, config and model card for three voices
    for url, _name, _size in calls:
        assert url.startswith(
            f"https://huggingface.co/rhasspy/piper-voices/resolve/{piper_download.REVISION}/hu/"
        )
    assert (tmp_path / "hu_HU-anna-medium.MODEL_CARD.txt").exists()
    assert {size for *_rest, size in calls} >= {63201294, 5018, 278}

    calls.clear()
    piper_download.download_voices(dest=tmp_path)
    assert calls == []  # nothing is fetched twice


def test_a_voice_needs_both_model_and_config(tmp_path):
    (tmp_path / "hu_HU-anna-medium.onnx").write_bytes(b"x")
    assert piper_download.installed_voices(tmp_path) == []
    (tmp_path / "hu_HU-anna-medium.onnx.json").write_bytes(b"{}")
    assert piper_download.installed_voices(tmp_path) == ["hu_HU-anna-medium"]


def test_consent_text_names_the_license_terms():
    text = piper_download.CONSENT_TEXT
    assert "CC0" in text and "lessac" in text and "research" in text
    assert "huggingface.co/rhasspy/piper-voices" in text


def test_urls_use_the_model_card_file_name():
    folder = "hu/hu_HU/anna/medium"
    url = piper_download._url(folder, "hu_HU-anna-medium", ".MODEL_CARD")
    assert url.endswith("/hu/hu_HU/anna/medium/MODEL_CARD")
    assert piper_download._url(folder, "hu_HU-anna-medium", ".onnx").endswith(
        "/hu_HU-anna-medium.onnx"
    )


# --- controller and window ---------------------------------------------------------------------


def test_controller_uses_piper_when_selected(monkeypatch):
    controller = Controller(Settings(tts_engine="piper", tts_speed=1.25), save=lambda s: None)
    seen = {}

    class Fake:
        def synthesize(self, text, speed):
            seen["args"] = (text, speed)
            return np.ones(10, dtype=np.float32), 22050

    monkeypatch.setattr(controller, "piper", lambda: Fake())
    audio, sr = controller.render_tts("Szia")
    assert seen["args"] == ("Szia", 1.25) and sr == 22050 and audio.size == 10


def test_controller_explains_a_missing_voice(isolated_home):
    controller = Controller(Settings(tts_engine="piper", piper_voice="hu_HU-anna-medium"))
    with pytest.raises(ValueError, match="Download Hungarian voices"):
        controller.piper()
