import sys

import numpy as np
import pytest

from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.tts import f5_runtime, f5_text, f5_voices
from ttslts.tts.f5_engine import F5Error, F5Voice

VOCAB = set(" !',-.:;?\"abcdefghijklmnopqrstuvwxyzáéíóöőúüű0123456789") | {"–", "„", "…"}


def test_text_is_lowercased_and_kept_to_the_vocabulary():
    # capitals are not in the Hungarian vocabulary: they would be read as spaces ("tésőbb")
    assert f5_text.prepare_text("Később  a „Duna” (partján)", VOCAB) == 'később a „duna" ,partján,'
    assert f5_text.prepare_text("Ő ÜGETETT ★", VOCAB) == "ő ügetett"


def test_reference_text_ends_in_a_full_stop_and_a_space():
    assert f5_text.reference_text("Jól látták", VOCAB) == "jól látták. "
    assert f5_text.reference_text("Jól látták?", VOCAB) == "jól látták? "


def test_filler_is_cut_just_before_the_first_word():
    words = [(" Nos,", 0.0, 0.42), (" Reggelente", 0.55, 1.1), (" a", 1.1, 1.2)]
    cut, ok = f5_text.filler_cut(words, "reggelente")
    assert ok and cut == pytest.approx(0.58)


def test_a_stray_sound_heard_as_a_word_is_cut_too():
    words = [(" U", 0.0, 0.2), (" nos,", 0.2, 0.6), (" galambos", 0.7, 1.2)]
    cut, ok = f5_text.filler_cut(words, "galambos")
    assert ok and cut == pytest.approx(0.73)


def test_filler_and_first_word_heard_as_one():
    cut, ok = f5_text.filler_cut([(" Nosvinetu", 0.2, 1.2), (" és", 1.3, 1.4)], "winnetou")
    assert ok and 0.2 < cut < 0.7  # inside the merged word, where "vinetu" starts


def test_a_start_that_matches_nothing_is_reported():
    cut, ok = f5_text.filler_cut([(" Nos,", 0.0, 0.4), (" valami", 0.6, 1.0)], "reggelente")
    assert not ok and cut == pytest.approx(0.5)
    assert f5_text.filler_cut([], "x") == (0.0, False)


def make_voice_folder(tmp_path, name="galambos-f5", text="Jól látták."):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "galambos-f5-update12300.safetensors").write_bytes(b"weights")
    (folder / "vocab.txt").write_text("\n".join(sorted(VOCAB)), encoding="utf-8")
    (folder / "reference.wav").write_bytes(b"RIFF")
    (folder / "reference.txt").write_text(text, encoding="utf-8")
    return folder / "galambos-f5-update12300.safetensors"


def test_adding_an_f5_voice(tmp_path):
    base = tmp_path / "models"
    voice_id = f5_voices.add_voice(make_voice_folder(tmp_path), base)
    assert voice_id == "f5:galambos"  # named after its folder
    assert f5_voices.installed_voices(base) == ["f5:galambos"]
    files = f5_voices.voice_files(voice_id, base)
    assert files["model"].read_bytes() == b"weights"
    assert files["ref_text"].read_text(encoding="utf-8") == "Jól látták."
    assert f5_voices.display_name(voice_id) == "galambos (own F5 voice)"
    assert f5_voices.is_f5(voice_id) and not f5_voices.is_f5("hu_HU-imre-medium")


def test_an_f5_voice_needs_its_sample_and_text(tmp_path):
    model = make_voice_folder(tmp_path)
    (model.parent / "reference.wav").unlink()
    with pytest.raises(f5_voices.NotAnF5Voice, match="reference.wav"):
        f5_voices.add_voice(model, tmp_path / "models")
    with pytest.raises(f5_voices.NotAnF5Voice):
        f5_voices.add_voice(tmp_path / "x.onnx", tmp_path / "models")
    empty = make_voice_folder(tmp_path, "other", text="  ")
    with pytest.raises(f5_voices.NotAnF5Voice, match="empty"):
        f5_voices.add_voice(empty, tmp_path / "models")


FAKE_WORKER = """
import json, sys
import numpy as np
for line in sys.stdin:
    msg = json.loads(line)
    if msg["op"] == "load":
        print(json.dumps({"type": "loaded", "device": msg["device"]}), flush=True)
    elif msg["op"] == "say":
        if "rossz" in msg["text"]:
            print(json.dumps({"type": "error", "message": "broken"}), flush=True)
            continue
        np.save(msg["out"], np.full(100, 0.5, dtype=np.float32))
        print(json.dumps({"type": "said", "out": msg["out"], "sr": 1000, "ok": True}), flush=True)
    elif msg["op"] == "quit":
        break
"""


@pytest.fixture
def fake_voice(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    files = {k: tmp_path / k for k in ("model", "vocab", "ref_audio", "ref_text")}
    voice = F5Voice([sys.executable, str(script)], {}, files, gpu=False)
    yield voice
    voice.close()


def test_sentences_are_read_one_by_one_with_pauses(fake_voice):
    audio, sr = fake_voice.synthesize("Első mondat. Második mondat!")
    assert sr == 1000 and fake_voice.device == "cpu"
    pause = int(0.25 * sr)  # after a full stop, as with Piper
    assert len(audio) == 100 + pause + 100
    assert np.abs(audio).max() == pytest.approx(0.9)


def test_stop_ends_the_reading_between_sentences(fake_voice):
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 1

    audio, _ = fake_voice.synthesize("Első. Második. Harmadik.", cancelled=cancelled)
    assert len(audio) == 100


def test_a_failed_sentence_is_reported(fake_voice):
    with pytest.raises(F5Error, match="broken"):
        fake_voice.synthesize("Ez rossz mondat.")
    audio, _ = fake_voice.synthesize("De ez jó.")  # the process is still usable
    assert len(audio) == 100


def test_controller_reads_with_the_f5_voice(monkeypatch):
    controller = Controller(Settings(piper_voice="f5:galambos"), save=lambda s: None)
    got = []

    class Voice:
        def synthesize(self, text, speed, cancelled):
            got.append((text, speed))
            return np.ones(10, dtype=np.float32), 24000

    monkeypatch.setattr(controller, "f5", lambda: Voice())
    audio, sr = controller.render_tts("Szia")
    assert sr == 24000 and got == [("Szia", 1.0)]


def test_f5_voice_needs_its_environment(monkeypatch):
    monkeypatch.setattr(f5_runtime, "is_installed", lambda base=None: False)
    controller = Controller(Settings(piper_voice="f5:galambos"), save=lambda s: None)
    with pytest.raises(ValueError, match="F5 environment"):
        controller.f5()


def test_starting_the_voice_changer_ends_the_f5_process():
    controller = Controller(Settings(), save=lambda s: None)
    closed = []
    controller._f5 = type("V", (), {"close": lambda self: closed.append(True)})()
    controller._f5_key = ("f5:x", True)
    controller._release_tts_models()
    assert closed and controller._f5 is None


def test_f5_environment_is_its_own(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not f5_runtime.is_installed(tmp_path / "none")
    assert f5_runtime.root().name == "runtime-f5"
    assert "GB" in f5_runtime.consent_text()


def test_piper_settings_are_off_for_an_f5_voice(qt_application, tmp_path, monkeypatch):
    from ttslts.ui.main_window import MainWindow

    controller = Controller(Settings(), save=lambda s: None)
    w = MainWindow(controller, tray_available=False)
    try:
        assert w.variation_spin.isEnabled()
        w.settings.piper_voice = "f5:galambos"
        w._apply_voice_kind()
        assert not w.variation_spin.isEnabled() and not w.rhythm_spin.isEnabled()
    finally:
        w._quitting = True
        w.close()
        w.deleteLater()
