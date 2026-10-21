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
    assert f5_text.prepare_text("Később  a „Duna” (partján)", VOCAB) == "később a duna, partján,"
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
    audio, sr = fake_voice.synthesize(
        "Az első mondat elég hosszú ahhoz, hogy maga menjen. "
        "A második mondat is elég hosszú, ez is külön megy!"
    )
    assert sr == 1000 and fake_voice.device == "cpu"
    pause = int(0.25 * sr)  # after a full stop, as with Piper
    assert len(audio) == 100 + pause + 100
    assert np.abs(audio).max() == pytest.approx(0.9)


def test_stop_ends_the_reading_between_sentences(fake_voice):
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 1

    long = "Ez a mondat elég hosszú ahhoz, hogy külön olvassa fel."
    audio, _ = fake_voice.synthesize(f"{long} {long} {long}", cancelled=cancelled)
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


def test_a_number_heard_as_digits_still_matches():
    words = [(" Nos,", 0.0, 0.4), (" 1848", 0.5, 1.6), (" március", 1.6, 2.0)]
    cut, ok = f5_text.filler_cut(words, "ezernyolcszáznegyvennyolc március")
    assert ok and cut == pytest.approx(0.53)


def test_a_short_first_word_needs_the_next_one_too():
    # "a huszonegyedik": the first "a" Whisper hears is not the text's, the later one is
    words = [(" Ánosz", 0.0, 0.5), (" a", 0.5, 0.6), (" kert", 0.6, 1.0), (" a", 1.0, 1.1)]
    cut, ok = f5_text.filler_cut(words, "a huszonegyedik században")
    assert not ok and cut < 0.6  # no false match far into the reading
    words = [(" Nos,", 0.0, 0.4), (" a", 0.5, 0.6), (" huszonegyedik", 0.6, 1.4)]
    cut, ok = f5_text.filler_cut(words, "a huszonegyedik században")
    assert ok and cut == pytest.approx(0.53)


class FakeSd:
    class CallbackStop(Exception):
        pass

    class OutputStream:
        def __init__(self, callback, finished_callback, **kw):
            self.callback, self.finished = callback, finished_callback

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    def stop(self):
        pass


def test_speech_stream_plays_pieces_without_gaps(monkeypatch):
    from ttslts.audio import devices

    monkeypatch.setattr(devices, "_sd", lambda: FakeSd())
    stream = devices.SpeechStream(1000, None)
    stream.add(np.ones(3, dtype=np.float32))
    stream.add(np.full(4, 2.0, dtype=np.float32))
    out = np.zeros((5, 1), dtype=np.float32)
    stream._fill(out, 5, None, None)
    assert out[:, 0].tolist() == [1, 1, 1, 2, 2]  # across the join
    stream._fill(out, 5, None, None)  # the next piece is not made yet: silence, no end
    assert out[:, 0].tolist() == [2, 2, 0, 0, 0]
    stream.finish()
    with pytest.raises(FakeSd.CallbackStop):
        stream._fill(out, 5, None, None)
    stream.stop()


def test_f5_speech_starts_with_the_first_sentence(monkeypatch):
    import threading

    from PySide6.QtCore import Qt

    from ttslts.audio import devices

    controller = Controller(Settings(piper_voice="f5:galambos"), save=lambda s: None)
    played, made = [], []

    class Voice:
        sample_rate = 24000

        def pieces(self, text, speed, cancelled):
            for n in range(3):
                made.append(n)
                yield np.full(7 * 24000, n, dtype=np.float32)  # 7 s: more than the start buffer

    class Stream:
        def __init__(self, sr, device):
            played.append(("open", sr, len(made)))  # opened after the first sentence only

        def add(self, audio):
            played.append(("add", int(audio[0])))

        def finish(self):
            played.append(("finish",))

    monkeypatch.setattr(controller, "f5", lambda: Voice())
    monkeypatch.setattr(devices, "SpeechStream", Stream)
    monkeypatch.setattr(devices, "resolve", lambda *a: None)
    done = threading.Event()
    controller.ttsBusy.connect(
        lambda busy: None if busy else done.set(), Qt.ConnectionType.DirectConnection
    )
    controller.speak("Egy. Kettő. Három.")
    assert done.wait(5)
    assert played == [("open", 24000, 1), ("add", 0), ("add", 1), ("add", 2), ("finish",)]


def test_melody_range_tells_flat_from_lively():
    sr = 16000
    t = np.arange(2 * sr) / sr
    flat = np.sin(2 * np.pi * 110 * t).astype(np.float32)
    glide = np.sin(2 * np.pi * np.cumsum(np.geomspace(80, 160, len(t))) / sr).astype(np.float32)
    assert f5_text.melody_range(flat, sr) < 1
    assert 8 < f5_text.melody_range(glide, sr) < 12  # an octave, 10th to 90th percentile
    assert not f5_text.is_lively(3.0, "egy hosszabb mondat, amely laposra sikerült")
    assert f5_text.is_lively(3.0, "Igen.")  # a few words are flat anyway


def test_words_whisper_joins_differently_still_match():
    words = [(" Nos,", 0.0, 0.4), (" Anap", 0.5, 0.9), (" lassan", 0.9, 1.3)]
    cut, ok = f5_text.filler_cut(words, "a nap lassan lebukott")
    assert ok and cut == pytest.approx(0.53)
    words = [(" Nos,", 0.0, 0.4), (" Oldsetter", 0.5, 1.0), (" hend", 1.0, 1.3)]
    cut, ok = f5_text.filler_cut(words, "old shatterhand a tűz mellett")
    assert ok and cut == pytest.approx(0.53)


def test_an_own_f5_voice_is_chosen_when_none_is(qt_application, isolated_home, tmp_path):
    from ttslts.tts import piper_download
    from ttslts.ui.main_window import MainWindow

    f5_voices.add_voice(make_voice_folder(tmp_path))  # into the test's app folder
    piper = piper_download.voice_paths("hu_HU-imre-medium")
    piper[0].parent.mkdir(parents=True, exist_ok=True)
    for p in piper:
        p.write_text("{}")
    for saved, chosen in (("", "f5:galambos"), ("hu_HU-imre-medium", "hu_HU-imre-medium")):
        w = MainWindow(Controller(Settings(piper_voice=saved), save=lambda s: None), False)
        try:
            assert w.settings.piper_voice == chosen  # a choice already made stays
        finally:
            w._quitting = True
            w.close()
            w.deleteLater()


def test_short_sentences_are_read_with_the_next_one():
    sentences = [
        ("„Milyen az ellenzéki lét?", 0.35),
        ("Szar!", 0.35),
        ("De nem azért, mert ellenzéki, hanem mert a szakadék felé robog.", 0.25),
    ]
    joined = f5_text.join_short(sentences, 0.4)
    # only the very short one goes with the next: the question keeps its own pause
    assert joined == [sentences[0], (f"{sentences[1][0]} {sentences[2][0]}", 0.25)]
    # never over a line break, and a short last one goes with the one before
    two = [("Ez egy elég hosszú mondat, amely maga megy.", 0.25), ("Igen.", 0.25)]
    assert f5_text.join_short(two, 0.4) == [(f"{two[0][0]} Igen.", 0.25)]
    assert f5_text.join_short([("Cím", 0.7), ("Szöveg.", 0.25)], 0.4) == [
        ("Cím", 0.7),
        ("Szöveg.", 0.25),
    ]


def test_the_cut_never_takes_the_text_away():
    # "Szar!": Whisper put the cut after the whole word; at least half of it must be left
    assert f5_text.keep_enough(0.62, 0.65, "szar!") == pytest.approx(0.65 - 0.5 * 5 * 0.065)
    assert f5_text.keep_enough(0.8, 6.0, "a nap lassan lebukott") == 0.8


def test_short_f5_speech_plays_when_all_is_made(monkeypatch):
    import threading

    from PySide6.QtCore import Qt

    from ttslts.audio import devices

    controller = Controller(Settings(piper_voice="f5:galambos"), save=lambda s: None)
    played = []

    class Voice:
        sample_rate = 24000

        def pieces(self, text, speed, cancelled):
            yield np.ones(24000, dtype=np.float32)  # 1 s: under the start buffer

    class Stream:
        def __init__(self, sr, device):
            played.append("open")

        def add(self, audio):
            played.append(len(audio))

        def finish(self):
            played.append("finish")

    monkeypatch.setattr(controller, "f5", lambda: Voice())
    monkeypatch.setattr(devices, "SpeechStream", Stream)
    monkeypatch.setattr(devices, "resolve", lambda *a: None)
    done = threading.Event()
    controller.ttsBusy.connect(
        lambda busy: None if busy else done.set(), Qt.ConnectionType.DirectConnection
    )
    controller.speak("Igen.")
    assert done.wait(5)
    assert played == ["open", 24000, "finish"]


def test_leading_silence_before_a_word():
    sr = 1000
    audio = np.concatenate([np.zeros(300), np.ones(200)]).astype(np.float32)
    assert f5_text.leading_silence(audio, sr) == pytest.approx(0.25)
    assert f5_text.leading_silence(np.ones(100, np.float32), sr) == 0.0


def test_f5_speed_is_capped_and_the_rest_shortens_pauses(fake_voice, monkeypatch):
    asked = []
    real = fake_voice.say
    monkeypatch.setattr(fake_voice, "say", lambda t, speed, seed: (asked.append(speed), real(t))[1])
    long = "Ez a mondat elég hosszú ahhoz, hogy külön olvassa fel."
    audio, sr = fake_voice.synthesize(f"{long} {long}", speed=1.25)
    assert asked == [1.1, 1.1]  # F5 itself never reads faster than this
    assert len(audio) == 100 + int(0.25 / 1.25 * sr) + 100


def test_brackets_and_hanging_hyphens_are_read_naturally():
    vocab = VOCAB
    assert f5_text.prepare_text("a Körök (DPK) első", vocab) == "a körök, dpk, első"
    assert f5_text.prepare_text("kép- és hangfelvétel", vocab) == "kép és hangfelvétel"
    assert f5_text.prepare_text("bel- vagy külföldi", vocab) == "bel vagy külföldi"
    assert (
        f5_text.prepare_text("a szöveg (zárójelben) vége.", vocab) == "a szöveg, zárójelben, vége."
    )


def test_the_start_after_the_cut_is_checked():
    text = "itt rendezték meg szombaton"
    assert f5_text.starts_with_text([(" Itt", 0.0, 0.2), (" rendezték", 0.2, 0.8)], text)
    assert f5_text.starts_with_text([(" Ittrendezték", 0.0, 0.8)], text)  # joined by Whisper
    assert not f5_text.starts_with_text([(" szombaton", 0.0, 0.6)], text)  # words were cut away
    assert not f5_text.starts_with_text([(" s", 0.0, 0.1), (" ott", 0.1, 0.3)], "a nap")


def test_a_start_missing_its_first_word_does_not_pass():
    text = "itt rendezték meg szombaton"
    assert not f5_text.starts_with_text([(" rendezték", 0.0, 0.4), (" meg", 0.4, 0.6)], text)


def test_what_is_left_of_the_filler_is_cut():
    sr = 1000
    word = np.full(500, 0.5, np.float32)
    faint = np.full(100, 0.002, np.float32)  # 48 dB under the word: a faint "s"
    assert f5_text.trim_leftover(np.concatenate([faint, word]), sr) == pytest.approx(0.05)
    scrap = np.full(40, 0.2, np.float32)  # a short louder scrap, a gap, then the word
    gap = np.zeros(60, np.float32)
    assert f5_text.trim_leftover(np.concatenate([scrap, gap, word]), sr) == pytest.approx(0.05)
    assert f5_text.trim_leftover(word, sr) == 0.0  # nothing in front: nothing cut


def test_quotes_and_word_hyphens_are_left_out():
    # quotation marks were read as sounds („milyen -> imilyen, ország" -> országkő)
    said = f5_text.prepare_text('„Milyen az ország" – válaszolta', VOCAB)
    assert said == "milyen az ország – válaszolta"
    assert (
        f5_text.prepare_text("kisebb-nagyobb csoportokban", VOCAB) == "kissebb nagyobb csoportokban"
    )


def test_long_sentences_are_cut_at_clause_boundaries():
    text = (
        "a lezsák sándor által alapított, harminc milliárd forintból felépített, "
        "grandiózus ligetben"
    )
    parts = f5_text.split_parts(text, 60)
    assert parts == ["a lezsák sándor által alapított,", "harminc milliárd forintból felépített,",
                     "grandiózus ligetben"]  # fmt: skip
    assert all(len(p.encode("utf-8")) <= 60 for p in parts)
    assert f5_text.split_parts("rövid mondat.", 60) == ["rövid mondat."]
    long_clause = " ".join(["szó"] * 40)
    assert all(len(p.encode("utf-8")) <= 60 for p in f5_text.split_parts(long_clause, 60))


def test_a_word_with_its_start_cut_away_is_not_the_word():
    assert not f5_text.same_word("oldalinak", "jobboldalinak")
    assert f5_text.same_word("Vinnetu", "winnetou")  # v and w are one
    assert not f5_text.starts_with_text([(" oldalinak", 0.0, 0.5)], "jobboldalinak vagy")


def test_kisebb_is_read_with_a_long_s():
    assert f5_text.prepare_text("A kisebb és a legkisebb kisebbség", VOCAB) == (
        "a kissebb és a legkissebb kissebbség"
    )


def test_a_vowel_start_is_cut_to_the_vowel():
    sr = 24000
    rng = np.random.default_rng(1)
    scrap = (np.diff(rng.standard_normal(int(0.04 * sr) + 1)) * 0.05).astype(np.float32)  # hiss
    pause = np.zeros(int(0.3 * sr), np.float32)
    t = np.arange(int(0.4 * sr)) / sr
    vowel = (0.5 * np.sin(2 * np.pi * 140 * t)).astype(np.float32)
    audio = np.concatenate([scrap, pause, vowel])
    assert f5_text.vowel_onset(audio, sr) == pytest.approx(0.32, abs=0.011)  # 20 ms before "a"
    assert f5_text.vowel_onset(vowel, sr) == 0.0
    assert f5_text.starts_with_vowel("A tűzoltók") and not f5_text.starts_with_vowel("Zsófi a")
    assert f5_text.trim_leftover(audio, sr) == pytest.approx(0.29, abs=0.011)  # scrap + pause
