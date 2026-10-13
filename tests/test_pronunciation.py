import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ttslts import paths
from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.tts import pronunciation
from ttslts.tts.pronunciation import BUILTIN_HU, Entry


def test_whole_words_are_respelled_case_insensitively():
    entries = [Entry("Windows", "vindóz"), Entry("Nvidia", "envídia")]
    assert pronunciation.apply("A Windows és az NVIDIA.", entries) == "A vindóz és az envídia."
    assert pronunciation.apply("Windowsra", entries) == "Windowsra"  # only whole words
    assert pronunciation.apply("Windows-ban fut", entries) == "vindóz-ban fut"  # Hungarian suffix


def test_match_case_and_longest_entry_first():
    entries = [Entry("IT", "áj tí", match_case=True), Entry("New York", "nyú jork")]
    assert pronunciation.apply("Az IT és it, New York", entries) == "Az áj tí és it, nyú jork"
    both = [Entry("York", "jork"), Entry("New York", "nyú jork")]
    assert pronunciation.apply("New York", both) == "nyú jork"


def test_a_replacement_is_never_replaced_again():
    entries = [Entry("a", "b"), Entry("b", "c")]
    assert pronunciation.apply("a b", entries) == "b c"


def test_your_entry_beats_the_built_in_list_and_only_capitals_match():
    assert pronunciation.apply("Az FBI jelentése", [], BUILTIN_HU) == "Az ef bí áj jelentése"
    assert pronunciation.apply("fbi it", [], BUILTIN_HU) == "fbi it"  # not an abbreviation here
    mine = [Entry("FBI", "föderál")]
    assert pronunciation.apply("FBI", mine, BUILTIN_HU) == "föderál"


def test_saved_entries_round_trip_and_bad_files_are_ignored(tmp_path):
    file = tmp_path / "p.json"
    entries = [Entry("Árvíz", "árvíz tűrő", True), Entry("", "x"), Entry("y", "")]
    pronunciation.save(entries, file)
    assert pronunciation.load(file) == [Entry("Árvíz", "árvíz tűrő", True)]
    file.write_text("{ not json", encoding="utf-8")
    assert pronunciation.load(file) == []
    assert pronunciation.load(tmp_path / "missing.json") == []


def test_built_in_list_is_only_for_hungarian():
    assert pronunciation.apply_for("FBI", "hu") == "ef bí áj"
    assert pronunciation.apply_for("FBI", "en-us") == "FBI"
    assert pronunciation.apply_for("FBI", "hu", use_builtin=False) == "FBI"


def test_the_speech_engines_get_the_respelled_text(monkeypatch):
    pronunciation.save([Entry("Windows", "vindóz")])
    seen = []

    class Fake:
        def synthesize(self, text, *args, **kw):
            seen.append(text)
            return np.ones(10, dtype=np.float32), 22050

    piper = Controller(Settings(tts_engine="piper"), save=lambda s: None)
    monkeypatch.setattr(piper, "piper", lambda: Fake())
    piper.render_tts("A Windows és az FBI")
    kokoro = Controller(Settings(tts_engine="kokoro", tts_lang="en-us"), save=lambda s: None)
    monkeypatch.setattr(kokoro, "kokoro", lambda: Fake())
    kokoro.render_tts("A Windows és az FBI")
    assert seen == [
        "A vindóz és az ef bí áj",
        "A vindóz és az FBI",
    ]  # built-in list: Hungarian only


def test_kokoro_hungarian_uses_the_hungarian_text_clean_up():
    from ttslts.tts.kokoro_engine import KokoroEngine

    seen = []

    class Raw:
        def create(self, text, voice, speed, lang):
            seen.append(text)
            return np.ones(5, dtype=np.float32), 24000

    engine = KokoroEngine(Raw())
    engine.synthesize("Ez kb. 3 kg.", "af_heart", 1.0, "hu")
    engine.synthesize("Ez kb. 3 kg.", "af_heart", 1.0, "en-us")
    assert "körülbelül" in seen[0] and "kilogramm" in seen[0]
    assert seen[1] == "Ez kb. 3 kg."


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_dialog_edits_entries_and_listens(app):
    from ttslts.ui.pronunciation_dialog import PronunciationDialog

    heard = []
    dialog = PronunciationDialog(
        None, [Entry("Windows", "vindóz", False)], True, heard.append, word="Nvidia"
    )
    assert dialog.table.rowCount() == 2 and dialog.table.item(1, 0).text() == "Nvidia"
    dialog.table.item(1, 1).setText("envídia")
    dialog.table.setCurrentCell(1, 1)
    dialog._on_listen()
    assert heard == ["envídia"]
    dialog._add_row("", "", False)  # an empty row is dropped when saving
    dialog.builtin_check.setChecked(False)
    assert [e.word for e in dialog.entries()] == ["Windows", "Nvidia"]
    assert not dialog.use_builtin()
    dialog.table.selectRow(0)
    dialog._remove_selected()
    assert [e.word for e in dialog.entries()] == ["Nvidia"]
    dialog.deleteLater()
    app.processEvents()


def test_window_saves_the_dictionary(monkeypatch, app):
    from ttslts.ui import pronunciation_dialog
    from ttslts.ui.main_window import MainWindow

    class Accepting(pronunciation_dialog.PronunciationDialog):
        def exec(self):
            self._add_row("Zoom", "zúm", False)
            self.builtin_check.setChecked(False)
            return 1

    monkeypatch.setattr(pronunciation_dialog, "PronunciationDialog", Accepting)
    window = MainWindow(Controller(Settings(), save=lambda s: None), tray_available=False)
    window._edit_pronunciations()
    assert pronunciation.load() == [Entry("Zoom", "zúm", False)]
    assert window.settings.pron_builtin is False
    assert paths.config_dir() in pronunciation.dictionary_file().parents
    window._quitting = True
    window.close()
    window.deleteLater()
    app.processEvents()
