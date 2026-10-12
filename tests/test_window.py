import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.ui.main_window import MainWindow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(app, monkeypatch):
    quits = []
    monkeypatch.setattr(QApplication, "quit", lambda *a: quits.append(True))
    windows = []

    def make(tray_available=True, **settings):
        saved = []
        controller = Controller(Settings(**settings), save=saved.append)
        w = MainWindow(controller, tray_available=tray_available)
        w.saved, w.quits = saved, quits
        windows.append(w)
        w.show()
        app.processEvents()
        return w

    yield make
    for w in windows:
        w._quitting = True
        w.close()
        w.deleteLater()


def test_settings_checkboxes_default_checked(make_window):
    w = make_window()
    assert w.minimize_check.text() == "Minimize to tray" and w.minimize_check.isChecked()
    assert w.close_check.text() == "Close to tray" and w.close_check.isChecked()


def test_unchecking_saves_setting(make_window):
    w = make_window()
    w.close_check.setChecked(False)
    assert w.settings.close_to_tray is False and w.saved


def test_close_hides_to_tray(make_window):
    w = make_window()
    w.close()
    assert not w.isVisible()
    assert not w.quits


def test_close_quits_when_disabled(make_window):
    w = make_window(close_to_tray=False)
    w.close()
    assert w.quits


def test_close_quits_without_system_tray(make_window):
    w = make_window(tray_available=False)
    assert not w.close_check.isEnabled()
    w.close()
    assert w.quits


@pytest.mark.parametrize("enabled", [True, False])
def test_minimize_to_tray(make_window, app, enabled):
    w = make_window(minimize_to_tray=enabled)
    w.setWindowState(Qt.WindowState.WindowMinimized)
    app.processEvents()
    app.processEvents()
    assert w.isVisible() is (not enabled)


def test_tray_menu_builds(make_window):
    from ttslts.ui.tray import Tray

    w = make_window()
    tray = Tray(w, w.controller)
    labels = [a.text() for a in tray.contextMenu().actions() if a.text()]
    assert labels == ["Show window", "Voice changer", "Speak clipboard", "Stop speaking", "Quit"]
    w.controller.vcRunning.emit(True)
    assert tray.vc_action.isChecked()


def _answer_consent(monkeypatch, accept: bool):
    """Replace the modal consent box: record its text and click Accept or Cancel."""
    from PySide6.QtWidgets import QMessageBox

    seen = {}

    def fake_exec(self):
        seen["text"] = self.informativeText()
        seen["title"] = self.windowTitle()

    def fake_clicked(self):
        buttons = self.buttons()
        return buttons[0] if accept else buttons[-1]

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", fake_clicked)
    return seen


@pytest.mark.parametrize("accept", [True, False])
def test_rvc_base_download_needs_consent(make_window, monkeypatch, accept):
    w = make_window()
    started = []
    monkeypatch.setattr(w.controller, "download_rvc_base", lambda: started.append(True))
    seen = _answer_consent(monkeypatch, accept)

    w._download_rvc_base()

    assert "research use only" in seen["text"] and "仅供研究使用" in seen["text"]
    assert started == ([True] if accept else [])


def test_downloaded_rvc_base_paths_are_saved(make_window):
    w = make_window()
    w._on_rvc_base_downloaded("/m/contentvec.onnx", "/m/rmvpe.onnx")
    assert w.settings.contentvec_path == "/m/contentvec.onnx"
    assert w.settings.rmvpe_path == "/m/rmvpe.onnx"
    assert w.contentvec_edit.text() == "/m/contentvec.onnx" and w.saved


def test_engine_choice_switches_the_visible_rows(make_window):
    w = make_window()
    form = w._tts_form
    assert form.isRowVisible(w.voice_combo) and not form.isRowVisible(w.piper_combo)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("piper"))
    assert w.settings.tts_engine == "piper"
    assert form.isRowVisible(w.piper_combo)
    assert not form.isRowVisible(w.voice_combo) and not form.isRowVisible(w.lang_combo)


def test_installed_piper_voices_are_listed(make_window, monkeypatch):
    from ttslts.tts import piper_download

    monkeypatch.setattr(piper_download, "installed_voices", lambda base=None: ["hu_HU-imre-medium"])
    w = make_window(tts_engine="piper")
    assert [w.piper_combo.itemData(i) for i in range(w.piper_combo.count())] == [
        "hu_HU-imre-medium"
    ]
    assert w.settings.piper_voice == "hu_HU-imre-medium"


@pytest.mark.parametrize("accept", [True, False])
def test_piper_download_needs_consent(make_window, monkeypatch, accept):
    w = make_window()
    started = []
    monkeypatch.setattr(w.controller, "download_piper_voices", lambda: started.append(True))
    seen = _answer_consent(monkeypatch, accept)

    w._download_piper()

    assert "CC0" in seen["text"] and "research" in seen["text"]
    assert started == ([True] if accept else [])


def _pick_language(w, code):
    w.lang_combo.setCurrentIndex(w.lang_combo.findData(code))


def test_choosing_hungarian_switches_to_piper_and_offers_the_download(make_window, monkeypatch):
    from PySide6.QtCore import QTimer

    from ttslts.tts import piper_download

    monkeypatch.setattr(piper_download, "installed_voices", lambda base=None: [])
    w = make_window()
    offered = []
    monkeypatch.setattr(w, "_download_piper", lambda: offered.append(True))
    # run the deferred offer immediately instead of waiting for the event loop
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda _ms, fn: fn()))

    _pick_language(w, "hu")

    assert w.settings.tts_engine == "piper" and w.settings.tts_lang == "hu"
    assert offered == [True]


def test_no_download_offer_when_voices_are_installed(make_window, monkeypatch):
    from PySide6.QtCore import QTimer

    from ttslts.tts import piper_download

    monkeypatch.setattr(piper_download, "installed_voices", lambda base=None: ["hu_HU-anna-medium"])
    w = make_window()
    offered = []
    monkeypatch.setattr(w, "_download_piper", lambda: offered.append(True))
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda _ms, fn: fn()))

    _pick_language(w, "hu")

    assert w.settings.tts_engine == "piper" and offered == []


def test_other_languages_stay_on_kokoro(make_window):
    w = make_window()
    _pick_language(w, "it")
    assert w.settings.tts_engine == "kokoro" and w.settings.tts_lang == "it"


def test_engine_can_be_set_back_to_kokoro_for_experimental_hungarian(make_window, monkeypatch):
    from ttslts.tts import piper_download

    monkeypatch.setattr(piper_download, "installed_voices", lambda base=None: ["hu_HU-anna-medium"])
    w = make_window()
    _pick_language(w, "hu")
    assert w.settings.tts_engine == "piper"

    w.engine_combo.setCurrentIndex(w.engine_combo.findData("kokoro"))

    # the user's choice sticks: Kokoro keeps speaking Hungarian (experimental)
    assert w.settings.tts_engine == "kokoro" and w.settings.tts_lang == "hu"
    assert w._tts_form.isRowVisible(w.lang_combo)


def test_piper_settings_have_help_text(make_window, monkeypatch):
    from ttslts.tts import piper_download

    monkeypatch.setattr(
        piper_download,
        "installed_voices",
        lambda base=None: ["hu_HU-anna-medium", "hu_HU-imre-medium"],
    )
    w = make_window(tts_engine="piper")
    for widget in (w.engine_combo, w.piper_combo, w.variation_spin, w.rhythm_spin, w.speed_spin):
        assert widget.toolTip().strip(), widget
    assert "Default" in w.variation_spin.toolTip() and "Default" in w.rhythm_spin.toolTip()
    # every listed voice explains itself on hover
    tips = [w.piper_combo.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(2)]
    assert all(tips) and tips[0].startswith("Anna") and tips[1].startswith("Imre")
    # a visible hint is shown with the Piper settings and hidden for Kokoro
    assert w._tts_form.isRowVisible(w.piper_hint)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("kokoro"))
    assert not w._tts_form.isRowVisible(w.piper_hint)


def test_train_tab_lists_files_and_saves_options(make_window, monkeypatch):
    w = make_window()
    assert w.tabs.tabText(2) == "Train"
    assert w.train_name_edit.text() == "my-voice" and w.train_epochs_spin.value() == 200
    w.train_epochs_spin.setValue(50)
    assert w.settings.train_epochs == 50
    w.train_name_edit.setText("My Voice: 1/2")
    assert w._train_name() == "My Voice 12"

    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *a, **k: (["a.mp3", "b.wav", "a.mp3"], "")
    )
    w._train_add_files()
    w._train_add_files()  # adding the same files again does not duplicate them
    assert w.train_files.count() == 3
    w.train_files.setCurrentRow(0)
    w.train_files.item(0).setSelected(True)
    w._train_remove_files()
    assert w.train_files.count() == 2


def test_train_tab_follows_training_state(make_window):
    w = make_window()
    w.controller.trainRunning.emit(True)
    assert not w.train_start_btn.isEnabled() and w.train_stop_btn.isEnabled()
    assert not w.train_name_edit.isEnabled()
    w.controller.trainProgress.emit(
        {"stage": "train", "fraction": 0.5, "epoch": 2, "epochs": 4, "step": 1, "steps": 2,
         "elapsed": 30.0}
    )  # fmt: skip
    assert w.train_bar.value() == 500 and "Epoch 2/4" in w.train_status.text()
    w.controller.trainFinished.emit("v.pth", "v.onnx", True)
    w.controller.trainRunning.emit(False)
    assert w.train_use_btn.isEnabled() and w.train_name_edit.isEnabled()
    w.train_use_btn.click()
    assert w.settings.rvc_model_path == "v.onnx" and w.rvc_model_edit.text() == "v.onnx"
