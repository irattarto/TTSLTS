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
