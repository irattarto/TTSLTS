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
