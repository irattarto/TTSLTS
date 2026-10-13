import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TTSLTS_NO_GPU_PROBE", "1")  # no extra processes in tests


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep settings and models out of the real user profile."""
    monkeypatch.setenv("TTSLTS_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """One Qt application for the whole run, created first and kept alive until the end.

    Tests that each created (and dropped) their own made the process crash when it exited.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()
