import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep settings and models out of the real user profile."""
    monkeypatch.setenv("TTSLTS_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
