import json
import sys
import time

import pytest
from PySide6.QtWidgets import QApplication

from ttslts.controller import Controller
from ttslts.settings import Settings


def test_unavailable_reason_when_packaged(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert "not part of the packaged app" in Controller.training_unavailable_reason()


def test_start_requires_recordings_and_contentvec(tmp_path):
    controller = Controller(Settings(), save=lambda s: None)
    errors = []
    controller.error.connect(errors.append)
    if Controller.training_unavailable_reason():
        pytest.skip("training dependencies are not installed")
    controller.start_training([], "voice", resume=False)
    assert "at least one recording" in errors[-1]
    controller.start_training(["a.wav"], "voice", resume=False)
    assert "ContentVec" in errors[-1]


def test_runner_events_reach_the_ui(tmp_path, monkeypatch):
    """A stand-in training process reports through the same JSON-lines protocol."""
    fake = tmp_path / "fake_run.py"
    fake.write_text(
        "import json\n"
        "e = lambda **k: print(json.dumps(k), flush=True)\n"
        "e(type='stage', stage='prepare', text='Cutting')\n"
        "e(type='progress', stage='train', fraction=0.5, epoch=1, epochs=2, step=1, steps=1,"
        " elapsed=1.0)\n"
        "e(type='saved', pth='x.pth', epoch=1)\n"
        "e(type='done', pth='x.pth', onnx='x.onnx')\n"
    )
    import subprocess

    real_popen = subprocess.Popen
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, **kw: real_popen([sys.executable, str(fake)], **kw),
    )
    contentvec = tmp_path / "cv.onnx"
    contentvec.write_bytes(b"x")
    g, d = tmp_path / "g.pth", tmp_path / "d.pth"
    g.write_bytes(b"x")
    d.write_bytes(b"x")
    from ttslts.train import pretrained

    monkeypatch.setattr(pretrained, "paths_for", lambda sr, dest=None: (g, d))
    monkeypatch.setattr(Controller, "training_unavailable_reason", staticmethod(lambda: ""))

    controller = Controller(Settings(contentvec_path=str(contentvec)), save=lambda s: None)
    got = []
    controller.trainStage.connect(lambda t: got.append(("stage", t)))
    controller.trainProgress.connect(lambda p: got.append(("progress", p["fraction"])))
    controller.trainSaved.connect(lambda p, e: got.append(("saved", e)))
    controller.trainFinished.connect(lambda p, o, ok: got.append(("done", o, ok)))
    controller.error.connect(lambda m: got.append(("error", m)))
    controller.start_training(["a.wav"], "voice", resume=False)
    app = QApplication.instance() or QApplication([])
    deadline = time.time() + 20
    while len(got) < 4 and time.time() < deadline:  # signals arrive from the reader thread
        app.processEvents()
        time.sleep(0.02)
    assert got == [
        ("stage", "Cutting"),
        ("progress", 0.5),
        ("saved", 1),
        ("done", "x.onnx", True),
    ]
    job = json.loads((tmp_path / "home" / "train" / "voice" / "job.json").read_text())
    assert job["sample_rate"] == 40000 and job["files"] == ["a.wav"]
