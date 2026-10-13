import json
import sys
import time

from PySide6.QtWidgets import QApplication

from ttslts.controller import Controller
from ttslts.settings import Settings


def test_start_requires_environment_recordings_and_contentvec(monkeypatch):
    controller = Controller(Settings(), save=lambda s: None)
    errors = []
    controller.error.connect(errors.append)
    monkeypatch.setattr(Controller, "training_runtime_ready", staticmethod(lambda: False))
    controller.start_training(["a.wav"], "voice", resume=False)
    assert "not set up" in errors[-1]
    monkeypatch.setattr(Controller, "training_runtime_ready", staticmethod(lambda: True))
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
    from ttslts.train import pretrained, runtime

    monkeypatch.setattr(pretrained, "paths_for", lambda sr, dest=None: (g, d))
    monkeypatch.setattr(Controller, "training_runtime_ready", staticmethod(lambda: True))
    monkeypatch.setattr(runtime, "python_command", lambda root=None: [sys.executable])
    monkeypatch.setattr(runtime, "prepare_for_run", lambda root=None: {})

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


def test_models_stay_off_the_gpu_while_training(monkeypatch):
    controller = Controller(Settings(use_gpu=True), save=lambda s: None)
    assert controller.gpu_allowed()
    controller._train_proc = type("P", (), {"poll": lambda self: None})()
    assert not controller.gpu_allowed()  # training is running
    controller.settings.use_gpu = False
    controller._train_proc = None
    assert not controller.gpu_allowed()  # the user switched the GPU off


def test_training_start_frees_loaded_models():
    controller = Controller(Settings(), save=lambda s: None)
    controller._kokoro, controller._kokoro_key = object(), ("a", "b", True)
    controller._rvc, controller._rvc_key = object(), ("c",)
    controller._release_gpu_models()
    assert controller._kokoro is None and controller._rvc is None and controller._rvc_key is None


def test_retry_gpu_clears_what_was_remembered(monkeypatch):
    from ttslts import onnx_util

    cleared = []
    monkeypatch.setattr(onnx_util, "clear_gpu_failures", lambda: cleared.append(True))
    controller = Controller(Settings(), save=lambda s: None)
    controller._kokoro, controller._kokoro_key = object(), ("a",)
    controller.retry_gpu()
    assert cleared and controller._kokoro is None
