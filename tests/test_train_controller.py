import json
import sys
import time

import numpy as np
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
    controller._piper, controller._piper_key = object(), ("a", True)
    controller._rvc, controller._rvc_key_loaded = object(), ("c",)
    controller._release_gpu_models()
    assert (
        controller._piper is None and controller._rvc is None and controller._rvc_key_loaded is None
    )


def test_retry_gpu_clears_what_was_remembered(monkeypatch):
    from ttslts import onnx_util

    cleared = []
    monkeypatch.setattr(onnx_util, "clear_gpu_failures", lambda: cleared.append(True))
    controller = Controller(Settings(), save=lambda s: None)
    controller._piper, controller._piper_key = object(), ("a",)
    controller.retry_gpu()
    assert cleared and controller._piper is None


def test_live_voice_changer_gets_its_own_model(monkeypatch):
    controller = Controller(Settings(rvc_model_path="v.onnx"), save=lambda s: None)
    built = []

    def build(key, live=False):
        built.append(live)
        return type("Model", (), {"index_rate": 0, "protect": 0})()

    monkeypatch.setattr(controller, "_build_rvc", build)
    tts = controller.rvc()
    assert controller.rvc() is tts and len(built) == 1  # text to speech reuses its model
    live = controller.live_rvc()
    assert live is not tts and len(built) == 2  # a GPU session is only fast for one input size
    assert controller.live_rvc() is not live  # every start gets a fresh one
    assert built == [False, True, True]  # only the live one searches its index on the GPU


def test_long_speech_is_converted_in_short_pieces(monkeypatch):
    pieces = []

    class Rvc:
        sample_rate = 40000
        pitch_fn = None

        def convert(self, audio, shift, sid):
            pieces.append(len(audio) / 16000)
            return np.zeros(10, dtype=np.float32)

    controller = Controller(
        Settings(tts_through_rvc=True, tts_auto_pitch=False), save=lambda s: None
    )
    monkeypatch.setattr(controller, "rvc", lambda: Rvc())
    monkeypatch.setattr(
        controller,
        "piper",
        lambda: type(
            "P", (), {"synthesize": lambda *a, **k: (np.ones(22050 * 40, "f") * 0.1, 22050)}
        )(),
    )
    try:
        controller.render_tts("a very long text")
    finally:
        controller.shutdown()
    assert (
        len(pieces) >= 6 and max(pieces) <= 8
    )  # not one 15 s lump: that held 3.9 GB of video memory


def test_text_to_speech_model_is_released_for_the_live_voice_changer(monkeypatch):
    controller = Controller(Settings(rvc_model_path="v.onnx"), save=lambda s: None)
    controller._rvc, controller._rvc_key_loaded = object(), ("k",)
    controller._release_tts_rvc()
    assert controller._rvc is None and controller._rvc_key_loaded is None


def test_text_to_speech_stays_on_the_cpu_while_the_live_changer_runs(monkeypatch):
    controller = Controller(Settings(use_gpu=True), save=lambda s: None)
    monkeypatch.setattr(controller._gpu_memory, "allowed", lambda: True)
    assert controller._rvc_key()[3] and controller._rvc_key(live=True)[3]
    monkeypatch.setattr(Controller, "vc_running", property(lambda self: True))
    assert not controller._rvc_key()[3]  # text to speech: CPU, so it cannot fill the card
    assert controller._rvc_key(live=True)[3]  # the live model keeps the GPU


def test_idle_text_to_speech_model_is_dropped_after_a_while(monkeypatch):
    from ttslts import controller as controller_module

    monkeypatch.setattr(controller_module, "RVC_IDLE_RELEASE_S", 0.05)
    controller = Controller(Settings(), save=lambda s: None)
    controller._rvc, controller._rvc_key_loaded = object(), ("k",)
    controller._rvc_idle_timer()
    deadline = time.time() + 5
    while controller._rvc is not None and time.time() < deadline:
        time.sleep(0.02)
    assert controller._rvc is None


def test_tts_pieces_keep_their_length_and_get_context():
    from ttslts.controller import RVC_TTS_PAD_S, convert_piece

    seen = []

    class Rvc:
        sample_rate = 40000

        def convert(self, audio, shift, sid):  # like RVC: a little shorter than the input
            seen.append(len(audio))
            return np.arange(round(len(audio) * 2.5) - 50, dtype=np.float32)

    piece = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32)
    out = convert_piece(Rvc(), piece, 0.0, 0)
    assert len(out) == 40000 * 3  # exactly as long as the piece, so long texts keep their timing
    assert seen == [len(piece) + 2 * int(RVC_TTS_PAD_S * 16000)]
    assert out[0] == round(RVC_TTS_PAD_S * 16000 * 2.5)  # the padding is cut off again
