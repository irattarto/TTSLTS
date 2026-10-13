import json

import numpy as np
import pytest

from ttslts import paths
from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.train import voice_info


def make_work(root, f0):
    (root / "f0nsf").mkdir(parents=True)
    np.save(root / "f0nsf" / "0.npy", np.asarray(f0, dtype=np.float32))
    return root


def test_median_ignores_unvoiced_frames(tmp_path):
    work = make_work(tmp_path / "w", [0] * 100 + [100.0] * 60 + [200.0] * 61)
    info = voice_info.median_f0(work)
    assert info["median_f0"] == 200.0 and info["f0_p10"] == 100.0
    assert voice_info.median_f0(make_work(tmp_path / "few", [100.0] * 10)) is None


def test_write_and_load_next_to_the_voice(tmp_path):
    work = make_work(tmp_path / "w", [120.0] * 100)
    out = tmp_path / "voices" / "anna"
    out.mkdir(parents=True)
    voice_info.write(out, "anna", work)
    assert json.loads((out / "anna.json").read_text())["median_f0"] == 120.0
    assert voice_info.load(out / "anna.onnx")["median_f0"] == 120.0
    assert voice_info.load(out / "anna.pth")["median_f0"] == 120.0
    assert voice_info.load(tmp_path / "elsewhere.onnx") is None


def test_older_voices_are_worked_out_from_their_training_folder():
    make_work(paths.train_dir() / "old", [90.0] * 100)
    folder = paths.voices_dir() / "old"
    folder.mkdir(parents=True)
    assert voice_info.load(folder / "old.onnx")["median_f0"] == 90.0
    assert (folder / "old.json").exists()  # remembered for next time


def test_match_shift_is_in_semitones_and_limited():
    assert voice_info.match_shift(np.full(50, 200.0), 100.0) == pytest.approx(-12.0)
    assert voice_info.match_shift(np.full(50, 100.0), 100.0) == pytest.approx(0.0)
    assert voice_info.match_shift(np.full(50, 800.0), 50.0) == -24.0
    assert voice_info.match_shift(np.zeros(50), 100.0) is None  # no voiced speech
    assert voice_info.match_shift(np.full(50, 100.0), 0.0) is None


class FakeRvc:
    sample_rate = 40000

    def __init__(self):
        self.shifts = []

    def pitch_fn(self, audio16k):
        return np.full(len(audio16k) // 160, 200.0, dtype=np.float32)

    def convert(self, audio16k, shift, sid):
        self.shifts.append(shift)
        return audio16k


def render(monkeypatch, tmp_path, **settings):
    folder = paths.voices_dir() / "anna"
    folder.mkdir(parents=True)
    (folder / "anna.json").write_text(json.dumps({"median_f0": 100.0}))
    controller = Controller(
        Settings(
            tts_through_rvc=True,
            rvc_model_path=str(folder / "anna.onnx"),
            pitch_shift=2,
            **settings,
        ),
        save=lambda s: None,
    )
    fake = FakeRvc()
    monkeypatch.setattr(controller, "rvc", lambda: fake)
    monkeypatch.setattr(
        controller,
        "kokoro",
        lambda: type("K", (), {"synthesize": lambda *a: (np.ones(32000, "f"), 16000)})(),
    )
    controller.render_tts("hello")
    return fake.shifts


def test_text_to_speech_is_shifted_into_the_voices_range(monkeypatch, tmp_path):
    assert render(monkeypatch, tmp_path) == [pytest.approx(-12.0 + 2)]  # 200 Hz -> 100 Hz, +manual


def test_auto_pitch_can_be_switched_off(monkeypatch, tmp_path):
    assert render(monkeypatch, tmp_path, tts_auto_pitch=False) == [2]
