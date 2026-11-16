import json

import pytest

from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.train import f5_data, f5_train
from ttslts.tts import f5_voices

VOCAB = set("abcdefghijklmnopqrstuvwxyzáéíóöőúüű ,.?!-")


def test_training_text_is_written_as_the_app_reads():
    text = f5_data.training_text("Az MTI szerint 3 alma van.", VOCAB)
    assert text == text.lower()
    assert "3" not in text and "három" in text
    assert set(text) <= VOCAB


def test_the_reference_is_a_calm_kept_sentence_of_a_good_length():
    clips = [
        {"name": "a", "text": "rövid", "dur": 3.0, "sim": 0.99, "agree": 1.0, "wobble": 10.0},
        {"name": "b", "text": "jó hosszú", "dur": 8.0, "sim": 0.90, "agree": 1.0, "wobble": 40.0},
        {"name": "c", "text": "nyugodt", "dur": 7.0, "sim": 0.90, "agree": 1.0, "wobble": 20.0},
        {"name": "d", "text": "bizonytalan", "dur": 7.0, "sim": 0.99, "agree": 0.8,
         "wobble": 10.0},
        {"name": "e", "text": "kihagyott", "dur": 8.0, "sim": 0.99, "agree": 1.0, "wobble": 1.0},
    ]  # fmt: skip
    assert f5_data.pick_reference(clips, {"a", "b", "c", "d"})["name"] == "c"
    assert f5_data.pick_reference(clips, {"a"})["name"] == "a"  # the only one
    with pytest.raises(ValueError):
        f5_data.pick_reference(clips, set())


class FakeTrainer:
    is_main = True

    def __init__(self):
        self.saved = []

    def save_checkpoint(self, update, last=False):
        self.saved.append((update, last))


def test_the_progress_bar_reports_and_stops_after_saving(tmp_path, capsys):
    t, state = FakeTrainer(), {"update": 0}
    stop = tmp_path / "STOP"
    Bar = f5_train._progress_bar(t, 2, stop, state)
    bar = Bar(range(10), desc="Epoch 2/2", initial=3)  # a resumed second epoch
    bar.set_postfix(update="13", loss=0.61234)
    bar.update(1)
    bar.update(1)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert state["update"] == 15
    assert events[-1]["update"] == 15 and events[-1]["updates"] == 20
    assert events[-1]["epoch"] == 2 and events[-1]["loss"] == 0.6123
    assert events[-1]["fraction"] == pytest.approx(0.75)
    stop.write_text("stop")
    with pytest.raises(f5_train.Stopped):
        bar.update(1)
    assert t.saved == [(16, True)]  # where it was, for the next start


def test_saved_voices_are_listed_and_installed_under_the_voice_name(tmp_path, monkeypatch):
    from ttslts import paths

    monkeypatch.setattr(paths, "train_dir", lambda: tmp_path / "train")
    voices = Controller.f5_prep_dir("hang") / "train" / "voices"
    for name in ("u002000", "last", "u001000", "u003000"):
        folder = voices / name
        folder.mkdir(parents=True)
        for f in f5_voices.FILES:
            (folder / f).write_text("x" if f != "reference.txt" else "Szöveg.", encoding="utf-8")
    (voices / "u004000").mkdir()  # still being written: no files yet
    found = [d.name for d in Controller.f5_candidates("hang")]
    assert found == ["u001000", "u002000", "u003000", "last"]
    base = tmp_path / "models"
    voice_id = f5_voices.add_voice(voices / "u002000" / "model.safetensors", base, name="hang")
    assert voice_id == "f5:hang" and f5_voices.installed_voices(base) == ["f5:hang"]


def test_f5_training_waits_for_the_chosen_sentences(tmp_path, monkeypatch):
    from ttslts import paths

    monkeypatch.setattr(paths, "train_dir", lambda: tmp_path / "train")
    c = Controller(Settings(), save=lambda _s: None)
    errors = []
    c.error.connect(errors.append)
    c.start_f5_train("hang", 40, 1000)
    assert errors and "check the sentences" in errors[0]
    assert not c.f5_training
