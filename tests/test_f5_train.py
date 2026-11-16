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


def _prepared_set(work, src):
    """An F5 set as the preparing step leaves it: one recording, three sentences, two kept."""
    (work / "files").mkdir(parents=True)
    (work / "prepare.json").write_text(json.dumps({"files": [str(src)]}), encoding="utf-8")
    found = [
        {"name": "000_00000", "file": 0, "text": "egy", "start": 0.5, "end": 1.5, "dur": 1.0},
        {"name": "000_00001", "file": 0, "text": "kettő", "start": 2.0, "end": 7.0, "dur": 5.0},
        {"name": "000_00002", "file": 0, "text": "három", "start": 7.5, "end": 8.5, "dur": 1.0},
    ]
    (work / "files" / "000.json").write_text(json.dumps(found), encoding="utf-8")
    for c in found:
        c["keep"] = c["name"] != "000_00002"
    f5_data.save_selection(work, "normal", found, {"kept": 2, "kept_minutes": 0.1})


def test_a_voice_changer_voice_is_cut_from_the_chosen_sentences(tmp_path):
    import numpy as np
    import soundfile as sf

    from ttslts.train import prepare

    src = tmp_path / "chapter.wav"
    sr = 22050
    t = np.arange(9 * sr) / sr
    sf.write(str(src), (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    work = tmp_path / "f5"
    _prepared_set(work, src)
    spans = f5_data.kept_spans(work)
    assert spans == [
        {"file": str(src), "start": 0.5, "end": 1.5},
        {"file": str(src), "start": 2.0, "end": 7.0},
    ]  # the left-out third sentence is not there
    out = tmp_path / "rvc"
    count = prepare.prepare_spans(spans, 40000, out)
    lengths = sorted(len(np.load(out / "wavs" / f"{n}.npy")) / 40000 for n in range(count))
    assert count == 3  # 1.06 s, and the 5 s sentence cut into two clips
    assert lengths[0] == pytest.approx(1.06, abs=0.01) and max(lengths) <= prepare.CLIP_S + 0.01


def test_no_chosen_sentences_no_voice_changer_training(tmp_path, monkeypatch):
    from ttslts import paths

    monkeypatch.setattr(paths, "train_dir", lambda: tmp_path / "train")
    monkeypatch.setattr(Controller, "training_runtime_ready", staticmethod(lambda: True))
    c = Controller(Settings(), save=lambda _s: None)
    errors = []
    c.error.connect(errors.append)
    c.start_training([], "hang", resume=False, from_f5=True)
    assert errors and "Choose the sentences first" in errors[0]
    assert not c.training
