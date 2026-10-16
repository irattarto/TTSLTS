import json

import numpy as np

from ttslts import paths
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
