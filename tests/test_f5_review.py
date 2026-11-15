import json

from ttslts.train import f5_data
from ttslts.ui.f5_review_dialog import F5ReviewDialog


def _measured(work):
    clips = []
    for n in range(40):
        clips.append({"name": f"000_{n:05d}", "text": f"mondat {n}", "dur": 3.0,
                      "sim": 0.9 - n / 1000, "core": True, "f0": 110.0, "wobble": 50.0 + n,
                      "agree": 1.0})  # fmt: skip
    clips.append({"name": "000_00040", "text": "más hang", "dur": 3.0, "sim": 0.3, "core": False,
                  "f0": 200.0, "wobble": 50.0, "agree": 1.0})  # fmt: skip
    work.mkdir(parents=True)
    (work / "measured.json").write_text(json.dumps(clips), encoding="utf-8")


def test_the_check_shows_groups_plays_and_saves_the_choice(tmp_path):
    work = tmp_path / "voice"
    _measured(work)
    played = []
    dlg = F5ReviewDialog(work, played.append)
    assert "Kept 28 of 41" in dlg.report.text() and "3 not the narrator" in dlg.report.text()
    other = dlg.trees["other_voice"]
    items = [other.topLevelItem(i) for i in range(other.topLevelItemCount())]
    dlg._play(next(i for i in items if i.text(0) == "más hang"))
    assert played == [work / "wavs" / "000_00040.wav"]
    normal = dlg.result_report["kept"]
    dlg.strictness.setCurrentIndex(dlg.strictness.findData("strict"))
    assert dlg.result_report["kept"] < normal
    dlg._accept()
    saved = f5_data.load_selection(work)
    assert saved["strictness"] == "strict" and len(saved["kept"]) == dlg.result_report["kept"]
    again = F5ReviewDialog(work, played.append)  # opens with the saved strictness
    assert again.strictness_key() == "strict"
