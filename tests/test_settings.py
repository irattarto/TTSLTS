import json

from ttslts import paths
from ttslts.settings import Settings


def test_tray_options_are_checked_by_default():
    s = Settings()
    assert s.minimize_to_tray is True
    assert s.close_to_tray is True


def test_missing_file_gives_defaults():
    assert Settings.load() == Settings()


def test_round_trip():
    s = Settings(close_to_tray=False, pitch_shift=-5, tts_speed=1.25, rvc_model_path="C:/v.onnx")
    s.save()
    assert paths.settings_file().exists()
    assert Settings.load() == s


def test_invalid_values_fall_back_to_defaults(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(
        json.dumps(
            {
                "minimize_to_tray": "no",  # wrong type
                "pitch_shift": True,  # bool is not an int here
                "tts_speed": 1,  # int is fine for a float
                "unknown": 1,
            }
        )
    )
    s = Settings.load(f)
    assert s.minimize_to_tray is True
    assert s.pitch_shift == 0
    assert s.tts_speed == 1.0 and isinstance(s.tts_speed, float)


def test_corrupt_file_gives_defaults(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("{not json")
    assert Settings.load(f) == Settings()
