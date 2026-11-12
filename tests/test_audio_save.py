import numpy as np
import soundfile as sf

from ttslts.audio.wav import write_audio


def _tone(sr):
    t = np.arange(sr) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_mp3_is_written_when_the_name_ends_in_mp3(tmp_path):
    path = tmp_path / "speech.MP3"
    write_audio(path, _tone(24000), 24000)
    head = path.read_bytes()[:3]
    assert head == b"ID3" or (head[0] == 0xFF and head[1] & 0xE0 == 0xE0)  # tag or MPEG frame sync
    data, sr = sf.read(path)
    assert sr == 24000
    assert abs(len(data) - 24000) < 3000  # MP3 adds encoder padding


def test_other_names_are_saved_as_wav(tmp_path):
    path = tmp_path / "speech.wav"
    write_audio(path, _tone(22050) * 4, 22050)  # clipped, not wrapped
    data, sr = sf.read(path)
    assert sr == 22050 and path.read_bytes()[:4] == b"RIFF"
    assert data.max() <= 1.0 and data.max() > 0.99
