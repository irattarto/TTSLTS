import numpy as np
import pytest

from ttslts.train import prepare

SR = 16000


def speechlike(seconds: float, sr: int = SR, freq: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )


def test_split_at_pauses_cuts_in_the_middle_of_long_silences():
    audio = np.concatenate([speechlike(1.0), np.zeros(SR), speechlike(1.0)])
    parts = prepare.split_at_pauses(audio, SR)
    assert len(parts) == 2
    assert sum(len(p) for p in parts) == len(audio)
    assert abs(len(parts[0]) - 1.5 * SR) < 0.05 * SR


def test_short_silences_do_not_split():
    audio = np.concatenate([speechlike(1.0), np.zeros(int(0.2 * SR)), speechlike(1.0)])
    assert len(prepare.split_at_pauses(audio, SR)) == 1


def test_clips_overlap_and_drop_scraps():
    long = speechlike(10.0)
    pieces = prepare.clips(long, SR)
    assert len(pieces) >= 3
    assert all(prepare.MIN_CLIP_S * SR <= len(p) <= prepare.CLIP_S * SR for p in pieces)
    assert prepare.clips(speechlike(0.5), SR) == []


def test_normalise_levels_and_rejects_damaged_clips():
    out = prepare.normalise(speechlike(1.0) * 0.1)
    assert 0.5 < np.abs(out).max() < 1.0
    assert prepare.normalise(np.full(1000, 3.0, dtype=np.float32)) is None
    assert prepare.normalise(np.zeros(1000, dtype=np.float32)) is None


def test_prepare_audio_resamples_and_writes_clips(tmp_path):
    sf = pytest.importorskip("soundfile")
    src = tmp_path / "voice.wav"
    sf.write(
        str(src),
        np.concatenate([speechlike(5.0, 22050), np.zeros(22050), speechlike(3.0, 22050)]),
        22050,
    )
    work = tmp_path / "work"
    count = prepare.prepare_audio([str(src)], 40000, work)
    files = sorted((work / "wavs").glob("*.npy"))
    assert count == len(files) >= 3
    assert all(np.load(f).dtype == np.float32 and len(np.load(f)) >= 40000 for f in files)


def test_mp3_is_read(tmp_path):
    sf = pytest.importorskip("soundfile")
    if "MP3" not in sf.available_formats():
        pytest.skip("this libsndfile cannot read mp3")
    src = tmp_path / "voice.mp3"
    sf.write(str(src), speechlike(2.0, 24000), 24000)
    audio, sr = prepare.load_audio(src)
    assert sr == 24000 and audio.ndim == 1 and abs(len(audio) - 48000) < 4000


def test_unreadable_file_reports_its_name(tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(prepare.AudioError, match="broken.mp3"):
        prepare.load_audio(bad)


def test_silent_recording_is_rejected(tmp_path):
    sf = pytest.importorskip("soundfile")
    src = tmp_path / "silence.wav"
    sf.write(str(src), np.zeros(SR * 3, dtype=np.float32), SR)
    with pytest.raises(prepare.AudioError, match="No usable speech"):
        prepare.prepare_audio([str(src)], 40000, tmp_path / "work")


def test_extract_features_writes_aligned_frames(tmp_path):
    work = tmp_path / "work"
    (work / "wavs").mkdir(parents=True)
    np.save(work / "wavs" / "0.npy", speechlike(2.0, 40000))

    def contentvec(audio16k):
        return np.ones((1, len(audio16k) // 320, 768), dtype=np.float32)

    def pitch(audio16k):
        return np.full(len(audio16k) // 160 + 1, 200.0, dtype=np.float32)

    prepare.extract_features(work, 40000, contentvec, pitch)
    feats = np.load(work / "feats" / "0.npy")
    coarse = np.load(work / "f0" / "0.npy")
    f0 = np.load(work / "f0nsf" / "0.npy")
    assert feats.shape == (100, 768)
    assert len(coarse) == len(f0) == 201
    assert coarse.dtype == np.int64 and 1 <= coarse.min() and coarse.max() <= 255
