import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from ttslts.train import f5_prepare

SR = 24000


def _recording(path):
    """Two "sentences" (tones) with pauses around them, 4 s."""
    t = np.arange(SR) / SR
    tone = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    quiet = np.zeros(SR // 2, np.float32)
    sf.write(str(path), np.concatenate([quiet, tone, quiet, tone, quiet]), SR)


class FakeWhisper:
    calls = 0

    def __init__(self, texts):
        self.texts = texts

    def transcribe(self, audio, **kw):
        FakeWhisper.calls += 1
        if len(audio) > 16000 * 3:  # a whole recording: two sentences, Whisper a bit early
            segs = [
                SimpleNamespace(text=self.texts[0], start=0.4, end=1.45, avg_logprob=-0.1,
                                no_speech_prob=0.0, compression_ratio=1.2),
                SimpleNamespace(text=self.texts[1], start=1.9, end=2.95, avg_logprob=-0.1,
                                no_speech_prob=0.0, compression_ratio=1.2),
            ]  # fmt: skip
            return iter(segs), SimpleNamespace(duration=len(audio) / 16000)
        return iter([SimpleNamespace(text=self.texts[0])]), None


def test_recordings_are_cut_into_sentences_and_rechecked(tmp_path, monkeypatch):
    src = tmp_path / "chapter.wav"
    _recording(src)
    monkeypatch.setattr(
        f5_prepare, "_whisper", lambda name, job: FakeWhisper(["Első mondat.", "Második mondat."])
    )
    monkeypatch.setattr(f5_prepare, "measure", lambda job, work, clips, stopped: None)
    monkeypatch.setattr(f5_prepare, "emit", lambda **e: None)
    job = {
        "files": [str(src)],
        "work_dir": str(tmp_path / "work"),
        "whisper_dir": "",
        "device": "cpu",
    }
    measured = json.loads(f5_prepare.run_job(job).read_text(encoding="utf-8"))
    assert [c["text"] for c in measured] == ["Első mondat.", "Második mondat."]
    first = measured[0]
    assert 0.45 <= first["start"] <= 0.5 and first["end"] >= 1.5  # cut at the pauses
    assert (tmp_path / "work" / "wavs" / f"{first['name']}.wav").exists()
    assert measured[0]["agree"] == 1.0 and measured[1]["agree"] < 0.7  # recheck heard "Első"
    assert (tmp_path / "work" / f5_prepare.MEASURED_DONE).exists()  # all measured
    calls = FakeWhisper.calls
    f5_prepare.run_job(job)  # again: nothing is written down twice
    assert FakeWhisper.calls == calls


def test_a_stop_file_stops_the_job(tmp_path, monkeypatch):
    src = tmp_path / "chapter.wav"
    _recording(src)
    work = tmp_path / "work"
    monkeypatch.setattr(f5_prepare, "emit", lambda **e: None)
    job = {"files": [str(src)], "work_dir": str(work), "whisper_dir": "", "device": "cpu"}
    monkeypatch.setattr(
        f5_prepare,
        "_whisper",
        lambda name, job: (work / "STOP").write_text("x") and FakeWhisper(["a", "b"]),
    )
    assert f5_prepare.main([str(_job_file(tmp_path, job))]) == 0
    assert not (work / "files" / "000.json").exists()
    assert not (work / f5_prepare.MEASURED_DONE).exists()


def _job_file(tmp_path, job):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    return p


def test_whisper_keeps_to_the_cpu_without_cuda(tmp_path, monkeypatch):
    import sys
    import types

    made = []
    fake = types.SimpleNamespace(
        WhisperModel=lambda folder, device, compute_type: made.append(device) or device,
        download_model=lambda name, output_dir: None,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(f5_prepare, "_has_cuda", lambda: False)
    (tmp_path / "large-v3").mkdir()
    (tmp_path / "large-v3" / "model.bin").write_bytes(b"x")
    assert (
        f5_prepare._whisper("large-v3", {"whisper_dir": str(tmp_path), "device": "cuda"}) == "cpu"
    )
