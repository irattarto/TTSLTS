"""Measures audiobook recordings for an own F5 voice, in the F5 runtime:
``python -m ttslts.train.f5_prepare job.json``. See ``f5_data`` for why each step is done.

Writes into the job's ``work_dir``:

* ``files/NNN.json`` - the sentences Whisper large-v3 found in recording NNN, with their cut;
  a recording that has one is not transcribed again (the step can be stopped and resumed)
* ``wavs/NNN_SSSSS.wav`` - every usable sentence at 24 kHz, cut at its pauses
* ``measured.json`` - all sentences with the second transcript, likeness, pitch and trembling;
  ``f5_data.select`` chooses from it. A stopped run leaves it half measured (to go on from);
  ``measured.done`` is written when every sentence is measured

Events on stdout, one JSON object a line (as ``train.run``): ``stage``, ``progress``
(``fraction`` of the whole job), ``done`` (``measured``: the file), ``stopped``, ``error``.
The job (JSON): ``files``, ``work_dir``, ``whisper_dir``, ``ecapa_dir``, ``rmvpe_path`` (may be
empty: then a simpler pitch tracker), ``device`` ("cuda" or "cpu").
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from ttslts.train import f5_data

SR = 24000  # F5's sample rate
WHISPER = "large-v3"
RECHECK = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
STOP_FILE = "STOP"
MEASURED_DONE = "measured.done"
# share of the whole job per step, for one progress bar
SHARES = {"transcribe": 0.55, "recheck": 0.25, "measure": 0.2}


def emit(**event) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


class Stopped(Exception):
    pass


def _resample(audio: np.ndarray, sr: int, to: int) -> np.ndarray:
    from ttslts.audio.dsp import resample

    return resample(audio, sr, to).astype(np.float32)


def _whisper(name: str, job: dict):
    """A Whisper model from the app's own models folder, downloaded there the first time as plain
    files (the Hugging Face cache links its files, which Windows refuses without admin rights)."""
    from faster_whisper import WhisperModel, download_model

    folder = Path(job["whisper_dir"]) / name.split("/")[-1].replace("faster-whisper-", "")
    if not (folder / "model.bin").exists():
        emit(type="stage", stage="download", text=f"Downloading Whisper {folder.name}…")
        download_model(name, output_dir=str(folder))
    if job.get("device") == "cuda" and _has_cuda():
        import torch

        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
        try:
            return WhisperModel(str(folder), device="cuda", compute_type="float16")
        except (RuntimeError, ValueError) as exc:  # no usable CUDA for CTranslate2
            emit(type="stage", stage="transcribe", text=f"Whisper runs on the CPU ({exc})")
    return WhisperModel(str(folder), device="cpu", compute_type="int8")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _write_wav(path: Path, audio: np.ndarray, soft_end: bool) -> None:
    import soundfile as sf

    piece = audio.copy()
    fade = min(len(piece), SR // 100)
    piece[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    tail = min(len(piece), SR * 4 // 100) if soft_end else fade  # read on: fade the cut word
    piece[len(piece) - tail :] *= np.linspace(1, 0, tail, dtype=np.float32)
    sf.write(str(path), np.clip(piece, -1, 1), SR, subtype="PCM_16")


def transcribe(job: dict, work: Path, stopped) -> list[dict]:
    """Whisper large-v3 over every recording: usable sentences cut at their pauses."""
    from ttslts.train.prepare import load_audio

    files_dir, wavs = work / "files", work / "wavs"
    files_dir.mkdir(parents=True, exist_ok=True)
    wavs.mkdir(parents=True, exist_ok=True)
    model = None
    clips: list[dict] = []
    share = SHARES["transcribe"]
    for n, src in enumerate(job["files"]):
        done = files_dir / f"{n:03d}.json"
        if done.exists():
            clips += json.loads(done.read_text(encoding="utf-8"))
            continue
        if stopped():
            raise Stopped
        emit(type="stage", stage="transcribe",
             text=f"Writing down recording {n + 1} of {len(job['files'])}…")  # fmt: skip
        if model is None:
            emit(type="stage", stage="transcribe", text="Loading Whisper large-v3…")
            model = _whisper(WHISPER, job)
        audio, sr = load_audio(src)
        audio = _resample(audio, sr, SR)
        segments, info = model.transcribe(
            _resample(audio, SR, 16000), language="hu", beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )  # fmt: skip
        found = []
        for seg in segments:
            if stopped():
                raise Stopped
            found.append(seg)
            heard = min(1.0, seg.end / max(info.duration, 1))  # of this recording
            emit(
                type="progress",
                stage="transcribe",
                fraction=share * (n + heard) / len(job["files"]),
            )
        quiet = f5_data.quiet_frames(audio, SR)
        kept = []
        for i, seg in enumerate(found):
            text, dur = seg.text.strip(), seg.end - seg.start
            if not f5_data.usable_segment(
                text, dur, seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio
            ):
                continue
            nxt = found[i + 1].start if i + 1 < len(found) else None
            start, end, soft = f5_data.cut_edges(quiet, seg.start, seg.end, nxt)
            name = f"{n:03d}_{i:05d}"
            _write_wav(wavs / f"{name}.wav", audio[int(start * SR) : int(end * SR)], soft)
            kept.append({
                "name": name, "file": n, "text": text, "start": round(start, 2),
                "end": round(end, 2), "dur": round(end - start, 2),
                "logprob": round(seg.avg_logprob, 3), "soft_end": soft,
            })  # fmt: skip
        done.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
        clips += kept
    del model
    return clips


def recheck(job: dict, work: Path, clips: list[dict], stopped) -> None:
    """Whisper large-v3-turbo writes every sentence down again: how well the two agree."""
    import soundfile as sf

    todo = [c for c in clips if "agree" not in c]
    if not todo:
        return
    emit(type="stage", stage="recheck", text="Checking the transcripts with a second Whisper…")
    model = _whisper(RECHECK, job)
    base = SHARES["transcribe"]
    for k, c in enumerate(todo):
        if stopped():
            raise Stopped
        audio, sr = sf.read(str(work / "wavs" / f"{c['name']}.wav"), dtype="float32")
        segs, _ = model.transcribe(
            _resample(audio, sr, 16000), language="hu", beam_size=5,
            condition_on_previous_text=False,
        )  # fmt: skip
        c["text_check"] = " ".join(s.text.strip() for s in segs)
        c["agree"] = f5_data.agreement(c["text"], c["text_check"])
        if k % 20 == 0:
            emit(type="progress", stage="recheck",
                 fraction=base + SHARES["recheck"] * k / len(todo))  # fmt: skip
    del model


def measure(job: dict, work: Path, clips: list[dict], stopped) -> None:
    """Likeness to the narrator (ECAPA), pitch and trembling of every sentence."""
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    emit(type="stage", stage="measure", text="Finding the narrator's own voice…")
    device = "cuda" if job.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_hparams(
        source=ECAPA, savedir=job["ecapa_dir"], run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )  # fmt: skip
    pitch = _pitch_tracker(job)
    base = SHARES["transcribe"] + SHARES["recheck"]
    embs = []
    for k, c in enumerate(clips):
        if stopped():
            raise Stopped
        audio, sr = sf.read(str(work / "wavs" / f"{c['name']}.wav"), dtype="float32")
        x = _resample(audio, sr, 16000)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(x)[None].to(device)).squeeze().cpu().numpy()
        embs.append(e / (np.linalg.norm(e) + 1e-12))
        f0 = pitch(x)
        c["f0"], c["wobble"] = f5_data.median_pitch(f0), f5_data.wobble(f0)
        if k % 20 == 0:
            emit(type="progress", stage="measure",
                 fraction=base + SHARES["measure"] * k / max(1, len(clips)))  # fmt: skip
    sims, core = f5_data.narrator_core(np.array(embs))
    in_core = set(core.tolist())
    for k, c in enumerate(clips):
        c["sim"], c["core"] = round(float(sims[k]), 3), k in in_core


def _pitch_tracker(job: dict):
    """RMVPE (the voice changer's pitch model) when it is there, else librosa's pYIN; both give
    one value per 10 ms of 16 kHz audio, 0 where there is no voice."""
    if job.get("rmvpe_path") and Path(job["rmvpe_path"]).exists():
        from ttslts.onnx_util import make_session
        from ttslts.vc.rmvpe_mel import RmvpeMelPitch, is_mel_model
        from ttslts.vc.rvc import RmvpePitch

        session = make_session(job["rmvpe_path"], False)
        return RmvpeMelPitch(session) if is_mel_model(session) else RmvpePitch(session)
    import librosa

    def pyin(x):
        f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=500, sr=16000, hop_length=160)
        return np.where(voiced, np.nan_to_num(f0), 0.0)

    return pyin


def run_job(job: dict) -> Path:
    work = Path(job["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    stop_file = work / STOP_FILE
    stop_file.unlink(missing_ok=True)
    (work / MEASURED_DONE).unlink(missing_ok=True)

    def stopped() -> bool:
        return stop_file.exists()

    clips = transcribe(job, work, stopped)
    if not clips:
        raise ValueError("Whisper found no usable sentences in these recordings.")
    measured = work / "measured.json"
    if measured.exists():  # a stopped job: keep what was rechecked
        old = {c["name"]: c for c in json.loads(measured.read_text(encoding="utf-8"))}
        for c in clips:
            c.update({k: v for k, v in old.get(c["name"], {}).items() if k not in c})
    try:
        recheck(job, work, clips, stopped)
        measure(job, work, clips, stopped)
    finally:
        measured.write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    (work / MEASURED_DONE).write_text("ok", encoding="utf-8")
    return measured


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        measured = run_job(job)
    except Stopped:
        emit(type="stopped")
        return 0
    except Exception as exc:  # noqa: BLE001 - reported to the app, details in the log
        traceback.print_exc()
        emit(type="error", message=str(exc))
        return 1
    emit(type="progress", stage="measure", fraction=1.0)
    emit(type="done", measured=str(measured))
    return 0


if __name__ == "__main__":
    sys.exit(main())
