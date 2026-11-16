"""Trains an own F5 voice in the F5 runtime: ``python -m ttslts.train.f5_train job.json``.

Fine-tunes the Hungarian F5 base model on the sentences chosen in the check step
(``f5_data.load_selection``), with the settings that made the first own voices (work notes
2026-10): learning rate 1e-5, 2400 audio frames a batch (at most 32 sentences), two batches an
update, 300 warm-up updates, 8-bit Adam and bf16, the model's activations recomputed so it fits
in 8 GB of video memory.

Writes into ``<work_dir>/train``:

* ``dataset/`` - the sentences and their texts as F5 reads them (made again when the choice changes)
* ``ckpts/`` - ``pretrained_base.pt`` (the base model in F5's checkpoint form) and
  ``model_last.pt`` (saved every ``LAST_EVERY`` updates and on Stop: a stopped run goes on there)
* ``voices/u<update>/`` - every ``save_every`` updates a voice to listen to and install: the
  averaged (EMA) weights in half precision, the vocabulary and a reference sentence

Events on stdout, one JSON object a line: ``stage``, ``progress`` (``fraction``, ``update``,
``updates``, ``epoch``, ``epochs``, ``loss``), ``saved`` (``update``, ``folder``), ``done``,
``stopped``, ``error``. The job (JSON): ``work_dir`` (the preparing step's folder), ``base_dir``
(where the base model is, downloaded there the first time), ``epochs``, ``save_every``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

BASE_REPO = "Maxdorger29/f5-tts-hungarian"  # CC BY-NC 4.0
BASE_FILES = ("model_last_final.safetensors", "vocab.txt")
STOP_FILE = "STOP"
LAST_EVERY = 250
SETTINGS = dict(
    learning_rate=1e-5,
    batch_size_per_gpu=2400,
    batch_size_type="frame",
    max_samples=32,
    grad_accumulation_steps=2,
    num_warmup_updates=300,
    bnb_optimizer=True,
)
WORKERS = 4  # data loader processes: each is a new process on Windows, 16 is too many
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MEL = dict(
    n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000,
    mel_spec_type="vocos",
)  # fmt: skip


def emit(**event) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


class Stopped(Exception):
    pass


def _patch_audio() -> None:
    """torchaudio reads and writes through soundfile (its own reader needs FFmpeg DLLs that are
    not there). At import, so the data loader processes, which import this module, get it too."""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
    except ImportError:  # the app's own Python: only the helpers below are used there
        return

    def load(fp, *_a, **_kw):
        data, sr = sf.read(str(fp), dtype="float32")
        return torch.from_numpy(data[np.newaxis, :] if data.ndim == 1 else data.T), sr

    def save(fp, wav, sr, *_a, **_kw):
        wav = wav.squeeze(0) if wav.dim() > 1 else wav
        sf.write(str(fp), wav.cpu().float().numpy(), sr)

    torchaudio.load, torchaudio.save = load, save


_patch_audio()


def base_files(base_dir: Path) -> tuple[Path, Path]:
    """The Hungarian base model and its vocabulary, downloaded the first time as plain files."""
    base_dir.mkdir(parents=True, exist_ok=True)
    if not all((base_dir / f).exists() for f in BASE_FILES):
        from huggingface_hub import hf_hub_download

        emit(type="stage", stage="download", text="Downloading the Hungarian F5 base model…")
        for f in BASE_FILES:
            hf_hub_download(BASE_REPO, f, local_dir=str(base_dir))
    return base_dir / BASE_FILES[0], base_dir / BASE_FILES[1]


def make_dataset(work: Path, out: Path, vocab_path: Path) -> list[float]:
    """The kept sentences as F5's dataset (``raw.arrow``, ``duration.json``); returns the
    durations. Made again only when the choice of sentences has changed."""
    from ttslts.train import f5_data

    selection = f5_data.load_selection(work)
    if not selection or not selection["kept"]:
        raise ValueError("Choose the sentences first (2. Check the sentences).")
    stamp = json.dumps([c["name"] for c in selection["kept"]])
    if (out / "selection.json").exists() and (out / "selection.json").read_text("utf-8") == stamp:
        return json.loads((out / "duration.json").read_text("utf-8"))["duration"]
    import soundfile as sf
    from datasets.arrow_writer import ArrowWriter

    emit(type="stage", stage="dataset", text="Writing the sentences for the training…")
    vocab = set(vocab_path.read_text(encoding="utf-8").split("\n"))
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows = []
    for c in selection["kept"]:
        text = f5_data.training_text(c["text"], vocab)
        wav = work / "wavs" / f"{c['name']}.wav"
        if text and wav.exists():
            rows.append({"audio_path": wav.as_posix(), "text": list(text),
                         "duration": float(sf.info(str(wav)).duration)})  # fmt: skip
    if not rows:
        raise ValueError("None of the chosen sentences could be used.")
    with ArrowWriter(path=str(out / "raw.arrow")) as writer:
        for row in rows:
            writer.write(row)
        writer.finalize()
    durations = [r["duration"] for r in rows]
    (out / "duration.json").write_text(json.dumps({"duration": durations}), encoding="utf-8")
    (out / "selection.json").write_text(stamp, encoding="utf-8")
    return durations


def pretrained_checkpoint(base_model: Path, ckpts: Path) -> None:
    """The base model as F5's trainer starts a fine-tune from it: the weights as its averaged
    (EMA) model, keys prefixed."""
    target = ckpts / "pretrained_base.pt"
    if target.exists():
        return
    import torch
    from safetensors.torch import load_file

    ckpts.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(base_model))
    ema = {"ema_model." + k: v.float() for k, v in weights.items()}
    ema["initted"], ema["step"] = torch.tensor(True), torch.tensor(0)
    torch.save({"ema_model_state_dict": ema}, str(target) + ".part")
    os.replace(str(target) + ".part", target)


def write_reference(work: Path, folder: Path) -> None:
    """``reference.wav`` / ``reference.txt`` of a voice: a calm kept sentence of the narrator."""
    from ttslts.train import f5_data

    kept = {c["name"] for c in f5_data.load_selection(work)["kept"]}
    clips = json.loads((work / "measured.json").read_text(encoding="utf-8"))
    ref = f5_data.pick_reference(clips, kept)
    shutil.copyfile(work / "wavs" / f"{ref['name']}.wav", folder / "reference.wav")
    (folder / "reference.txt").write_text(ref["text"], encoding="utf-8")


def export_voice(ema_state: dict, vocab: Path, work: Path, folder: Path) -> None:
    """An app voice from the averaged weights (what F5 reads with), in half precision."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    weights = {
        k[len("ema_model.") :]: v.half().contiguous()
        for k, v in ema_state.items()
        if k.startswith("ema_model.")
    }
    save_file(weights, str(folder / "model.safetensors.part"))
    os.replace(folder / "model.safetensors.part", folder / "model.safetensors")
    shutil.copyfile(vocab, folder / "vocab.txt")
    write_reference(work, folder)


def run_job(job: dict) -> Path:
    work = Path(job["work_dir"])
    train = work / "train"
    stop_file = train / STOP_FILE
    train.mkdir(parents=True, exist_ok=True)
    stop_file.unlink(missing_ok=True)
    base_model, vocab = base_files(Path(job["base_dir"]))
    durations = make_dataset(work, train / "dataset", vocab)
    ckpts, voices = train / "ckpts", train / "voices"
    pretrained_checkpoint(base_model, ckpts)
    if stop_file.exists():
        raise Stopped

    emit(type="stage", stage="load", text="Loading the F5 model…")
    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
    from f5_tts.model import CFM, DiT, trainer
    from f5_tts.model.dataset import CustomDataset
    from f5_tts.model.utils import get_tokenizer

    epochs, save_every = int(job["epochs"]), int(job["save_every"])
    vocab_map, vocab_size = get_tokenizer(str(vocab), "custom")
    model = CFM(
        transformer=DiT(**MODEL_CFG, text_num_embeds=vocab_size, mel_dim=MEL["n_mel_channels"],
                        checkpoint_activations=True),  # fits in 8 GB
        mel_spec_kwargs=MEL, vocab_char_map=vocab_map,
    )  # fmt: skip
    t = trainer.Trainer(
        model, epochs, num_warmup_updates=SETTINGS["num_warmup_updates"],
        learning_rate=SETTINGS["learning_rate"], save_per_updates=save_every,
        keep_last_n_checkpoints=0, checkpoint_path=str(ckpts),
        batch_size_per_gpu=SETTINGS["batch_size_per_gpu"],
        batch_size_type=SETTINGS["batch_size_type"], max_samples=SETTINGS["max_samples"],
        grad_accumulation_steps=SETTINGS["grad_accumulation_steps"], logger=None,
        last_per_updates=LAST_EVERY, bnb_optimizer=SETTINGS["bnb_optimizer"],
        mel_spec_type=MEL["mel_spec_type"],
    )  # fmt: skip

    save = t.save_checkpoint

    def save_checkpoint(update, last=False):
        if last:
            return save(update, last=True)
        if t.is_main:  # instead of a 5 GB training checkpoint: a voice to listen to
            folder = voices / f"u{update:06d}"
            export_voice(t.ema_model.state_dict(), vocab, work, folder)
            emit(type="saved", update=update, folder=str(folder))

    t.save_checkpoint = save_checkpoint
    state = {"update": 0}
    trainer.tqdm = _progress_bar(t, epochs, stop_file, state)

    from datasets import Dataset

    data = CustomDataset(
        Dataset.from_file(str(train / "dataset" / "raw.arrow")), durations=durations,
        preprocessed_mel=False, **{k: v for k, v in MEL.items()},
    )  # fmt: skip
    emit(type="stage", stage="train", text="Training the voice…")
    t.train(data, num_workers=WORKERS, resumable_with_seed=666)
    if t.is_main:
        export_voice(t.ema_model.state_dict(), vocab, work, voices / "last")
        emit(type="saved", update=state["update"], folder=str(voices / "last"))
    return voices


def _progress_bar(t, epochs: int, stop_file: Path, state: dict):
    """F5's progress bar (one an epoch), made to report to the app and to stop when asked, after
    saving where it is. ``state["update"]``: the updates made."""

    class Bar:
        def __init__(self, iterable=None, desc="", initial=0, **_kw):
            self.per_epoch = len(iterable) if iterable is not None else 0
            self.epoch = int(desc.split()[1].split("/")[0]) if desc.startswith("Epoch") else 1
            self.n = initial
            self.loss = None

        def update(self, n=1):
            self.n += n
            done = state["update"] = (self.epoch - 1) * self.per_epoch + self.n
            updates = self.per_epoch * epochs
            if stop_file.exists():
                t.save_checkpoint(done, last=True)
                raise Stopped
            if done % 5 == 0 or self.n == self.per_epoch:
                emit(type="progress", stage="train", fraction=min(1.0, done / max(1, updates)),
                     update=done, updates=updates, epoch=self.epoch, epochs=epochs,
                     loss=self.loss)  # fmt: skip

        def set_postfix(self, loss=None, **_kw):
            self.loss = round(float(loss), 4) if loss is not None else None

        def close(self):
            pass

    return Bar


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        voices = run_job(job)
    except Stopped:
        emit(type="stopped")
        return 0
    except Exception as exc:  # noqa: BLE001 - reported to the app, details in the log
        traceback.print_exc()
        emit(type="error", message=str(exc))
        return 1
    emit(type="progress", stage="train", fraction=1.0)
    emit(type="done", voices=str(voices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
