"""Runs one training job in its own process: ``python -m ttslts.train.run job.json``.

A separate process keeps PyTorch out of the app, lets the app stop training by ending the process,
and frees all GPU memory afterwards. Progress is reported as one JSON object per line on stdout:

``{"type": "stage", "stage": "prepare" | "features" | "train" | "export", "text": ...}``
``{"type": "progress", "stage": ..., "fraction": 0..1, ...}``
``{"type": "saved", "pth": ..., "epoch": ...}``  and finally  ``{"type": "done", "pth", "onnx"}``
(``"stopped"`` instead of ``"done"`` if stopped early) or ``{"type": "error", "message": ...}``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

STOP_FILE = "STOP"  # the app creates this file in the work folder to ask for a graceful stop


def emit(**event) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def run_job(job: dict) -> None:
    from ttslts.train import hparams, prepare, trainer, voice_info
    from ttslts.vc import rvc_pth

    cfg = hparams.SAMPLE_RATES[int(job["sample_rate"])]
    work = Path(job["work_dir"])
    out = Path(job["out_dir"])
    name = job["name"]
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    resume = bool(job.get("resume")) and (work / trainer.CHECKPOINT).exists()

    if not resume:
        emit(type="stage", stage="prepare", text="Cutting the recordings into clips…")
        count = prepare.prepare_audio(
            job["files"],
            cfg.sample_rate,
            work,
            lambda f: emit(type="progress", stage="prepare", fraction=f),
        )
        emit(type="stage", stage="features", text=f"Analysing {count} clips…")
        from ttslts.onnx_util import make_session
        from ttslts.vc.rvc import ContentVec, RmvpePitch

        use_gpu = bool(job.get("use_gpu", True))
        contentvec = ContentVec(make_session(job["contentvec_path"], use_gpu))
        pitch_fn = None
        if job.get("rmvpe_path"):
            from ttslts.vc.rmvpe_mel import RmvpeMelPitch, is_mel_model

            session = make_session(job["rmvpe_path"], use_gpu)
            pitch_fn = RmvpeMelPitch(session) if is_mel_model(session) else RmvpePitch(session)
        if pitch_fn is None:
            from ttslts.audio.dsp import estimate_f0

            pitch_fn = estimate_f0
        prepare.extract_features(
            work,
            cfg.sample_rate,
            contentvec,
            pitch_fn,
            lambda f: emit(type="progress", stage="features", fraction=f),
        )

    voice_info.write(out, name, work)  # the voice's natural pitch, for matching text to speech
    emit(type="stage", stage="train", text="Training…")
    stop_file = work / STOP_FILE
    stop_file.unlink(missing_ok=True)
    pth = out / f"{name}.pth"

    def on_progress(p: dict) -> None:
        done = (p["epoch"] - 1 + p["step"] / p["steps"]) / p["epochs"]
        emit(type="progress", stage="train", fraction=done, **p)

    stopped = False
    try:
        trainer.train(
            work=work,
            cfg=cfg,
            pretrained_g=Path(job["pretrained_g"]),
            pretrained_d=Path(job["pretrained_d"]),
            epochs=int(job["epochs"]),
            batch_size=int(job["batch_size"]),
            save_every=max(1, int(job["save_every"])),
            pth=pth,
            on_save=lambda path, epoch: emit(type="saved", pth=str(path), epoch=epoch),
            on_progress=on_progress,
            cancelled=stop_file.exists,
            use_gpu=bool(job.get("use_gpu", True)),
            fp16=bool(job.get("fp16", True)),
            resume=resume,
        )
    except trainer.Cancelled:
        stopped = True
    stop_file.unlink(missing_ok=True)

    if not pth.exists():
        emit(type="stopped", pth="", onnx="")
        return
    emit(type="stage", stage="export", text="Writing the ONNX file…")
    onnx_path = out / f"{name}.onnx"
    rvc_pth.convert(str(pth), str(onnx_path))
    emit(type="stopped" if stopped else "done", pth=str(pth), onnx=str(onnx_path))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        run_job(job)
    except Exception as exc:
        log.exception("Training failed")
        emit(type="error", message=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
