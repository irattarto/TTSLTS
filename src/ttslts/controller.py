"""Glue between the UI and the engines. Heavy work runs on background threads."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from ttslts import onnx_util, paths
from ttslts.audio import devices
from ttslts.audio.dsp import resample
from ttslts.audio.wav import write_wav
from ttslts.settings import Settings

log = logging.getLogger(__name__)

RVC_TTS_CHUNK_S = 15  # convert long TTS output in pieces to bound memory use


class Controller(QObject):
    status = Signal(str)
    error = Signal(str)
    ttsBusy = Signal(bool)
    vcRunning = Signal(bool)
    vcLatency = Signal(float)
    voicesLoaded = Signal(list)
    kokoroDownloaded = Signal(str, str)
    rvcBaseDownloaded = Signal(str, str)
    piperDownloaded = Signal(list)
    pretrainedDownloaded = Signal(int)
    trainRuntimeInstalled = Signal()
    trainRunning = Signal(bool)
    trainStage = Signal(str)
    trainProgress = Signal(dict)
    trainSaved = Signal(str, int)
    trainFinished = Signal(str, str, bool)  # .pth, .onnx, finished (False: stopped early)

    def __init__(self, settings: Settings, save: Callable[[Settings], None] | None = None):
        super().__init__()
        self.settings = settings
        self._save = save or (lambda s: s.save())
        self._lock = threading.Lock()
        self._kokoro = None
        self._kokoro_key = None
        self._rvc = None
        self._rvc_key = None
        self._piper = None
        self._piper_key = None
        self._vc = None
        self._train_proc: subprocess.Popen | None = None
        onnx_util.set_fallback_notifier(self.status.emit)
        self._train_work: Path | None = None

    # -- settings --------------------------------------------------------
    def save_settings(self) -> None:
        try:
            self._save(self.settings)
        except OSError as exc:
            self.error.emit(f"Could not save settings: {exc}")

    # -- models ----------------------------------------------------------
    def gpu_allowed(self) -> bool:
        """Whether the app's own models may use the GPU right now.

        Training keeps the graphics card busy (and nearly all its memory), so while it runs the
        speech and voice models stay on the CPU; they move back to the GPU afterwards.
        """
        return self.settings.use_gpu and not self.training

    def kokoro(self):
        s = self.settings
        key = (s.kokoro_model_path, s.kokoro_voices_path, self.gpu_allowed())
        with self._lock:
            if self._kokoro is None or self._kokoro_key != key:
                from ttslts.tts.kokoro_engine import KokoroEngine

                self.status.emit("Loading Kokoro…")
                self._kokoro = KokoroEngine.load(*key)
                self._kokoro_key = key
            return self._kokoro

    def piper(self):
        s = self.settings
        key = (s.piper_voice, self.gpu_allowed())
        with self._lock:
            if self._piper is None or self._piper_key != key:
                from ttslts.tts import piper_download
                from ttslts.tts.piper_engine import PiperVoice

                if s.piper_voice not in piper_download.installed_voices():
                    raise ValueError(
                        "No Hungarian voice is installed "
                        "(Settings > Models > Download Hungarian voices)"
                    )
                model, config = piper_download.voice_paths(s.piper_voice)
                self.status.emit("Loading Hungarian voice…")
                self._piper = PiperVoice.load(str(model), str(config), key[1])
                self._piper_key = key
            return self._piper

    def rvc(self):
        s = self.settings
        key = (
            s.rvc_model_path,
            s.contentvec_path,
            s.rmvpe_path,
            self.gpu_allowed(),
            s.rvc_sample_rate,
        )
        with self._lock:
            if self._rvc is None or self._rvc_key != key:
                from ttslts.vc.rvc import load_rvc

                self.status.emit("Loading voice model…")
                self._rvc = load_rvc(*key)
                self._rvc.index = self._load_index(s.rvc_model_path)
                self._rvc_key = key
            # these two can be changed while the model stays loaded
            self._rvc.index_rate = s.rvc_index_rate
            self._rvc.protect = s.rvc_protect
            return self._rvc

    @staticmethod
    def _load_index(model_path: str):
        """The voice's retrieval index, made now from its training features if only those exist."""
        from ttslts.vc import retrieval

        path = retrieval.index_path(model_path)
        try:
            if not path.exists():
                voice = Path(model_path)
                features = paths.train_dir() / voice.stem / "feats"
                in_voices = voice.parent.parent == paths.voices_dir()
                if not (in_voices and features.is_dir()):
                    return None  # a voice from elsewhere has no index
                retrieval.build(features, path)
            return retrieval.RetrievalIndex.load(path)
        except (OSError, ValueError) as exc:
            log.warning("Could not use the voice index %s: %s", path, exc)
            return None

    def _background(self, fn: Callable[[], None], name: str) -> None:
        def run():
            try:
                fn()
            except Exception as exc:
                log.exception("%s failed", name)
                self.error.emit(str(exc))

        threading.Thread(target=run, daemon=True, name=name).start()

    def load_voices(self) -> None:
        def work():
            self.voicesLoaded.emit(self.kokoro().voices())
            self.status.emit("Kokoro ready")

        self._background(work, "load-voices")

    def download_kokoro(self) -> None:
        from ttslts.tts.download import download_kokoro

        def progress(name, done, total):
            pct = f" {done * 100 // total}%" if total else ""
            self.status.emit(f"Downloading {name}{pct}")

        def work():
            model, voices = download_kokoro(progress)
            self.kokoroDownloaded.emit(str(model), str(voices))
            self.status.emit("Kokoro downloaded")

        self._background(work, "download-kokoro")

    def download_piper_voices(self) -> None:
        """Fetch the Hungarian Piper voices. Call only after consent."""
        from ttslts.tts import piper_download

        def progress(name, done, total):
            pct = f" {done * 100 // total}%" if total else ""
            self.status.emit(f"Downloading {name}{pct}")

        def work():
            installed = piper_download.download_voices(progress)
            self.piperDownloaded.emit(installed)
            self.status.emit("Hungarian voices ready")

        self._background(work, "download-piper")

    def download_rvc_base(self) -> None:
        """Fetch ContentVec and RMVPE from the original RVC project. Call only after consent."""
        from ttslts.vc import rvc_base

        def progress(name, done, total):
            pct = f" {done * 100 // total}%" if total else ""
            self.status.emit(f"Downloading {name}{pct}")

        def work():
            contentvec, rmvpe = rvc_base.download_and_convert(progress, self.status.emit)
            self.rvcBaseDownloaded.emit(str(contentvec), str(rmvpe))
            self.status.emit("RVC base models ready")

        self._background(work, "download-rvc-base")

    def download_pretrained(self, sample_rate: int) -> None:
        """Fetch the pretrained voice models training starts from. Call only after consent."""
        from ttslts.train import pretrained

        def progress(name, done, total):
            pct = f" {done * 100 // total}%" if total else ""
            self.status.emit(f"Downloading {name}{pct}")

        def work():
            pretrained.download(sample_rate, progress)
            self.pretrainedDownloaded.emit(sample_rate)
            self.status.emit("Pretrained voice models ready")

        self._background(work, "download-pretrained")

    # -- training --------------------------------------------------------
    @staticmethod
    def training_runtime_ready() -> bool:
        """Whether PyTorch is set up for training (see ``train.runtime``)."""
        from ttslts.train import runtime

        return runtime.is_installed()

    def install_training_runtime(self) -> None:
        """Set up the private PyTorch environment. Call only after consent."""
        from ttslts.train import runtime

        def progress(name, done, total):
            pct = f" {done * 100 // total}%" if total else ""
            self.status.emit(f"Downloading {name}{pct}")

        def work():
            runtime.install(progress, self.status.emit)
            self.trainRuntimeInstalled.emit()
            self.status.emit("Training environment ready")

        self._background(work, "install-train-runtime")

    @property
    def training(self) -> bool:
        return self._train_proc is not None and self._train_proc.poll() is None

    def start_training(self, files: list[str], name: str, resume: bool) -> None:
        """Train a voice from the recordings in ``files`` (see ``Settings.train_*``)."""
        from ttslts.train import hparams, pretrained, runtime

        s = self.settings
        if self.training:
            return
        if not self.training_runtime_ready():
            self.error.emit("The training environment is not set up yet.")
            return
        if not files and not resume:
            self.error.emit("Add at least one recording (mp3 or wav) to train on.")
            return
        if not s.contentvec_path or not Path(s.contentvec_path).exists():
            self.error.emit(
                "Training needs ContentVec (Settings > Models > Download ContentVec + RMVPE)."
            )
            return
        if s.train_sample_rate not in hparams.SAMPLE_RATES:
            self.error.emit(f"Unsupported sample rate {s.train_sample_rate}")
            return
        g_path, d_path = pretrained.paths_for(s.train_sample_rate)
        if not (g_path.exists() and d_path.exists()):
            self.error.emit("The pretrained voice models are not downloaded yet.")
            return

        work = paths.train_dir() / name
        job = {
            "name": name,
            "files": files,
            "sample_rate": s.train_sample_rate,
            "epochs": s.train_epochs,
            "batch_size": s.train_batch_size,
            "save_every": s.train_save_every,
            "work_dir": str(work),
            "out_dir": str(paths.voices_dir() / name),
            "use_gpu": s.use_gpu,
            "fp16": True,
            "contentvec_path": s.contentvec_path,
            "rmvpe_path": s.rmvpe_path,
            "pretrained_g": str(g_path),
            "pretrained_d": str(d_path),
            "resume": resume,
        }
        work.mkdir(parents=True, exist_ok=True)
        job_file = work / "job.json"
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        log_file = open(work / "train.log", "w", encoding="utf-8")  # closed by the reader thread
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            proc = subprocess.Popen(
                [*runtime.python_command(), "-u", "-m", "ttslts.train.run", str(job_file)],
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
                encoding="utf-8",
                creationflags=flags,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", **runtime.prepare_for_run()},
            )
        except OSError as exc:
            log_file.close()
            self.error.emit(f"Could not start training: {exc}")
            return
        self._train_proc, self._train_work = proc, work
        self._release_gpu_models()
        self.trainRunning.emit(True)
        threading.Thread(
            target=self._read_training, args=(proc, log_file, work), daemon=True, name="train"
        ).start()

    def retry_gpu(self) -> None:
        """Forget which models failed on the GPU and reload everything, so all try it again."""
        onnx_util.clear_gpu_failures()
        self._release_gpu_models()
        self.status.emit("The graphics card will be tried again for every model")

    def _release_gpu_models(self) -> None:
        """Free the graphics card for training: stop the voice changer, drop loaded models."""
        if self.vc_running:
            self.stop_vc()
            self.status.emit("Voice changer stopped: the graphics card is needed for training")
        with self._lock:
            self._kokoro = self._piper = self._rvc = None
            self._kokoro_key = self._piper_key = self._rvc_key = None

    def _read_training(self, proc: subprocess.Popen, log_file, work: Path) -> None:
        last_error = ""
        ended = False
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("type")
                if kind == "stage":
                    self.trainStage.emit(event.get("text", ""))
                elif kind == "progress":
                    self.trainProgress.emit(event)
                elif kind == "saved":
                    self.trainSaved.emit(event["pth"], int(event["epoch"]))
                elif kind in ("done", "stopped"):
                    ended = True
                    self.trainFinished.emit(
                        event.get("pth", ""), event.get("onnx", ""), kind == "done"
                    )
                elif kind == "error":
                    last_error = event.get("message", "")
            proc.wait()
        finally:
            log_file.close()
            self._train_proc = None
            self.trainRunning.emit(False)
        if not ended:
            self.error.emit(
                f"Training stopped: {last_error or 'the training process ended unexpectedly'}"
                f"\n\nDetails: {work / 'train.log'}"
            )

    def stop_training(self) -> None:
        """Ask training to stop after saving; ends the process if it does not respond."""
        proc, work = self._train_proc, self._train_work
        if proc is None or work is None or proc.poll() is not None:
            return
        try:
            (work / "STOP").write_text("stop", encoding="utf-8")
        except OSError:
            proc.terminate()
            return
        self.trainStage.emit("Stopping after saving…")

        def force():
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.terminate()

        threading.Thread(target=force, daemon=True, name="train-stop").start()

    # -- text to speech --------------------------------------------------
    def render_tts(self, text: str) -> tuple[np.ndarray, int]:
        s = self.settings
        if s.tts_engine == "piper":
            audio, sr = self.piper().synthesize(
                text, s.tts_speed, noise_scale=s.piper_variation, noise_w=s.piper_rhythm
            )
        else:
            audio, sr = self.kokoro().synthesize(text, s.tts_voice, s.tts_speed, s.tts_lang)
        if s.tts_through_rvc and audio.size:
            model = self.rvc()
            self.status.emit("Applying voice model…")
            src = resample(audio, sr, 16000)
            shift = s.pitch_shift + (self._auto_pitch_shift(model, src) if s.tts_auto_pitch else 0)
            step = 16000 * RVC_TTS_CHUNK_S
            parts = [
                model.convert(src[i : i + step], shift, s.speaker_id)
                for i in range(0, len(src), step)
            ]
            audio, sr = np.concatenate(parts), model.sample_rate
        return audio, sr

    def _auto_pitch_shift(self, model, src16k: np.ndarray) -> float:
        """Semitones that bring this speech into the voice model's natural pitch range."""
        from ttslts.train import voice_info

        info = voice_info.load(self.settings.rvc_model_path)
        if info is None:  # a voice from elsewhere: only the manual Pitch shift applies
            self.status.emit(
                "This voice has no pitch information: set Pitch shift by hand if the speech "
                "sounds unnatural (about -12 for a female speech voice into a male voice)"
            )
            return 0.0
        try:
            f0 = model.pitch_fn(src16k[: 16000 * 30])
        except Exception as exc:  # pitch matching is a nicety, never a reason to fail
            log.warning("Pitch matching skipped: %s", exc)
            return 0.0
        shift = voice_info.match_shift(f0, float(info["median_f0"]))
        if shift is None:
            return 0.0
        self.status.emit(f"Pitch matched to the voice: {shift:+.1f} semitones")
        return shift

    def speak(self, text: str) -> None:
        def work():
            self.ttsBusy.emit(True)
            try:
                self.status.emit("Synthesizing…")
                audio, sr = self.render_tts(text)
                if audio.size:
                    devices.play(
                        audio, sr, devices.resolve(self.settings.tts_output_device, "output")
                    )
                self.status.emit("Speaking")
            finally:
                self.ttsBusy.emit(False)

        self._background(work, "tts")

    def save_wav(self, text: str, path: str) -> None:
        def work():
            self.ttsBusy.emit(True)
            try:
                audio, sr = self.render_tts(text)
                write_wav(path, audio, sr)
                self.status.emit(f"Saved {path}")
            finally:
                self.ttsBusy.emit(False)

        self._background(work, "tts-save")

    def stop_speaking(self) -> None:
        devices.stop()

    # -- voice changer ---------------------------------------------------
    @property
    def vc_running(self) -> bool:
        return self._vc is not None and self._vc.running

    def start_vc(self) -> None:
        if self.vc_running:
            return

        def work():
            from ttslts.vc.realtime import RealtimeVoiceChanger

            s = self.settings
            model = self.rvc()
            if model.session.get_providers()[0] == "CPUExecutionProvider":
                self.status.emit(
                    "Voice changer is running on the CPU, which is usually too slow for "
                    "real time (needs a GPU build of ONNX Runtime)"
                )
            vc = RealtimeVoiceChanger(
                model,
                devices.resolve(s.input_device, "input"),
                devices.resolve(s.output_device, "output"),
                pitch_shift=lambda: self.settings.pitch_shift,
                speaker_id=s.speaker_id,
                chunk_ms=s.chunk_ms,
                context_ms=s.context_ms,
                crossfade_ms=s.crossfade_ms,
                silence_db=s.silence_threshold_db,
                on_stats=self.vcLatency.emit,
                on_error=self._vc_failed,
            )
            vc.start()
            self._vc = vc
            self.vcRunning.emit(True)
            self.status.emit("Voice changer on")

        self._background(work, "vc-start")

    def _vc_failed(self, message: str) -> None:
        self.error.emit(f"Voice changer stopped: {message}")
        threading.Thread(target=self.stop_vc, daemon=True).start()

    def stop_vc(self) -> None:
        vc, self._vc = self._vc, None
        if vc is not None:
            vc.stop()
        self.vcRunning.emit(False)
        self.status.emit("Voice changer off")

    def toggle_vc(self) -> None:
        if self.vc_running:
            self.stop_vc()
        else:
            self.start_vc()

    def shutdown(self) -> None:
        proc = self._train_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()  # the last periodic checkpoint is kept for resuming
        self.stop_vc()
        self.stop_speaking()
