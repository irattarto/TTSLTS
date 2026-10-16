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
from ttslts.audio.dsp import fit_length, resample, split_at_quiet
from ttslts.audio.wav import write_wav
from ttslts.gpu_memory import GpuMemoryGate
from ttslts.settings import Settings

log = logging.getLogger(__name__)

# Convert long TTS output in pieces: the video memory a voice model holds grows with the length
# of the longest piece it has converted (0.8 GB loaded, 1.8 GB after 6 s, 3.9 GB after 15 s).
RVC_TTS_PIECE_S = 6.0
RVC_IDLE_RELEASE_S = 60.0  # the text-to-speech voice model leaves the GPU after this long unused
LIVE_GPU_NEED_MB = 1024  # free video memory the live voice changer needs (0.8 GB measured + room)
RVC_TTS_PAD_S = 0.3  # mirrored speech around each piece, so its edges are converted with context


def convert_piece(model, piece16k: np.ndarray, shift: float, speaker_id: int) -> np.ndarray:
    """Convert one piece of speech with mirrored padding on both sides, as the original RVC
    does, and cut it back to the piece's own length.

    Without padding the model works right up to the edges, where it hears nothing before or
    after, and returns about 20 ms less than it was given, so a long text came out shorter
    with every piece.
    """
    pad = min(int(RVC_TTS_PAD_S * 16000), len(piece16k) - 1)
    padded = np.pad(piece16k, (pad, pad), mode="reflect") if pad > 0 else piece16k
    out = model.convert(padded, shift, speaker_id)
    ratio = model.sample_rate / 16000
    return fit_length(out[round(max(pad, 0) * ratio) :], round(len(piece16k) * ratio))


class Controller(QObject):
    status = Signal(str)
    error = Signal(str)
    ttsBusy = Signal(bool)
    vcRunning = Signal(bool)
    vcLatency = Signal(float)
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
        self._rvc = None
        self._rvc_key_loaded = None
        self._piper = None
        self._piper_key = None
        self._vc = None
        self._train_proc: subprocess.Popen | None = None
        self._rvc_timer: threading.Timer | None = None
        self._gpu_memory = GpuMemoryGate()
        # One text to speech at a time; Stop also cancels speech that is still being made.
        self._tts_lock = threading.Lock()
        self._tts_generation = 0
        # Starting the voice changer takes seconds (loading, warm-up): a second start in that
        # time must not make a second one, and a stop must cancel the one being started.
        self._vc_lock = threading.Lock()
        self._vc_starting = False
        self._vc_cancel = False
        self._busy_jobs: set[str] = set()  # downloads and set-ups that are running
        onnx_util.set_fallback_notifier(self.status.emit)
        self._train_work: Path | None = None

    # -- settings --------------------------------------------------------
    def save_settings(self) -> None:
        try:
            self._save(self.settings)
        except OSError as exc:
            self.error.emit(f"Could not save settings: {exc}")
        vc = self._vc
        if vc is not None:  # these two also apply to a running voice changer
            vc.model.index_rate = self.settings.rvc_index_rate
            vc.model.protect = self.settings.rvc_protect

    # -- models ----------------------------------------------------------
    def gpu_allowed(self) -> bool:
        """Whether the app's own models may use the GPU right now.

        Training keeps the graphics card busy (and nearly all its memory), so while it runs the
        speech and voice models stay on the CPU; they move back to the GPU afterwards. The same
        goes while something else (a game) has filled the card's memory.
        """
        return self.settings.use_gpu and not self.training and self._gpu_memory.allowed()

    def piper(self):
        s = self.settings
        # on the CPU while the live voice changer runs: the GPU runs one model at a time, so a
        # sentence on the GPU would hold the voice changer up
        key = (s.piper_voice, self.gpu_allowed() and not self.vc_running)
        with self._lock:
            if self._piper is None or self._piper_key != key:
                from ttslts.tts import piper_download
                from ttslts.tts.piper_engine import PiperVoice

                if s.piper_voice not in piper_download.installed_voices():
                    raise ValueError(
                        "No Hungarian voice is installed "
                        "(Settings > Models > Download Hungarian voices, or Add voice)"
                    )
                model, config = piper_download.voice_paths(s.piper_voice)
                self.status.emit("Loading Hungarian voice…")
                self._piper = PiperVoice.load(str(model), str(config), key[1])
                self._piper_key = key
            return self._piper

    def _rvc_key(self, live: bool = False, gpu: bool | None = None) -> tuple:
        s = self.settings
        # While the live voice changer runs, text to speech converts on the CPU so it cannot
        # take the video memory the live model needs.
        if gpu is None:
            gpu = self.gpu_allowed() and (live or not self.vc_running)
        return (s.rvc_model_path, s.contentvec_path, s.rmvpe_path, gpu, s.rvc_sample_rate)

    def _build_rvc(self, key: tuple, live: bool = False):
        from ttslts.vc.rvc import load_rvc

        model = load_rvc(*key)
        # the live voice changer searches the index on the GPU too (always the same size there)
        model.index = self._load_index(key[0], use_gpu=live and key[3])
        model.index_rate = self.settings.rvc_index_rate
        model.protect = self.settings.rvc_protect
        return model

    def rvc(self):
        """The voice model for text to speech (loaded once, reused)."""
        s = self.settings
        key = self._rvc_key()
        with self._lock:
            if self._rvc is None or self._rvc_key_loaded != key:
                self.status.emit("Loading voice model…")
                self._rvc = self._build_rvc(key)
                self._rvc_key_loaded = key
            # these two can be changed while the model stays loaded
            self._rvc.index_rate = s.rvc_index_rate
            self._rvc.protect = s.rvc_protect
            return self._rvc

    def live_rvc(self, gpu: bool | None = None):
        """A voice model of its own for the real-time voice changer.

        A GPU session (DirectML) is only fast for the first input size it sees; every other size
        is several times slower on every run. Text to speech feeds the shared model different
        lengths all the time, so the voice changer, which always feeds the same size, gets a
        fresh model and stays fast (measured: 68 ms instead of 603 ms per chunk).
        """
        self.status.emit("Loading voice model for the voice changer…")
        return self._build_rvc(self._rvc_key(live=True, gpu=gpu), live=True)

    @staticmethod
    def _load_index(model_path: str, use_gpu: bool = False):
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
                retrieval.build_once(features, path)
            return retrieval.RetrievalIndex.load(path, use_gpu)
        except (OSError, ValueError) as exc:
            log.warning("Could not use the voice index %s: %s", path, exc)
            return None

    def _background(self, fn: Callable[[], None], name: str, once: bool = False) -> None:
        """Run ``fn`` on its own thread. ``once``: not again while it is still running (two
        downloads of the same files would write the same partial file)."""
        if once:
            with self._vc_lock:
                if name in self._busy_jobs:
                    self.status.emit("Already in progress")
                    return
                self._busy_jobs.add(name)

        def run():
            try:
                fn()
            except Exception as exc:
                log.exception("%s failed", name)
                self.error.emit(str(exc))
            finally:
                if once:
                    with self._vc_lock:
                        self._busy_jobs.discard(name)

        threading.Thread(target=run, daemon=True, name=name).start()

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

        self._background(work, "download-piper", once=True)

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

        self._background(work, "download-rvc-base", once=True)

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

        self._background(work, "download-pretrained", once=True)

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

        self._background(work, "install-train-runtime", once=True)

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

    def _release_tts_rvc(self) -> None:
        """Drop the text-to-speech voice model, which frees the video memory it holds."""
        with self._lock:
            self._rvc, self._rvc_key_loaded = None, None

    def _release_tts_models(self) -> None:
        """Drop both text-to-speech models (voice model and Piper) and their video memory."""
        with self._lock:
            self._rvc, self._rvc_key_loaded = None, None
            self._piper, self._piper_key = None, None

    def _live_gpu(self) -> bool:
        """Whether the voice changer about to start may use the GPU.

        Asked right after the text-to-speech models were freed, with a fresh reading: the
        memory check remembers the last few seconds, when those models still filled the card,
        and on its own sent the voice changer to the CPU after speech until the app restarted.
        """
        if not self.settings.use_gpu:
            self.status.emit("The GPU is switched off in Settings: the voice changer uses the CPU")
            return False
        if self.training:
            return False
        room, free = self._gpu_memory.room_for(LIVE_GPU_NEED_MB)
        if not room:
            self.status.emit(
                f"The graphics card's memory is full ({free} MB free, {LIVE_GPU_NEED_MB} MB "
                "needed): the voice changer runs on the CPU, which is too slow for real time. "
                "Close programs that use the graphics card and start it again."
            )
        return room

    def _rvc_idle_timer(self) -> None:
        """(Re)start the countdown after which the unused text-to-speech voice model is dropped."""
        if self._rvc_timer is not None:
            self._rvc_timer.cancel()
        self._rvc_timer = threading.Timer(RVC_IDLE_RELEASE_S, self._release_tts_rvc)
        self._rvc_timer.daemon = True
        self._rvc_timer.start()

    def _release_gpu_models(self) -> None:
        """Free the graphics card for training: stop the voice changer, drop loaded models."""
        if self.vc_running or self._vc_starting:  # also one that is still starting
            self.stop_vc()
            self.status.emit("Voice changer stopped: the graphics card is needed for training")
        with self._lock:
            self._piper = self._rvc = None
            self._piper_key = self._rvc_key_loaded = None

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
        from ttslts.tts import pronunciation

        s = self.settings
        text = pronunciation.apply_for(text, "hu", s.pron_builtin)
        audio, sr = self.piper().synthesize(
            text, s.tts_speed, noise_scale=s.piper_variation, noise_w=s.piper_rhythm
        )
        if s.tts_through_rvc and audio.size:
            model = self.rvc()
            self.status.emit("Applying voice model…")
            src = resample(audio, sr, 16000)
            shift = s.pitch_shift + (self._auto_pitch_shift(model, src) if s.tts_auto_pitch else 0)
            parts = [
                convert_piece(model, piece, shift, s.speaker_id)
                for piece in split_at_quiet(src, 16000, RVC_TTS_PIECE_S)
            ]
            audio, sr = np.concatenate(parts), model.sample_rate
            self._rvc_idle_timer()
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
        generation = self._tts_generation

        def work():
            self.ttsBusy.emit(True)
            try:
                with self._tts_lock:
                    if generation != self._tts_generation:
                        return  # stopped before it started
                    self.status.emit("Synthesizing…")
                    audio, sr = self.render_tts(text)
                    if generation != self._tts_generation:
                        self.status.emit("Stopped")
                        return
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
                with self._tts_lock:
                    audio, sr = self.render_tts(text)
                write_wav(path, audio, sr)
                self.status.emit(f"Saved {path}")
            finally:
                self.ttsBusy.emit(False)

        self._background(work, "tts-save")

    def stop_speaking(self) -> None:
        self._tts_generation += 1  # speech still being made is not played
        devices.stop()

    # -- voice changer ---------------------------------------------------
    @property
    def vc_running(self) -> bool:
        return self._vc is not None and self._vc.running

    def start_vc(self) -> None:
        with self._vc_lock:
            if self._vc_starting:  # still starting: Start again after a Stop means go on
                self._vc_cancel = False
                return
            if self.vc_running:
                return
            self._vc_starting, self._vc_cancel = True, False

        def work():
            try:
                start()
            except Exception:
                self.vcRunning.emit(False)  # the button must not stay on
                raise
            finally:
                with self._vc_lock:
                    self._vc_starting = False

        def start():
            from ttslts.vc.realtime import RealtimeVoiceChanger

            s = self.settings
            self._release_tts_models()  # their video memory is needed by the live model
            gpu = self._live_gpu()
            model = self.live_rvc(gpu)
            if gpu and model.session.get_providers()[0] == "CPUExecutionProvider":
                self.status.emit(
                    "Voice changer is running on the CPU, which is usually too slow for "
                    "real time (the graphics card could not be used for this model)"
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
            with self._vc_lock:
                cancelled = self._vc_cancel
                if not cancelled:
                    self._vc = vc
            if cancelled:  # Stop (or training) came while it was starting
                vc.stop()
                self.vcRunning.emit(False)
                self.status.emit("Voice changer off")
                return
            self.vcRunning.emit(True)
            self.status.emit("Voice changer on")

        self._background(work, "vc-start")

    def _vc_failed(self, message: str) -> None:
        self.error.emit(f"Voice changer stopped: {message}")
        threading.Thread(target=self.stop_vc, daemon=True).start()

    def stop_vc(self) -> None:
        with self._vc_lock:
            self._vc_cancel = self._vc_starting  # a start in progress ends as soon as it can
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
        if self._rvc_timer is not None:
            self._rvc_timer.cancel()
        proc = self._train_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()  # the last periodic checkpoint is kept for resuming
        self.stop_vc()
        self.stop_speaking()
