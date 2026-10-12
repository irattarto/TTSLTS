"""Glue between the UI and the engines. Heavy work runs on background threads."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

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

    # -- settings --------------------------------------------------------
    def save_settings(self) -> None:
        try:
            self._save(self.settings)
        except OSError as exc:
            self.error.emit(f"Could not save settings: {exc}")

    # -- models ----------------------------------------------------------
    def kokoro(self):
        s = self.settings
        key = (s.kokoro_model_path, s.kokoro_voices_path, s.use_gpu)
        with self._lock:
            if self._kokoro is None or self._kokoro_key != key:
                from ttslts.tts.kokoro_engine import KokoroEngine

                self.status.emit("Loading Kokoro…")
                self._kokoro = KokoroEngine.load(*key)
                self._kokoro_key = key
            return self._kokoro

    def piper(self):
        s = self.settings
        key = (s.piper_voice, s.use_gpu)
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
                self._piper = PiperVoice.load(str(model), str(config), s.use_gpu)
                self._piper_key = key
            return self._piper

    def rvc(self):
        s = self.settings
        key = (s.rvc_model_path, s.contentvec_path, s.rmvpe_path, s.use_gpu, s.rvc_sample_rate)
        with self._lock:
            if self._rvc is None or self._rvc_key != key:
                from ttslts.vc.rvc import load_rvc

                self.status.emit("Loading voice model…")
                self._rvc = load_rvc(*key)
                self._rvc_key = key
            return self._rvc

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
            step = 16000 * RVC_TTS_CHUNK_S
            parts = [
                model.convert(src[i : i + step], s.pitch_shift, s.speaker_id)
                for i in range(0, len(src), step)
            ]
            audio, sr = np.concatenate(parts), model.sample_rate
        return audio, sr

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
            vc = RealtimeVoiceChanger(
                self.rvc(),
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
        self.stop_vc()
        self.stop_speaking()
