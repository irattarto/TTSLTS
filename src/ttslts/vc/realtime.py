"""Real-time voice changing: microphone -> RVC -> output device."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from ttslts.audio.dsp import crossfade, fit_length, resample, rms_db, to_mono

log = logging.getLogger(__name__)

SR_16K = 16000


class StreamConverter:
    """Turns fixed-size input blocks into converted output blocks.

    Each call converts the new block together with some preceding audio (context) so the model
    sees enough speech, and keeps only the newest part of the result. Neighbouring blocks are
    joined with SOLA: the start of the new block is moved a few milliseconds (up to ``sola_ms``)
    to where its waveform lines up best with the end of the previous block, then the two are
    crossfaded. That hides the joins that a plain crossfade leaves as a buzz. Output is delayed
    by the crossfade length (plus up to ``sola_ms``).

    ``convert(window_16k, keep_16k)`` must return the newest ``keep_16k / 16000`` seconds of the
    converted window (a model that cannot skip may return all of it).
    """

    def __init__(
        self,
        convert: Callable[[np.ndarray, int], np.ndarray],
        model_sr: int,
        in_sr: int,
        out_sr: int,
        chunk_ms: int = 200,
        context_ms: int = 600,
        crossfade_ms: int = 40,
        silence_db: float = -50.0,
        sola_ms: int = 12,
    ):
        self.convert = convert
        self.model_sr = model_sr
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.chunk_in = in_sr * chunk_ms // 1000
        self.chunk_16k = SR_16K * chunk_ms // 1000
        self.chunk_out = out_sr * chunk_ms // 1000
        self.xfade_16k = SR_16K * crossfade_ms // 1000
        self.xfade_out = out_sr * crossfade_ms // 1000
        self.sola_16k = SR_16K * sola_ms // 1000
        self.sola_out = out_sr * sola_ms // 1000
        self.history_16k = SR_16K * context_ms // 1000 + self.xfade_16k + self.sola_16k
        self.keep_16k = self.chunk_16k + self.xfade_16k + self.sola_16k
        self.need_out = self.chunk_out + self.xfade_out + self.sola_out
        self.silence_db = silence_db
        self._history = np.zeros(self.history_16k, dtype=np.float32)
        self._tail = np.zeros(self.xfade_out, dtype=np.float32)

    def _best_offset(self, segment: np.ndarray) -> int:
        """Start (0..sola_out) in ``segment`` whose waveform best continues the previous tail."""
        if self.sola_out == 0 or self.xfade_out == 0:
            return self.sola_out
        tail = self._tail
        if float(np.dot(tail, tail)) < 1e-9:
            return self.sola_out  # nothing to line up with (start, or after silence)
        span = segment[: self.sola_out + self.xfade_out]
        corr = np.correlate(span, tail, mode="valid")
        energy = np.convolve(span * span, np.ones(self.xfade_out, dtype=np.float32), "valid")
        score = corr / np.sqrt(energy + 1e-9)
        # among equally good offsets prefer the latest, which is the least delayed
        return self.sola_out - int(np.argmax(score[::-1]))

    def process(self, block: np.ndarray) -> np.ndarray:
        block = to_mono(block)
        new16k = fit_length(resample(block, self.in_sr, SR_16K), self.chunk_16k)
        window = np.concatenate([self._history, new16k])
        self._history = window[-self.history_16k :] if self.history_16k else self._history

        if rms_db(block) < self.silence_db:
            segment = np.zeros(self.need_out, dtype=np.float32)
        else:
            converted = resample(self.convert(window, self.keep_16k), self.model_sr, self.out_sr)
            segment = converted[-self.need_out :]
            if len(segment) < self.need_out:  # short output: keep it at the end of the block
                segment = np.pad(segment, (self.need_out - len(segment), 0))

        start = self._best_offset(segment)
        piece = segment[start : start + self.chunk_out + self.xfade_out]
        emitted = piece[: self.chunk_out].copy()
        n = min(self.xfade_out, self.chunk_out)
        emitted[:n] = crossfade(self._tail[:n], emitted[:n])
        self._tail = piece[self.chunk_out :].copy()
        return emitted


class OutputBuffer:
    """Thread-safe FIFO of samples read by the output stream callback."""

    def __init__(self, max_samples: int):
        self._lock = threading.Lock()
        self._data = np.zeros(0, dtype=np.float32)
        self.max_samples = max_samples

    def push(self, x: np.ndarray) -> None:
        with self._lock:
            self._data = np.concatenate([self._data, x])
            if len(self._data) > self.max_samples:  # drop oldest to bound latency
                self._data = self._data[-self.max_samples :]

    def pull(self, n: int) -> np.ndarray:
        with self._lock:
            out, self._data = self._data[:n], self._data[n:]
        return fit_length(out, n)


class RealtimeVoiceChanger:
    """Owns the audio streams and the conversion worker thread."""

    def __init__(
        self,
        model,
        input_device: int | None,
        output_device: int | None,
        pitch_shift: Callable[[], float],
        speaker_id: int = 0,
        chunk_ms: int = 200,
        context_ms: int = 600,
        crossfade_ms: int = 40,
        silence_db: float = -50.0,
        on_stats: Callable[[float], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.input_device = input_device
        self.output_device = output_device
        self.pitch_shift = pitch_shift
        self.speaker_id = speaker_id
        self.chunk_ms = chunk_ms
        self.context_ms = context_ms
        self.crossfade_ms = crossfade_ms
        self.silence_db = silence_db
        self.on_stats = on_stats
        self.on_error = on_error
        self._in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._streams: list = []

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        import sounddevice as sd

        in_sr = int(sd.query_devices(self.input_device, "input")["default_samplerate"])
        out_sr = int(sd.query_devices(self.output_device, "output")["default_samplerate"])
        converter = StreamConverter(
            lambda x, keep: self.model.convert(x, self.pitch_shift(), self.speaker_id, keep),
            self.model.sample_rate,
            in_sr,
            out_sr,
            self.chunk_ms,
            self.context_ms,
            self.crossfade_ms,
            self.silence_db,
        )
        out_buf = OutputBuffer(max_samples=converter.chunk_out * 4)

        def in_cb(indata, frames, t, status):
            try:
                self._in_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        def out_cb(outdata, frames, t, status):
            outdata[:, 0] = out_buf.pull(frames)
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]

        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, args=(converter, out_buf), daemon=True, name="rvc-worker"
        )
        self._worker.start()
        try:
            self._streams = [
                sd.InputStream(
                    device=self.input_device,
                    channels=1,
                    samplerate=in_sr,
                    dtype="float32",
                    callback=in_cb,
                    blocksize=converter.chunk_in // 4,
                ),
                sd.OutputStream(
                    device=self.output_device,
                    channels=1,
                    samplerate=out_sr,
                    dtype="float32",
                    callback=out_cb,
                ),
            ]
            for s in self._streams:
                s.start()
        except Exception:
            self.stop()
            raise

    def _run(self, converter: StreamConverter, out_buf: OutputBuffer) -> None:
        pending = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            try:
                pending = np.concatenate([pending, self._in_q.get(timeout=0.1)])
            except queue.Empty:
                continue
            while len(pending) >= converter.chunk_in:
                block, pending = pending[: converter.chunk_in], pending[converter.chunk_in :]
                started = time.perf_counter()
                try:
                    out_buf.push(converter.process(block))
                except Exception as exc:
                    log.exception("Voice conversion failed")
                    if self.on_error:
                        self.on_error(str(exc))
                    self._stop.set()
                    return
                if self.on_stats:
                    self.on_stats((time.perf_counter() - started) * 1000.0)

    def stop(self) -> None:
        self._stop.set()
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                log.debug("Error closing stream", exc_info=True)
        self._streams = []
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
        self._worker = None
