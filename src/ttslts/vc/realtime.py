"""Real-time voice changing: microphone -> RVC -> output device."""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from math import gcd

import numpy as np

from ttslts.audio.dsp import (
    band_crossfade,
    crossfade,
    fit_length,
    highpass_filter,
    resample,
    rms_db,
    to_mono,
)

log = logging.getLogger(__name__)

SR_16K = 16000
RESAMPLE_MARGIN_16K = 80  # 5 ms
OUTPUT_MARGIN_MS = 40  # extra delay that absorbs chunks taking a little longer (see OutputBuffer)
OUTPUT_MARGIN_STEP_MS = 20
QUIET_DB = -55.0  # output this quiet may be cut to bring the delay back down
GATE_HOLD_MS = 200  # the silence gate stays open this long after the last sound above it
GATE_FRAME_MS = 20  # the gate listens to the loudest stretch of this length in a block
# During silence the model still runs this often: after about 1.5 s without work the graphics
# card slows down and the first chunk after the pause took 230-280 ms instead of 85 (measured
# on DirectML), too slow for a 200 ms chunk.
KEEP_WARM_MS = 800


class StreamConverter:
    """Turns fixed-size input blocks into converted output blocks.

    Each call converts the new block together with some preceding audio (context) so the model
    sees enough speech, and keeps only the newest part of the result. Neighbouring blocks are
    joined with SOLA: the start of the new block is moved a few milliseconds (up to ``sola_ms``)
    to where its waveform lines up best with the end of the previous block, then the two are
    crossfaded. That hides the joins that a plain crossfade leaves as a buzz. Output is delayed
    by the crossfade length (plus up to ``sola_ms``).

    ``convert(window_16k, keep_16k, position)`` must return the newest ``keep_16k / 16000``
    seconds of the converted window (a model that cannot skip may return all of it).
    ``position`` is where the window starts in the stream, in 16 kHz samples.
    """

    def __init__(
        self,
        convert: Callable[[np.ndarray, int, int], np.ndarray],
        model_sr: int,
        in_sr: int,
        out_sr: int,
        chunk_ms: int = 200,
        context_ms: int = 1500,
        crossfade_ms: int = 40,
        silence_db: float = -50.0,
        sola_ms: int = 20,  # one period of a 50 Hz voice
    ):
        self.convert = convert
        crossfade_ms = min(crossfade_ms, chunk_ms)  # a longer crossfade would not fit a chunk
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
        # a little more than is emitted, so the resampling filter's edge stays outside of it
        self.keep_16k = self.chunk_16k + self.xfade_16k + self.sola_16k + RESAMPLE_MARGIN_16K
        self.need_out = self.chunk_out + self.xfade_out + self.sola_out
        self.silence_db = silence_db
        self.window_16k = self.history_16k + self.chunk_16k
        # The microphone audio is kept at its own rate and the whole window is resampled at
        # once: resampling every block on its own leaves a small glitch at each block edge.
        down = in_sr // gcd(in_sr, SR_16K)
        raw = -(-(self.window_16k * in_sr) // SR_16K) + RESAMPLE_MARGIN_16K * in_sr // SR_16K
        self._raw = np.zeros(-(-raw // down) * down, dtype=np.float32)
        self._highpass = highpass_filter(in_sr)
        self._tail = np.zeros(self.xfade_out, dtype=np.float32)
        self._position = -self.window_16k  # stream position of the window's start (16 kHz)
        self._hold_blocks = max(1, -(-GATE_HOLD_MS // chunk_ms))
        self._open_for = 0  # blocks the gate stays open without new sound
        self._gate_frame = max(1, in_sr * GATE_FRAME_MS // 1000)
        self._keep_warm_blocks = max(1, KEEP_WARM_MS // chunk_ms)
        self._quiet_blocks = 0
        self.converted = False  # whether the last block went through the model

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

    def warm_up(self) -> float:
        """Convert one window of near silence, so the first word is not held up.

        The first runs of a GPU session set it up for that input size (1.7 s, then 0.17 s,
        measured on DirectML); doing that before the microphone opens keeps it from swallowing
        the start of the first thing said. Returns how long the last run took (seconds).
        """
        hiss = np.random.default_rng(0).normal(0.0, 1e-4, self.window_16k).astype(np.float32)
        for _ in range(3):  # the second run was still twice as slow as later ones
            started = time.perf_counter()
            self.convert(hiss, self.keep_16k, 0)
        return time.perf_counter() - started

    def _window(self, block: np.ndarray) -> np.ndarray:
        """Add a microphone block; returns the newest ``window_16k`` samples at 16 kHz."""
        filtered = fit_length(self._highpass(block), self.chunk_in)
        self._raw = np.concatenate([self._raw[len(filtered) :], filtered])
        resampled = resample(self._raw, self.in_sr, SR_16K)
        return np.ascontiguousarray(fit_length(resampled[::-1], self.window_16k)[::-1])

    def _gate_open(self, block: np.ndarray) -> bool:
        """Whether to convert this block: open while there is sound and ``GATE_HOLD_MS`` after.

        The loudest short stretch decides, not the whole block: a quiet "sz" or "t" at the end
        of a word would otherwise vanish in a block that is mostly silence.
        """
        n = len(block) // self._gate_frame
        frames = block[: n * self._gate_frame].reshape(n, -1) if n else block[None]
        loudest = max(rms_db(f) for f in frames) if len(block) else -120.0
        if loudest >= self.silence_db:
            self._open_for = self._hold_blocks
            return True
        if self._open_for > 0:
            self._open_for -= 1
            return True
        return False

    def process(self, block: np.ndarray) -> np.ndarray:
        block = to_mono(block)
        window = self._window(block)
        self._position += self.chunk_16k

        self.converted = self._gate_open(block)
        if not self.converted:
            segment = np.zeros(self.need_out, dtype=np.float32)
            self._quiet_blocks += 1
            if self._quiet_blocks % self._keep_warm_blocks == 0:
                self.convert(window, self.keep_16k, self._position)  # result not used
        else:
            self._quiet_blocks = 0
            out = self.convert(window, self.keep_16k, self._position)
            converted = resample(out, self.model_sr, self.out_sr)
            segment = converted[-self.need_out :]
            if len(segment) < self.need_out:  # short output: keep it at the end of the block
                segment = np.pad(segment, (self.need_out - len(segment), 0))

        start = self._best_offset(segment)
        piece = segment[start : start + self.chunk_out + self.xfade_out]
        emitted = piece[: self.chunk_out].copy()
        emitted[: self.xfade_out] = band_crossfade(
            self._tail, emitted[: self.xfade_out], self.out_sr
        )
        self._tail = piece[self.chunk_out :].copy()
        return emitted


class Pacer:
    """When to hand a chunk's output to the speaker: a steady time after its audio came in.

    A chunk of silence is done in a few milliseconds, one of speech takes about 100 ms. Handed
    on as soon as they are done, the first chunk of every sentence would come 100 ms "late"
    after the quick silent ones and the speaker would run dry. So every chunk, silent or not, is
    handed on ``delay`` after its audio arrived: a little more than most conversions took
    lately (their 95th percentile), but never more than ``max_s``.
    """

    def __init__(self, max_s: float, history: int = 50, spare_s: float = 0.01):
        self.max_s = max_s
        self.spare_s = spare_s
        self._times: collections.deque[float] = collections.deque(maxlen=history)

    def note(self, seconds: float) -> None:
        self._times.append(seconds)

    @property
    def delay(self) -> float:
        if not self._times:
            return 0.0
        return min(self.max_s, float(np.percentile(self._times, 95)) + self.spare_s)


class OutputBuffer:
    """Thread-safe FIFO of samples read by the output stream callback, with a safety margin.

    Converted audio arrives a whole chunk at a time, but is played continuously. Played as soon
    as it arrives, the buffer runs dry just before every next chunk, so any chunk that takes a
    little longer to convert than the one before leaves a gap (a click). So playback starts
    ``margin`` behind; whenever the buffer runs dry anyway, the sound fades out instead of
    stopping dead, the margin grows by ``margin_step`` (up to ``max_margin``) and the next chunk
    fades back in. If the buffer instead keeps more than the margin for a while (the microphone's
    clock running a little faster than the speakers', or chunks that came in a burst after a
    slow one), the surplus is cut out with a short crossfade at a quiet moment, so the delay
    does not creep up.
    """

    def __init__(
        self,
        max_samples: int,
        margin: int = 0,
        margin_step: int = 0,
        max_margin: int | None = None,
        fade: int = 0,
        watch: int = 25,
    ):
        self._lock = threading.Lock()
        self._data = np.zeros(0, dtype=np.float32)
        self.max_samples = max_samples
        self.margin = margin
        self.margin_step = margin_step
        self.max_margin = max_margin if max_margin is not None else max_samples // 2
        self.fade = fade
        self.underruns = 0
        self._priming = True  # before the first chunk and after running dry
        self._levels: collections.deque[int] = collections.deque(maxlen=watch)

    def _cut(self, data: np.ndarray, count: int) -> np.ndarray:
        """``data`` without ``count`` samples after its start, joined with a crossfade."""
        n = min(self.fade, len(data) - count)
        if n <= 0:
            return data[count:]
        joined = crossfade(data[:n], data[count : count + n])
        return np.concatenate([joined, data[count + n :]])

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        with self._lock:
            if self._priming:
                self._priming = False
                x = x.copy()
                n = min(self.fade, len(x))
                x[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                x = np.concatenate([np.zeros(self.margin, dtype=np.float32), x])
                self._levels.clear()
            else:
                self._levels.append(len(self._data))
                surplus = min(self._levels) - self.margin if self._levels else 0
                watched = len(self._levels) == self._levels.maxlen
                quiet = rms_db(self._data[: surplus + self.fade]) < QUIET_DB
                # cut where it is quiet, unless the delay has grown past the largest margin
                if watched and surplus > self.fade and (quiet or surplus > self.max_margin):
                    self._data = self._cut(self._data, surplus)
                    self._levels.clear()
            self._data = np.concatenate([self._data, x])
            if len(self._data) > self.max_samples:  # drop oldest to bound latency
                self._data = self._cut(self._data, len(self._data) - self.max_samples)

    def pull(self, n: int) -> np.ndarray:
        with self._lock:
            out, self._data = self._data[:n], self._data[n:]
            if len(out) < n and not self._priming:  # ran dry: fade out, keep a larger margin
                out = out.copy()
                k = min(self.fade, len(out))
                out[len(out) - k :] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
                self.underruns += 1
                self.margin = min(self.max_margin, self.margin + self.margin_step)
                self._priming = True
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
        context_ms: int = 1500,
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
            lambda x, keep, pos: self.model.convert(
                x, self.pitch_shift(), self.speaker_id, keep, pos
            ),
            self.model.sample_rate,
            in_sr,
            out_sr,
            self.chunk_ms,
            self.context_ms,
            self.crossfade_ms,
            self.silence_db,
        )
        pacer = Pacer(max_s=0.9 * self.chunk_ms / 1000)
        pacer.note(converter.warm_up())
        out_buf = OutputBuffer(
            max_samples=converter.chunk_out * 4,
            margin=out_sr * OUTPUT_MARGIN_MS // 1000,
            margin_step=out_sr * OUTPUT_MARGIN_STEP_MS // 1000,
            max_margin=converter.chunk_out,
            fade=out_sr * 5 // 1000,
        )

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
            target=self._run, args=(converter, out_buf, pacer), daemon=True, name="rvc-worker"
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
                    latency="low",
                ),
                sd.OutputStream(
                    device=self.output_device,
                    channels=1,
                    samplerate=out_sr,
                    dtype="float32",
                    callback=out_cb,
                    # the output buffer keeps its own small margin; with "high" an MME device
                    # adds 180 ms instead of 90 ms (measured)
                    latency="low",
                ),
            ]
            for s in self._streams:
                s.start()
        except Exception:
            self.stop()
            raise

    def _run(self, converter: StreamConverter, out_buf: OutputBuffer, pacer: Pacer) -> None:
        pending = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            try:
                pending = np.concatenate([pending, self._in_q.get(timeout=0.1)])
            except queue.Empty:
                continue
            arrived = time.perf_counter()
            while len(pending) >= converter.chunk_in:
                block, pending = pending[: converter.chunk_in], pending[converter.chunk_in :]
                started = time.perf_counter()
                try:
                    out = converter.process(block)
                except Exception as exc:
                    log.exception("Voice conversion failed")
                    if self.on_error:
                        self.on_error(str(exc))
                    self._stop.set()
                    return
                took = time.perf_counter() - started
                if converter.converted:
                    pacer.note(took)
                    if self.on_stats:
                        self.on_stats(took * 1000.0)
                wait = arrived + pacer.delay - time.perf_counter()
                if wait > 0 and len(pending) < converter.chunk_in:  # no waiting when behind
                    self._stop.wait(wait)
                out_buf.push(out)

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
