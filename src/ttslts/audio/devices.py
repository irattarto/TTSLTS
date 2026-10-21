"""Audio device lookup and simple playback, via sounddevice (PortAudio)."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


def _sd():
    import sounddevice as sd

    return sd


def list_devices(kind: str) -> list[str]:
    """Names of devices with input (``kind="input"``) or output channels."""
    try:
        sd = _sd()
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        hostapis = sd.query_hostapis()
        names = []
        for d in sd.query_devices():
            if d[key] > 0:
                name = f"{d['name']} ({hostapis[d['hostapi']]['name']})"
                if name not in names:
                    names.append(name)
        return names
    except Exception as exc:
        log.warning("Could not list audio devices: %s", exc)
        return []


def resolve(name: str, kind: str) -> int | None:
    """Device index for a name from :func:`list_devices`; None means system default."""
    if not name:
        return None
    sd = _sd()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    hostapis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and f"{d['name']} ({hostapis[d['hostapi']]['name']})" == name:
            return i
    log.warning("Audio device %r not found, using the default", name)
    return None


# sounddevice's play()/stop() share one stream and are not thread-safe; speech is started on a
# background thread and stopped from the window, so they take turns.
_PLAY_LOCK = threading.Lock()


def play(audio: np.ndarray, sr: int, device: int | None) -> None:
    _stop_streams()  # new speech replaces what is playing, as sounddevice's play() does
    with _PLAY_LOCK:
        _sd().play(audio, sr, device=device)


def _stop_streams() -> None:
    with _PLAY_LOCK:
        streams = list(_STREAMS)
    for stream in streams:
        stream.stop()


def stop() -> None:
    with _PLAY_LOCK:
        try:
            _sd().stop()
        except Exception:
            log.debug("Could not stop playback", exc_info=True)
    _stop_streams()


_STREAMS: set[SpeechStream] = set()
END_SILENCE_S = 0.3  # added after the last piece of speech


class SpeechStream:
    """Plays speech piece by piece while the next pieces are still being made.

    Pieces follow each other without a gap; if the next one is not ready yet the stream plays
    silence until it is. ``finish`` says no more are coming: the stream then ends after the last
    one. :func:`stop` (the Stop button) ends every stream at once.
    """

    def __init__(self, sr: int, device: int | None):
        import collections

        self.sr = sr
        self._pieces: collections.deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._finished = False
        self.done = threading.Event()
        stop()  # new speech replaces what is playing
        sd = _sd()
        self._stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._fill,
            finished_callback=self.done.set,
        )
        with _PLAY_LOCK:
            _STREAMS.add(self)
        self._stream.start()

    def add(self, audio: np.ndarray) -> None:
        with self._lock:
            self._pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    def finish(self) -> None:
        with self._lock:
            # a little silence at the end: the last word was heard cut and rushed, as the end of
            # the speech was still in the device's buffer when the stream was closed
            self._pieces.append(np.zeros(int(END_SILENCE_S * self.sr), dtype=np.float32))
            self._finished = True

        def close_when_played():  # not from PortAudio's own thread, which must not close it
            self.done.wait()
            stream = self._stream
            if stream is not None:  # what the device still holds is played out (MME: ~0.2 s)
                time.sleep(float(getattr(stream, "latency", 0.0) or 0.0) + 0.1)
            self.stop()

        threading.Thread(target=close_when_played, daemon=True, name="speech-close").start()

    def _fill(self, outdata, frames, _time, _status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._pieces:
                piece = self._pieces[0]
                take = min(frames - filled, len(piece))
                out[filled : filled + take] = piece[:take]
                filled += take
                if take == len(piece):
                    self._pieces.popleft()
                else:
                    self._pieces[0] = piece[take:]
            ended = self._finished and not self._pieces
        out[filled:] = 0.0
        if ended:
            raise _sd().CallbackStop

    def stop(self) -> None:
        with self._lock:
            self._pieces.clear()
            self._finished = True
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                log.debug("Could not stop the speech stream", exc_info=True)
        self.done.set()
        with _PLAY_LOCK:
            _STREAMS.discard(self)
