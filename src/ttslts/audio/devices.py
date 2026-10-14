"""Audio device lookup and simple playback, via sounddevice (PortAudio)."""

from __future__ import annotations

import logging
import threading

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
    with _PLAY_LOCK:
        _sd().play(audio, sr, device=device)


def stop() -> None:
    with _PLAY_LOCK:
        try:
            _sd().stop()
        except Exception:
            log.debug("Could not stop playback", exc_info=True)
