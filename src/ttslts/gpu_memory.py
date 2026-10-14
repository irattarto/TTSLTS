"""Whether the graphics card has room for the app's models right now.

A full card makes GPU runs fail (out of memory) at random moments, for example while a game or a
training job holds it. This reads the free video memory (NVIDIA cards, through ``nvidia-smi``;
other makers cannot be read this way, so there the answer is always "yes" and the GPU fallback
handles an overflow) and answers with some hysteresis, so the models are not moved back and forth
when the free memory hovers around the limit.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable

START_MB = 1536  # free memory needed before models are put on the GPU
STAY_MB = 512  # models already there stay until less than this is free
CHECK_EVERY_S = 5.0


def parse_free_mb(output: str) -> int | None:
    """Free memory in MB of the first card in ``nvidia-smi --query-gpu=memory.free`` output."""
    for line in output.splitlines():
        try:
            return int(float(line.split(",")[0].strip()))
        except ValueError:
            continue
    return None


def read_free_mb() -> int | None:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_free_mb(done.stdout) if done.returncode == 0 else None


class GpuMemoryGate:
    def __init__(
        self,
        read: Callable[[], int | None] = read_free_mb,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._read = read
        self._clock = clock
        self._sleep = sleep
        self._ok = True
        self._checked = float("-inf")

    def room_for(
        self, need_mb: int, tries: int = 6, wait_s: float = 0.5
    ) -> tuple[bool, int | None]:
        """Whether ``need_mb`` is free right now, read fresh (not the last answer).

        Meant for right after the app freed models of its own: the card can take a moment to
        report that memory as free, so it is read a few times. Returns (room, free MB).
        """
        free = None
        for attempt in range(tries):
            free = self._read()
            if free is None or free >= need_mb:
                break
            if attempt + 1 < tries:
                self._sleep(wait_s)
        room = free is None or free >= need_mb
        if room:  # the card is fine again: models may use it from now on
            self._ok, self._checked = True, self._clock()
        return room, free

    def allowed(self) -> bool:
        """True if models may be on the GPU. Unknown free memory counts as allowed."""
        now = self._clock()
        if now - self._checked >= CHECK_EVERY_S:
            self._checked = now
            free = self._read()
            if free is None:
                self._ok = True
            else:
                self._ok = free >= (STAY_MB if self._ok else START_MB)
        return self._ok
