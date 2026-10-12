"""Downloads the Kokoro model files into the per-user models directory."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path

from ttslts import paths

_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


def download_kokoro(
    progress: Callable[[str, int, int], None] | None = None,
    dest: Path | None = None,
) -> tuple[Path, Path]:
    """Download (or reuse) the Kokoro model and voices; returns their paths."""
    dest = dest or paths.models_dir() / "kokoro"
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in KOKORO_FILES:
        target = dest / name
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(_BASE + name, timeout=60) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(name, done, total)
            tmp.replace(target)
        out.append(target)
    return out[0], out[1]
