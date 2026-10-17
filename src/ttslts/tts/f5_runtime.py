"""The private PyTorch environment F5-TTS voices run in (``%APPDATA%\\TTSLTS\\runtime-f5``).

Set up the same way as the training environment (see ``train.runtime``), with F5-TTS and
faster-whisper (used to cut off the filler word, see ``f5_text``). PyTorch and torchaudio are
pinned to one matching version: F5-TTS needs torchaudio, and a torchaudio built for another
PyTorch does not load.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ttslts import paths
from ttslts.train import runtime

MODULE = "f5_tts"  # a source checkout whose Python has this needs no environment of its own
TORCH = ["torch==2.11.*", "torchaudio==2.11.*"]
PACKAGES = ["f5-tts==1.1.22", "faster-whisper==1.2.1", "transformers<5", "soundfile"]
CHECK = "import torch, f5_tts.api, faster_whisper; print(torch.cuda.is_available())"

CONSENT_TITLE = "Set up F5-TTS voices"
CONSENT_TEXT = """\
F5-TTS voices need PyTorch and F5-TTS, which are too big to ship with the app. TTSLTS will set
up its own private Python environment in its data folder (nothing is installed system-wide) and
download:

  - Python (Python Software Foundation license), from python.org, and pip
  - PyTorch and torchaudio (BSD-3-Clause), from download.pytorch.org
  - F5-TTS (MIT), faster-whisper (MIT) and what they need, from PyPI
  - on first use: the Vocos vocoder (MIT) and the Whisper "small" model (MIT), from Hugging Face

Download size: about {size} (about 5.5 GB on disk). No voice is included: you add your own
F5 voices (a voice's weights may have their own license; the Hungarian F5 base model is for
non-commercial use).
You can delete the "runtime-f5" folder in the TTSLTS data folder at any time to remove it."""


def root() -> Path:
    return paths.config_dir() / "runtime-f5"


def consent_text() -> str:
    return CONSENT_TEXT.format(size="4 GB" if runtime.has_nvidia_gpu() else "1.5 GB")


def is_installed(base: Path | None = None) -> bool:
    return runtime.is_installed(base or root(), module=MODULE)


def python_command(base: Path | None = None) -> list[str]:
    return runtime.python_command(base or root(), module=MODULE)


def prepare_for_run(base: Path | None = None) -> dict[str, str]:
    return runtime.prepare_for_run(base or root(), module=MODULE)


def install(
    progress: Callable[[str, int, int], None] | None = None,
    status: Callable[[str], None] | None = None,
    base: Path | None = None,
) -> Path:
    return runtime.install(
        progress, status, base or root(), torch_packages=TORCH, packages=PACKAGES, check=CHECK
    )
