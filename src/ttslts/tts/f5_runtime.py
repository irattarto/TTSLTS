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


# Training an own F5 voice (``train.f5_prepare``) also needs the narrator finder: SpeechBrain
# (Apache-2.0) with its ECAPA speaker model, added to this environment on the first training.
TRAINING_PACKAGES = ["speechbrain>=1.0,<2"]
TRAINING_CHECK = "import speechbrain.inference.speaker"
TRAINING_MARKER = "training-tools.json"
TRAINING_CONSENT_TITLE = "Set up training F5 voices"
TRAINING_CONSENT_TEXT = """\
Training an own F5 voice writes down and measures your recordings first. For that TTSLTS adds
to its F5 environment and downloads, now or on first use:

  - SpeechBrain (Apache-2.0), from PyPI, and its ECAPA speaker model (Apache-2.0), from
    Hugging Face: it tells the narrator's own voice from the characters he or she plays
  - Whisper large-v3 and large-v3-turbo (MIT), from Hugging Face: they write the recordings
    down (about 4.6 GB together, into the TTSLTS models folder)
  - when the training starts: the Hungarian F5 base model (CC BY-NC 4.0, non-commercial use
    only), from Hugging Face (about 0.7 GB): the own voice is trained from it

Only use recordings of a voice you have permission to use."""


def training_tools_ready(base: Path | None = None) -> bool:
    base = base or root()
    return is_installed(base) and (base / TRAINING_MARKER).exists()


def install_training_tools(
    status: Callable[[str], None] | None = None, base: Path | None = None
) -> None:
    """Add what training an own F5 voice needs to the F5 environment (a small download)."""
    import json
    import os
    import subprocess

    base = base or root()
    python = python_command(base)
    say = status or (lambda _text: None)
    say("Installing the tools for training F5 voices…")
    runtime._run(
        [*python, "-m", "pip", "install", "--no-warn-script-location",
         "--disable-pip-version-check", *TRAINING_PACKAGES],
        status,
    )  # fmt: skip
    check = subprocess.run(
        [*python, "-c", TRAINING_CHECK], capture_output=True, text=True,
        env={**os.environ, **prepare_for_run(base)},
        creationflags=runtime._no_window(),
    )  # fmt: skip
    if check.returncode != 0:
        raise RuntimeError("The F5 training tools did not start:\n" + check.stderr[-600:])
    (base / TRAINING_MARKER).write_text(json.dumps({"packages": TRAINING_PACKAGES}), "utf-8")
