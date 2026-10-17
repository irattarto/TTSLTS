"""F5-TTS voices of one's own: where they live and how one is added.

An F5 voice is a folder ``<models>/f5/<name>/`` with

* ``model.safetensors`` - the (fine-tuned) F5TTS_v1_Base weights,
* ``vocab.txt``         - its character vocabulary,
* ``reference.wav``     - a calm 5-15 s sample of the voice the model reads with,
* ``reference.txt``     - the exact text of that sample.

No F5 voice is downloaded or shipped: the Hungarian F5 base model is licensed for non-commercial
use only (CC BY-NC 4.0), so voices are only ever added by the user, from their own files.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ttslts import paths

PREFIX = "f5:"  # voice ids in the settings: "f5:galambos"
FILES = ("model.safetensors", "vocab.txt", "reference.wav", "reference.txt")
NOTE = (
    "your own F5-TTS voice. Smoother and clearer than Piper, but it needs the F5 environment "
    "(Settings > Models), runs on the graphics card and takes a moment per sentence."
)


class NotAnF5Voice(ValueError):
    pass


def voices_dir(base: Path | None = None) -> Path:
    return base or paths.models_dir() / "f5"


def is_f5(voice_id: str) -> bool:
    return voice_id.startswith(PREFIX)


def voice_dir(voice_id: str, base: Path | None = None) -> Path:
    return voices_dir(base) / voice_id[len(PREFIX) :]


def voice_files(voice_id: str, base: Path | None = None) -> dict[str, Path]:
    folder = voice_dir(voice_id, base)
    keys = ("model", "vocab", "ref_audio", "ref_text")
    return {key: folder / name for key, name in zip(keys, FILES, strict=True)}


def installed_voices(base: Path | None = None) -> list[str]:
    folder = voices_dir(base)
    if not folder.is_dir():
        return []
    return sorted(
        PREFIX + d.name for d in folder.iterdir() if all((d / name).is_file() for name in FILES)
    )


def display_name(voice_id: str) -> str:
    return f"{voice_id[len(PREFIX) :]} (own F5 voice)"


def add_voice(model: str | Path, base: Path | None = None) -> str:
    """Copy an F5 voice into the voices folder; returns its id.

    ``model`` is the ``.safetensors`` file; ``vocab.txt``, ``reference.wav`` and
    ``reference.txt`` must be next to it. The voice is named after that folder
    ("galambos-f5" -> "galambos").
    """
    model = Path(model)
    if model.suffix.lower() != ".safetensors" or not model.is_file():
        raise NotAnF5Voice(f"{model.name} is not an F5 voice model (.safetensors)")
    folder = model.parent
    missing = [n for n in FILES[1:] if not (folder / n).is_file()]
    if missing:
        raise NotAnF5Voice(
            f"An F5 voice needs {', '.join(missing)} next to {model.name} "
            "(the vocabulary, a voice sample and its exact text)"
        )
    if not (folder / "reference.txt").read_text(encoding="utf-8").strip():
        raise NotAnF5Voice("reference.txt is empty: it must hold the voice sample's exact text")
    name = re.sub(r"[-_ ]?f5$", "", folder.name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w.-]+", "-", name).strip("-.") or "voice"
    target = voices_dir(base) / name
    tmp = target.with_name(name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(model, tmp / FILES[0])
    for n in FILES[1:]:
        shutil.copyfile(folder / n, tmp / n)
    shutil.rmtree(target, ignore_errors=True)
    tmp.replace(target)
    return PREFIX + name
