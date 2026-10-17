"""Downloads Piper's Hungarian voices from the original voice collection.

Nothing is redistributed: the files come straight from ``rhasspy/piper-voices`` on Hugging Face,
pinned to a fixed revision and size-checked, to the user's own machine.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ttslts import paths

REPO = "rhasspy/piper-voices"
REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"  # pinned so the size checks below stay valid
SOURCE_URL = f"https://huggingface.co/{REPO}"

# voice id -> (folder, display name, {file suffix: expected bytes})
_ONNX_BYTES = 63201294
VOICES: dict[str, tuple[str, str, dict[str, int]]] = {
    "hu_HU-anna-medium": (
        "hu/hu_HU/anna/medium",
        "Anna",
        {".onnx": _ONNX_BYTES, ".onnx.json": 5018, ".MODEL_CARD": 278},
    ),
    "hu_HU-berta-medium": (
        "hu/hu_HU/berta/medium",
        "Berta",
        {".onnx": _ONNX_BYTES, ".onnx.json": 4961, ".MODEL_CARD": 279},
    ),
    "hu_HU-imre-medium": (
        "hu/hu_HU/imre/medium",
        "Imre",
        {".onnx": _ONNX_BYTES, ".onnx.json": 5019, ".MODEL_CARD": 278},
    ),
}

# What to expect from each voice, shown as hover help. All three are single-speaker, 22.05 kHz,
# medium-quality voices; the pitch figures are the median speaking pitch measured in CI on a test
# sentence at normal speed.
VOICE_NOTES: dict[str, str] = {
    "hu_HU-anna-medium": (
        "higher-pitched voice (about 175 Hz when speaking), very close to Berta. "
        "Hungarian, medium quality."
    ),
    "hu_HU-berta-medium": (
        "higher-pitched voice (about 170 Hz when speaking), very close to Anna. "
        "Hungarian, medium quality."
    ),
    "hu_HU-imre-medium": (
        "lower-pitched voice (about 115 Hz when speaking), the deepest of the three. "
        "Hungarian, medium quality."
    ),
}

HELP_VOICE = (
    "Three different Hungarian speakers. They sound different (pitch and character), so "
    "listen to each one with your own text and keep the one you like. Hover a name for "
    "details."
)
HELP_VARIATION = (
    "How much the voice changes from word to word.\nLower: steadier, more monotone.\n"
    "Higher: more expressive, but can get unstable.\nDefault 0.67."
)
HELP_RHYTHM = (
    "How much the timing of syllables varies.\nLower: even, metronome-like.\n"
    "Higher: livelier, but can sound uneven.\nDefault 0.80."
)
HELP_SPEED = (
    "Speaking speed. 1.0 is normal. Slightly slower (0.9) can help with clarity.\n"
    "F5 voices read at most at 1.1: faster, they drop short words; above that only the pauses "
    "between sentences get shorter."
)
HINT = (
    "Tip: try all three voices first. If it sounds flat, raise Voice variation a little; if it "
    "sounds shaky, lower it. Hover any setting for details."
)

CONSENT_TITLE = "Hungarian voices (Piper): terms of use"
CONSENT_TEXT = f"""\
The Hungarian voices Anna, Berta and Imre are downloaded from the original Piper voice
collection:
{SOURCE_URL}

Their model cards state the training data is CC0 (public domain). The voices were then
fine-tuned from Piper's US English "lessac" voice, and the training data of that voice (the
Blizzard 2013 data from Lessac Technologies) is licensed for research use. The Piper project
adds no license of its own to the voices and leaves it to users to respect the original ones.

By continuing you accept this: use these voices for research and personal experiments, and
do not use them for commercial products. The model cards are saved next to the files.

Download size: about 190 MB."""


CUSTOM_NOTE = "your own Piper voice (added from a file). Hungarian text is read as Hungarian."


def voices_dir() -> Path:
    return paths.models_dir() / "piper"


def custom_dir(base: Path | None = None) -> Path:
    """Voices of one's own (trained elsewhere or here): ``<id>.onnx`` + ``<id>.onnx.json``."""
    return (base or voices_dir()) / "custom"


def display_name(voice_id: str) -> str:
    """The name in the voice list, with the engine: "Imre (Piper)", "galambos (own Piper voice)"."""
    if voice_id in VOICES:
        return f"{VOICES[voice_id][1]} (Piper)"
    # "hu_HU-galambos-medium" -> "galambos (own Piper voice)"
    parts = voice_id.split("-")
    name = parts[1] if len(parts) >= 3 else voice_id
    return f"{name} (own Piper voice)"


def voice_note(voice_id: str) -> str:
    return VOICE_NOTES.get(voice_id, CUSTOM_NOTE)


def voice_paths(voice_id: str, base: Path | None = None) -> tuple[Path, Path]:
    folder = (base or voices_dir()) if voice_id in VOICES else custom_dir(base)
    return folder / f"{voice_id}.onnx", folder / f"{voice_id}.onnx.json"


def custom_voices(base: Path | None = None) -> list[str]:
    folder = custom_dir(base)
    if not folder.is_dir():
        return []
    found = (p.name[: -len(".onnx")] for p in folder.glob("*.onnx"))
    return sorted(v for v in found if v not in VOICES and voice_paths(v, base)[1].exists())


def installed_voices(base: Path | None = None) -> list[str]:
    """Ids of the voices whose model and config are both present: the downloaded Hungarian
    voices first, then one's own."""
    official = [v for v in VOICES if all(p.exists() for p in voice_paths(v, base))]
    return official + custom_voices(base)


class NotAPiperVoice(ValueError):
    pass


def add_voice(model: str | Path, base: Path | None = None) -> str:
    """Copy a Piper voice (``x.onnx`` with its ``x.onnx.json`` next to it) into the own-voices
    folder; returns its id."""
    model = Path(model)
    config = model.with_name(model.name + ".json")
    if model.suffix.lower() != ".onnx" or not model.is_file():
        raise NotAPiperVoice(f"{model.name} is not a Piper voice model (.onnx)")
    if not config.is_file():
        raise NotAPiperVoice(f"The voice's settings file is missing: {config.name}")
    try:
        cfg = json.loads(config.read_text(encoding="utf-8"))
        if "sample_rate" not in cfg["audio"] or "phoneme_id_map" not in cfg:
            raise KeyError("sample_rate or phoneme_id_map")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NotAPiperVoice(f"{config.name} is not a Piper voice settings file") from exc
    voice_id = model.name[: -len(".onnx")]
    if voice_id in VOICES:
        raise NotAPiperVoice(f"{voice_id} is the name of a downloaded voice; rename the file")
    target_model, target_config = voice_paths(voice_id, base)
    target_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, target_model)
    shutil.copyfile(config, target_config)
    return voice_id


def _url(folder: str, voice_id: str, suffix: str) -> str:
    name = "MODEL_CARD" if suffix == ".MODEL_CARD" else voice_id + suffix
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{folder}/{name}"


def _target(dest: Path, voice_id: str, suffix: str) -> Path:
    # the model card is saved as <voice>.MODEL_CARD.txt so the three cards do not collide
    return dest / (f"{voice_id}{suffix}.txt" if suffix == ".MODEL_CARD" else f"{voice_id}{suffix}")


def _fetch(
    url: str,
    target: Path,
    expected: int | None,
    progress: Callable[[str, int, int], None] | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TTSLTS"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(target.name, done, total)
    if expected is not None and tmp.stat().st_size != expected:
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise OSError(f"{target.name}: downloaded {size} bytes, expected {expected}")
    tmp.replace(target)


def download_voices(
    progress: Callable[[str, int, int], None] | None = None,
    dest: Path | None = None,
    voices: list[str] | None = None,
) -> list[str]:
    """Download (or reuse) the voices; returns the ids that are installed afterwards."""
    dest = dest or voices_dir()
    for voice_id in voices or list(VOICES):
        folder, _name, files = VOICES[voice_id]
        for suffix, size in files.items():
            target = dest / (
                f"{voice_id}{suffix}" if suffix != ".MODEL_CARD" else f"{voice_id}.MODEL_CARD.txt"
            )
            if not target.exists():
                _fetch(_url(folder, voice_id, suffix), target, size, progress)
    return installed_voices(dest)
