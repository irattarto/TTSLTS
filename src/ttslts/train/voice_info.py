"""What the app knows about a trained voice besides its weights: its natural pitch.

An RVC voice keeps the pitch of whatever it converts. A text-to-speech voice sits at a different
pitch than the person the voice was trained on, so the speech has to be shifted to the voice's
own range or it sounds unnatural. The median pitch of the training recordings is saved next to
the voice (``<name>.json``) for that.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ttslts import paths

log = logging.getLogger(__name__)


def median_f0(work: Path) -> dict | None:
    """Pitch statistics (Hz) of the voiced frames of the prepared clips in ``work``."""
    files = sorted((work / "f0nsf").glob("*.npy"))
    if not files:
        return None
    f0 = np.concatenate([np.load(f) for f in files])
    voiced = f0[f0 > 0]
    if len(voiced) < 50:
        return None
    return {
        "median_f0": round(float(np.median(voiced)), 1),
        "f0_p10": round(float(np.percentile(voiced, 10)), 1),
        "f0_p90": round(float(np.percentile(voiced, 90)), 1),
    }


def info_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".json")


def write(out_dir: Path, name: str, work: Path) -> dict | None:
    info = median_f0(work)
    if info is not None:
        (out_dir / f"{name}.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def load(model_path: str | Path) -> dict | None:
    """Pitch info for a voice file, or ``None`` if unknown (voices from elsewhere)."""
    path = info_path(model_path)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if float(info["median_f0"]) > 0:
            return info
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Voices trained before this was recorded: work it out from their training folder.
    model = Path(model_path)
    if model.parent.name == model.stem and model.parent.parent == paths.voices_dir():
        try:
            info = write(model.parent, model.stem, paths.train_dir() / model.stem)
        except OSError as exc:
            log.debug("Could not save pitch info: %s", exc)
            return None
        return info
    return None
