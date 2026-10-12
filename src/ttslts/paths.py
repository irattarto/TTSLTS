"""Per-user locations for settings and downloaded models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ttslts import APP_NAME


def config_dir() -> Path:
    override = os.environ.get("TTSLTS_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME


def settings_file() -> Path:
    return config_dir() / "settings.json"


def models_dir() -> Path:
    return config_dir() / "models"


def voices_dir() -> Path:
    """Where voices trained in the app are saved (one folder per voice)."""
    return config_dir() / "voices"


def train_dir() -> Path:
    """Scratch space for training runs (prepared audio, features, resumable checkpoints)."""
    return config_dir() / "train"
