"""User settings, persisted as JSON in the per-user config directory."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ttslts import paths

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # Window behaviour
    minimize_to_tray: bool = True
    close_to_tray: bool = True
    start_minimized: bool = False

    # Audio devices, stored by name ("" means the system default)
    input_device: str = ""
    output_device: str = ""
    tts_output_device: str = ""

    # Text-to-speech (Kokoro)
    kokoro_model_path: str = ""
    kokoro_voices_path: str = ""
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_lang: str = "en-us"
    tts_through_rvc: bool = False
    tts_engine: str = "kokoro"  # "kokoro" or "piper"
    piper_voice: str = ""
    piper_variation: float = 0.667  # voice variation (Piper noise_scale)
    piper_rhythm: float = 0.8  # rhythm variation (Piper noise_w)

    # Voice changer (RVC v2)
    rvc_model_path: str = ""
    contentvec_path: str = ""
    rmvpe_path: str = ""
    rvc_sample_rate: int = 40000
    pitch_shift: int = 0
    speaker_id: int = 0
    chunk_ms: int = 200
    context_ms: int = 600
    crossfade_ms: int = 40
    silence_threshold_db: float = -50.0

    # Runtime
    use_gpu: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or paths.settings_file()
        settings = cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return settings
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable settings file %s: %s", path, exc)
            return settings
        if not isinstance(raw, dict):
            return settings
        for f in fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            default = getattr(settings, f.name)
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, float):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if ok else value
            elif isinstance(default, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
            else:
                ok = isinstance(value, str)
            if ok:
                setattr(settings, f.name, value)
            else:
                log.warning("Ignoring invalid value for %s: %r", f.name, value)
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or paths.settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)
