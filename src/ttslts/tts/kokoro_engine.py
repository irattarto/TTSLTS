"""Kokoro-82M text-to-speech via kokoro-onnx."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

LANGUAGES = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "es": "Spanish",
    "fr-fr": "French",
    "hi": "Hindi",
    # Not a Kokoro language: no voice was trained on it. espeak-ng pronounces the text
    # and an existing voice reads it, so expect a foreign accent.
    "hu": "Hungarian (experimental, existing voices)",
    "it": "Italian",
    "ja": "Japanese",
    "pt-br": "Portuguese (BR)",
    "cmn": "Mandarin Chinese",
}


class KokoroEngine:
    def __init__(self, kokoro):
        self._kokoro = kokoro

    @classmethod
    def load(cls, model_path: str, voices_path: str, use_gpu: bool = True) -> KokoroEngine:
        if not model_path or not voices_path:
            raise ValueError("Kokoro model files are not set (Settings > Models > Download)")
        from kokoro_onnx import Kokoro

        from ttslts.onnx_util import make_session

        return cls(Kokoro.from_session(make_session(model_path, use_gpu), voices_path))

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def synthesize(
        self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us"
    ) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), 24000
        if lang == "hu":
            from ttslts.tts.hu_text import normalize_hungarian

            text = normalize_hungarian(text)
        audio, sr = self._kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return np.asarray(audio, dtype=np.float32), int(sr)
