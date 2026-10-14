"""Offline text to speech with Piper voices (VITS models exported to ONNX).

The ONNX files and their ``.onnx.json`` configs come from the Piper voice collection. Text is
turned into IPA phonemes with espeak-ng (the espeakng-loader package), mapped to ids with the
voice's ``phoneme_id_map`` and run through ONNX Runtime. This follows Piper's own reference
implementation: ids are ``BOS, (phoneme, PAD)*, EOS`` and the model takes ``input``,
``input_lengths``, ``scales`` (noise, length, noise_w) and, for multi-speaker voices, ``sid``.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

BOS, EOS, PAD = "^", "$", "_"
SENTENCE_PAUSE_S = 0.25  # after a sentence that ends in a full stop
# Longer pauses where a speaker pauses longer: after a question or an exclamation, after "…",
# at a line break and most at a new paragraph (an empty line).
PAUSE_AFTER = {"?": 0.35, "!": 0.35, "…": 0.5}
LINE_PAUSE_S = 0.4
PARAGRAPH_PAUSE_S = 0.7
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_espeak_ready = False
_espeak_lock = threading.Lock()


def _ensure_espeak() -> None:
    """Point phonemizer at the espeak-ng library bundled with espeakng-loader (once)."""
    global _espeak_ready
    if _espeak_ready:
        return
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    _espeak_ready = True


def split_sentences(text: str) -> list[str]:
    return [s for s, _ in split_with_pauses(text)]


def split_with_pauses(text: str) -> list[tuple[str, float]]:
    """Sentences of ``text``, each with the pause (seconds) that follows it."""
    text = text.strip().replace("...", "…")
    out: list[tuple[str, float]] = []
    start = 0
    for match in [*_SENTENCE_END.finditer(text), None]:
        end = match.start() if match else len(text)
        sentence = text[start:end].strip()
        if sentence:
            gap = match.group() if match else ""
            pause = PAUSE_AFTER.get(sentence[-1], SENTENCE_PAUSE_S)
            if gap.count("\n") >= 2:
                pause = max(pause, PARAGRAPH_PAUSE_S)
            elif "\n" in gap:
                pause = max(pause, LINE_PAUSE_S)
            out.append((sentence, pause))
        if match:
            start = match.end()
    return out


class PiperVoice:
    def __init__(self, session, config: dict):
        self.session = session
        self.config = config
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.num_speakers = int(config.get("num_speakers", 1))
        inference = config.get("inference", {})
        self.noise_scale = float(inference.get("noise_scale", 0.667))
        self.length_scale = float(inference.get("length_scale", 1.0))
        self.noise_w = float(inference.get("noise_w", 0.8))
        self.espeak_voice = config.get("espeak", {}).get("voice", "en-us")
        if config.get("phoneme_type", "espeak") != "espeak":
            raise ValueError("Only espeak-phonemised Piper voices are supported")
        self.id_map: dict[str, list[int]] = config["phoneme_id_map"]
        self._input_names = {i.name for i in session.get_inputs()}

    @classmethod
    def load(cls, model_path: str, config_path: str = "", use_gpu: bool = True) -> PiperVoice:
        from ttslts.onnx_util import make_session

        config_path = config_path or f"{model_path}.json"
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(make_session(model_path, use_gpu), config)

    # -- text -> ids -----------------------------------------------------------------------
    def phonemize(self, sentence: str) -> list[str]:
        import phonemizer

        _ensure_espeak()
        with _espeak_lock:
            ipa = phonemizer.phonemize(
                sentence, self.espeak_voice, preserve_punctuation=True, with_stress=True
            )
        return list(unicodedata.normalize("NFD", ipa.strip()))

    def phoneme_ids(self, phonemes: list[str]) -> list[int]:
        ids = list(self.id_map[BOS])
        for p in phonemes:
            if p not in self.id_map:
                continue  # sounds the voice was never trained on are skipped, as Piper does
            ids.extend(self.id_map[p])
            ids.extend(self.id_map[PAD])
        ids.extend(self.id_map[EOS])
        return ids

    def missing_phonemes(self, phonemes: list[str]) -> set[str]:
        return {p for p in phonemes if p not in self.id_map}

    # -- ids -> audio ----------------------------------------------------------------------
    def _run(
        self,
        ids: list[int],
        speed: float,
        speaker_id: int,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> np.ndarray:
        noise_scale = self.noise_scale if noise_scale is None else noise_scale
        noise_w = self.noise_w if noise_w is None else noise_w
        feed = {
            "input": np.asarray([ids], dtype=np.int64),
            "input_lengths": np.asarray([len(ids)], dtype=np.int64),
            "scales": np.asarray(
                [noise_scale, self.length_scale / max(speed, 0.1), noise_w], dtype=np.float32
            ),
        }
        if "sid" in self._input_names:
            feed["sid"] = np.asarray([speaker_id], dtype=np.int64)
        audio = self.session.run(None, feed)[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: int = 0,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Speak ``text``. ``noise_scale`` varies the voice, ``noise_w`` the rhythm."""
        if self.espeak_voice.startswith("hu"):
            from ttslts.tts.hu_text import normalize_hungarian

            text = normalize_hungarian(text)
        parts: list[np.ndarray] = []
        pause = 0.0
        for sentence, pause_after in split_with_pauses(text):
            phonemes = self.phonemize(sentence)
            if not phonemes:
                continue
            if parts:
                parts.append(np.zeros(int(pause * self.sample_rate), dtype=np.float32))
            ids = self.phoneme_ids(phonemes)
            parts.append(self._run(ids, speed, speaker_id, noise_scale, noise_w))
            pause = pause_after
        if not parts:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        audio = np.concatenate(parts)
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio * (0.9 / peak)
        return audio.astype(np.float32), self.sample_rate
