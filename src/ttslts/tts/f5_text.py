"""Text and timing helpers for F5-TTS voices (no PyTorch, so the app and its tests can use them).

What was learnt making the galambos F5 voice (work notes, 2026-10-17):

* The Hungarian F5 model's vocabulary is lowercase only. A capital letter is not in it and turns
  into a space ("Később" was read as "tésőbb"), so the text is always lowercased first.
* F5 often starts a reading with a stray sound ("Ugalambos úr", "a reggelente"). A short filler
  word is read first and cut off afterwards, where speech recognition says the first real word
  starts (bad starts 65 % -> 32 % in the measurements); a reading that still starts wrong is
  tried again with another seed.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

FILLER = "nos, "  # read before the text, then cut off
CUT_AFTER_START_S = 0.03
# A reading's melody depends on its random seed: measured on 15 sentences x 6 seeds, about four
# in ten came out flat (pitch range under 6 semitones, against 9-10 for a lively one), which
# sounds tired. Such a reading is tried once more with another seed and the livelier one kept.
FLAT_SEMITONES = 6.0
SHORT_CHARS = 25  # a few words are flat anyway: not tried again for that
# characters that are not in the vocabulary but have a close one that is
_SAME = {"”": '"', "“": '"', "«": '"', "’": "'", "‘": "'", "—": "–", "(": ",", ")": ","}
_LETTERS = re.compile(r"[^a-záéíóöőúüű]")


def prepare_text(text: str, vocab: set[str]) -> str:
    """Lowercase ``text`` and keep only what the voice's vocabulary knows."""
    out = []
    for ch in unicodedata.normalize("NFC", text.lower()):
        ch = _SAME.get(ch, ch)
        if ch in vocab:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    return " ".join("".join(out).split())


def reference_text(text: str, vocab: set[str]) -> str:
    """The voice sample's text the way F5 wants it: prepared, ending in a full stop and a space."""
    text = prepare_text(text, vocab)
    if not text.endswith((".", "!", "?")):
        text += "."
    return text + " "


def letters(word: str) -> str:
    """The word's letters only; digits (Whisper writes "1848" for a spoken number) as words."""
    if any(ch.isdigit() for ch in word):
        from ttslts.tts.hu_numbers import spell_numbers

        word = spell_numbers(word)
    return _LETTERS.sub("", word.lower())


def same_word(heard: str, expected: str, close: float = 0.75) -> bool:
    """Whether speech recognition heard ``expected`` (allowing a letter or two to differ)."""
    a, b = letters(heard), letters(expected)
    if not a or not b:
        return False
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= close


def first_words(text: str, count: int = 2) -> list[str]:
    return [w for w in text.split() if letters(w)][:count]


def first_word(text: str) -> str:
    words = first_words(text, 1)
    return words[0] if words else ""


def filler_cut(words: list[tuple[str, float, float]], text: str) -> tuple[float, bool]:
    """Where to cut the filler off the reading of ``text``.

    ``words`` are (word, start s, end s) from speech recognition. Returns (cut time in seconds,
    whether the reading starts well). The cut is just after the start of the word that matches
    the text's first word, looked for among the first three; a short first word ("a", "az") only
    counts when the next word matches too, as another "a" comes soon. Recognition sometimes hears
    the filler and the first word as one ("nosvinetu"): then the cut goes where that word's last
    letters start. If nothing matches, the cut is between the first two words.
    """
    if not words:
        return 0.0, False
    wanted = first_words(text)
    if not wanted:
        return 0.0, False
    expected_first = wanted[0]
    expected = letters(expected_first)

    def starts_here(i: int) -> bool:
        if not same_word(words[i][0], expected_first):
            return False
        if len(expected) > 2 or len(wanted) < 2:
            return True
        return i + 1 < len(words) and same_word(words[i + 1][0], wanted[1])

    for i, (word, start, end) in enumerate(words[:3]):
        if starts_here(i):
            before = words[i - 1][2] if i else 0.0
            # 30 ms into it: Whisper puts a word's start early, and cutting right there left the
            # filler's "s" (heard as "és"); measured best of 0-200 ms, 120 ms cuts into words
            return max(before, start + CUT_AFTER_START_S), True
        heard = letters(word)
        if i:
            continue
        # the merged word's tail may be heard a letter or two shorter or longer than written
        # (names especially: "vinetu"), and at least two letters of filler are left in front
        for size in range(len(expected) + 1, max(1, len(expected) - 3), -1):
            if 2 <= len(heard) - size and same_word(heard[-size:], expected, close=0.65):
                share = (len(heard) - size) / len(heard)
                return start + (end - start) * share, True
    # Whisper joins and splits words its own way ("anap" for "a nap", "oldsetterhend"): compare
    # the letters of the text's first two words with what is heard from the second word on
    target = "".join(letters(w) for w in wanted)
    for i in (1, 2):
        heard = "".join(letters(w[0]) for w in words[i : i + 3])[: len(target) + 2]
        if heard and difflib.SequenceMatcher(None, heard[: len(target)], target).ratio() >= 0.7:
            return max(words[i - 1][2], words[i][1] + CUT_AFTER_START_S), True
    if len(words) >= 2:
        return (words[0][2] + words[1][1]) / 2, False
    return words[0][2], False


def melody_range(audio, sr: int) -> float:
    """How lively a reading sounds: its pitch range (10th to 90th percentile) in semitones."""
    import numpy as np

    from ttslts.audio.dsp import estimate_f0, resample

    f0 = estimate_f0(
        resample(np.asarray(audio, dtype=np.float32), sr, 16000), 16000, fmin=50, fmax=300
    )
    voiced = f0[f0 > 0]
    if len(voiced) < 20:
        return 0.0
    return float(12 * np.log2(np.percentile(voiced, 90) / np.percentile(voiced, 10)))


def is_lively(melody: float, text: str) -> bool:
    return melody >= FLAT_SEMITONES or len(text) < SHORT_CHARS
