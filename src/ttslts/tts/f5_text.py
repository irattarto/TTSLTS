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
    return _LETTERS.sub("", word.lower())


def same_word(heard: str, expected: str, close: float = 0.75) -> bool:
    """Whether speech recognition heard ``expected`` (allowing a letter or two to differ)."""
    a, b = letters(heard), letters(expected)
    if not a or not b:
        return False
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= close


def first_word(text: str) -> str:
    for word in text.split():
        if letters(word):
            return word
    return ""


def pause_cut(audio, sr: int, text_chars: int, filler_chars: int = len(FILLER)) -> float | None:
    """Where the filler ends, from the sound alone: the pause after "nos," (its comma).

    F5 gives every character about the same time, so the filler should end near
    ``duration * filler_chars / text_chars``; the pause (40 ms or more, 30 dB under the speech)
    closest to that point is taken, and the cut goes 20 ms before the next word starts.
    Returns None when there is no such pause in the first 1.6 s.
    """
    import numpy as np

    frame = int(0.01 * sr)
    head = np.asarray(audio[: int(1.6 * sr)], dtype=np.float32)
    n = len(head) // frame
    if n < 10 or text_chars <= 0:
        return None
    level = 20 * np.log10(np.sqrt((head[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9)
    quiet = level < np.percentile(level, 95) - 30
    voiced = np.flatnonzero(~quiet)
    if not voiced.size:
        return None
    expected = len(audio) / sr * filler_chars / text_chars
    best = None
    i = voiced[0]
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        if j < n and j - i >= 4:  # a pause of 40 ms or more, with speech after it
            end = j * 0.01
            if best is None or abs(end - expected) < abs(best - expected):
                best = end
        i = j
    if best is None or not (0.4 * expected <= best <= 2.5 * expected):
        return None
    return max(0.0, best - 0.02)


def filler_cut(words: list[tuple[str, float, float]], expected_first: str) -> tuple[float, bool]:
    """Where to cut the filler off a reading.

    ``words`` are (word, start s, end s) from speech recognition. Returns (cut time in seconds,
    whether the reading starts well). The cut is just before the word that matches the text's
    first word, looked for among the first three. Recognition sometimes hears the filler and the
    first word as one ("nosvinetu"): then the cut goes where that word's last letters start. If
    nothing matches, the cut is between the first two words.
    """
    if not words:
        return 0.0, False
    expected = letters(expected_first)
    for i, (word, start, end) in enumerate(words[:3]):
        if same_word(word, expected_first):
            before = words[i - 1][2] if i else 0.0
            return max(before, start - 0.03), True
        heard = letters(word)
        if i:
            continue
        # the merged word's tail may be heard a letter or two shorter or longer than written
        # (names especially: "vinetu"), and at least two letters of filler are left in front
        for size in range(len(expected) + 1, max(1, len(expected) - 3), -1):
            if 2 <= len(heard) - size and same_word(heard[-size:], expected, close=0.65):
                share = (len(heard) - size) / len(heard)
                return start + (end - start) * share, True
    if len(words) >= 2:
        return (words[0][2] + words[1][1]) / 2, False
    return words[0][2], False
