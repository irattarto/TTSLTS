"""Choosing the clips an own F5 voice is trained on, from audiobook recordings.

How it was done by hand for the first two own F5 voices (work notes, 2026-10/11), in the order
the app does it:

* Whisper large-v3 cuts every chapter into sentences and writes them down; a sentence is kept
  when it is 1-15 s long and Whisper was sure of it (``usable_segment``). Whisper large-v3-turbo
  writes every kept sentence down once more: where the two disagree, the transcript is suspect
  (``agreement``; at 0.95 or more it is used, under 0.70 the sentence is left out).
* The narrator is the largest voice group (``narrator_core``): an audiobook narrator also does
  the characters' voices, which the voice should not learn. A speaker embedding (ECAPA) of every
  sentence is compared with the average of the group; kept are the sentences at least as alike
  as the least alike few percent of the group, and with a pitch near the narrator's.
* The most trembling quarter is left out (``wobble`` in the measuring step): the voice learnt
  the trembling of an old narrator's tired sentences.
* Each sentence is cut again by loudness (``cut_edges``): Whisper's times often started inside
  the end of the sentence before, and a voice learns to start with that scrap.

Everything that is measured is measured once, in the training runtime
(``ttslts.train.f5_prepare``); choosing (``select``) uses those numbers only, so the strictness
can be changed in the app at once. Only numpy is needed here.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

MIN_S, MAX_S = 1.0, 15.0  # sentence length kept for training
MIN_LOGPROB = -0.6  # Whisper's average log probability: under it, it was not sure
MAX_NO_SPEECH = 0.5
MAX_COMPRESSION = 2.4  # repeated text: Whisper got stuck
AGREE_OK, AGREE_CHECK = 0.95, 0.70
CORE_SHARE, CORE_ROUNDS = 0.6, 3

# strictness -> (likeness floor: percentile of the narrator group, pitch window in semitones,
# share of the most trembling sentences left out). "normal" is what the first voices used.
STRICTNESS = {
    "loose": (2.0, 5.0, 0.15),
    "normal": (5.0, 4.0, 0.25),
    "strict": (15.0, 3.0, 0.35),
}

_NOT_LETTER = re.compile(r"[^a-záéíóöőúüű0-9 ]")


def usable_segment(
    text: str, dur: float, logprob: float, no_speech: float, compression: float
) -> bool:
    """Whether a sentence Whisper cut is worth keeping: some text, 1-15 s, Whisper sure of it."""
    return (
        bool(text.strip())
        and MIN_S <= dur <= MAX_S
        and logprob >= MIN_LOGPROB
        and no_speech <= MAX_NO_SPEECH
        and compression <= MAX_COMPRESSION
    )


def normalized(text: str) -> str:
    """Lowercase letters and digits with single spaces: what two transcripts are compared on."""
    from ttslts.tts.hu_numbers import spell_numbers

    return " ".join(_NOT_LETTER.sub(" ", spell_numbers(text).lower()).split())


def agreement(first: str, second: str) -> float:
    """How alike two transcripts of one sentence are (0-1)."""
    a, b = normalized(first), normalized(second)
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)


def narrator_core(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The narrator as the largest voice group: from all sentences, keep the 60 % most alike to
    their average and average those again (three rounds). Returns (likeness of every sentence
    to the group's average, indices of the group). ``embeddings`` are unit vectors, one a row."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    core = np.arange(len(embeddings))
    sims = np.ones(len(embeddings))
    for _ in range(CORE_ROUNDS):
        centroid = embeddings[core].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = embeddings @ centroid
        core = np.argsort(-sims)[: max(1, int(CORE_SHARE * len(embeddings)))]
    return sims, np.sort(core)


def select(clips: list[dict], strictness: str = "normal") -> dict:
    """Marks every clip (dicts from the measuring step: ``sim``, ``core``, ``f0``, ``wobble``,
    ``agree``, ``dur``) with ``narrator``, ``steady``, ``transcript`` ("ok", "check", "drop") and
    ``keep``; returns a report (counts and minutes)."""
    floor_pct, semitones, wobble_drop = STRICTNESS[strictness]
    core = [c for c in clips if c.get("core")]
    pitched = [c["f0"] for c in core if c.get("f0")]
    if not core or not pitched:
        for c in clips:
            c.update(narrator=False, steady=False, transcript="drop", keep=False)
        return {"clips": len(clips), "kept": 0, "minutes": 0.0, "kept_minutes": 0.0}
    floor = float(np.percentile([c["sim"] for c in core], floor_pct))
    narrator_f0 = float(np.median(pitched))
    voice = []
    for c in clips:
        f0 = c.get("f0")
        c["narrator"] = bool(
            c["sim"] >= floor and f0 and abs(12 * np.log2(f0 / narrator_f0)) <= semitones
        )
        if c["narrator"]:
            voice.append(c)
    wobbles = [c["wobble"] for c in voice if c.get("wobble") is not None]
    cut = float(np.percentile(wobbles, 100 * (1 - wobble_drop))) if wobbles else float("inf")
    for c in clips:
        w = c.get("wobble")
        c["steady"] = bool(c["narrator"] and (w is None or w <= cut))
        agree = c.get("agree", 1.0)
        c["transcript"] = "ok" if agree >= AGREE_OK else "check" if agree >= AGREE_CHECK else "drop"
        c["keep"] = c["steady"] and c["transcript"] == "ok"

    def minutes(group):
        return round(sum(c["dur"] for c in group) / 60, 1)

    kept = [c for c in clips if c["keep"]]
    return {
        "clips": len(clips),
        "minutes": minutes(clips),
        "narrator": len(voice),
        "steady": sum(c["steady"] for c in clips),
        "kept": len(kept),
        "kept_minutes": minutes(kept),
        "suspect": sum(c["steady"] and c["transcript"] == "check" for c in clips),
        "narrator_f0": round(narrator_f0),
        "likeness_floor": round(floor, 3),
    }


QUIET_DB = 40.0  # under the chapter's loud level by this much counts as quiet
START_PAUSE, END_PAUSE = 6, 8  # frames of 10 ms
LOOK_BEFORE, LOOK_AFTER = 40, 15  # how far the start is looked for around Whisper's
END_MAX = 70  # how far after Whisper's end the pause is looked for


def quiet_frames(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per 10 ms frame of a whole chapter: whether it is quiet (40 dB under its loud level)."""
    frame = sr // 100
    n = len(audio) // frame
    level = 10 * np.log10((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    return level < np.percentile(level, 99) - QUIET_DB


def cut_edges(quiet: np.ndarray, start_s: float, end_s: float, next_start_s: float | None = None):
    """Where to cut a sentence Whisper placed at ``start_s``-``end_s``: start after the last
    pause (60 ms of quiet) from 0.4 s before to 0.15 s after Whisper's start, 30 ms in hand; end
    at the first pause of 80 ms after Whisper's end (at most 0.7 s on), never into the next
    sentence. Returns (start, end, soft): soft when no pause came (the narrator read on), then
    the end is Whisper's own and should be faded out."""
    n = len(quiet)
    s_frame, e_frame = int(start_s * 100), int(end_s * 100)
    nxt = int(next_start_s * 100) if next_start_s is not None else n
    start = None
    j, stop = min(n - 1, s_frame + LOOK_AFTER), max(0, s_frame - LOOK_BEFORE)
    while j > stop:
        if quiet[max(0, j - START_PAUSE) : j].all() and j >= START_PAUSE:
            start = j - 3
            break
        j -= 1
    if start is None:
        start = max(0, s_frame - 2)
    end, j = None, e_frame
    while j < min(n - END_PAUSE, e_frame + END_MAX, nxt):
        if quiet[j : j + END_PAUSE].all():
            end = j + END_PAUSE
            break
        j += 1
    soft = end is None
    if soft:
        end = min(e_frame + 3, nxt)
    return start / 100, end / 100, soft


def wobble(f0: np.ndarray) -> float | None:
    """How much the pitch trembles: the voiced pitch in cents, its distance from a 9-frame running
    median (RMS), averaged over the two halves of the sentence. ``None`` with too little voice."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0[f0 > 0]
    if len(voiced) < 20:
        return None
    cents = 1200 * np.log2(voiced / 100)
    halves = []
    for part in np.array_split(cents, 2):
        padded = np.pad(part, 4, mode="edge")
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 9), axis=1)
        halves.append(float(np.sqrt(np.mean((part - smooth) ** 2))))
    return float(np.mean(halves))


def median_pitch(f0: np.ndarray) -> float | None:
    voiced = np.asarray(f0)[np.asarray(f0) > 0]
    return float(np.median(voiced)) if len(voiced) >= 20 else None


def groups(clips: list[dict]) -> dict[str, list[dict]]:
    """After ``select``: the kept clips and the ones left out, by why."""
    return {
        "kept": [c for c in clips if c["keep"]],
        "other_voice": [c for c in clips if not c["narrator"]],
        "trembling": [c for c in clips if c["narrator"] and not c["steady"]],
        "unsure": [c for c in clips if c["steady"] and c["transcript"] != "ok"],
    }


SELECTION = "selection.json"


def save_selection(work, strictness: str, clips: list[dict], report: dict) -> None:
    """What the training step reads: the strictness chosen and the clips kept."""
    import json
    from pathlib import Path

    data = {
        "strictness": strictness,
        "report": report,
        "kept": [
            {"name": c["name"], "text": c["text"], "dur": c["dur"]} for c in clips if c["keep"]
        ],
    }
    Path(work, SELECTION).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_selection(work) -> dict | None:
    import json
    from pathlib import Path

    path = Path(work, SELECTION)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
