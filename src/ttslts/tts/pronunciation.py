"""Pronunciation dictionary: respell words so the voice says them the way you want.

Both speech engines read the text with the spelling rules of their language, so a word they get
wrong is fixed by writing it the way it sounds in that language, for example ``Windows`` ->
``vindóz`` in Hungarian text. Entries are applied to the text before it is spoken.

A small built-in list covers abbreviations that Hungarians say with English letter names (FBI, USB).
It is only used for Hungarian speech, only for the exact capital-letter spelling, and an entry of
your own for the same word wins over it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ttslts import paths

log = logging.getLogger(__name__)

# Hungarian spelling of the English letter names: how these are said in Hungarian speech.
BUILTIN_HU: dict[str, str] = {
    "FBI": "ef bí áj",
    "CIA": "szí áj éj",
    "USA": "jú esz éj",
    "UK": "jú kéj",
    "DVD": "dí ví dí",
    "CD": "szí dí",
    "USB": "jú esz bí",
    "GPS": "dzsí pí esz",
    "PC": "pí szí",
    "AI": "éj áj",
    "IT": "áj tí",
    "VIP": "ví áj pí",
}


@dataclass
class Entry:
    word: str
    say: str
    match_case: bool = False  # off: "nvidia", "Nvidia" and "NVIDIA" are all replaced


def dictionary_file() -> Path:
    return paths.config_dir() / "pronunciations.json"


def load(path: Path | None = None) -> list[Entry]:
    path = path or dictionary_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable pronunciation file %s: %s", path, exc)
        return []
    entries = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("word"), str):
            entries.append(
                Entry(
                    item["word"].strip(),
                    str(item.get("say", "")),
                    bool(item.get("match_case", False)),
                )
            )
    return [e for e in entries if e.word]


def save(entries: list[Entry], path: Path | None = None) -> None:
    path = path or dictionary_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [asdict(e) for e in entries if e.word.strip() and e.say.strip()]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def apply(text: str, entries: list[Entry], builtin: dict[str, str] | None = None) -> str:
    """Replace every whole word (or the part before a hyphen: ``Windows-ban``) in ``text``."""
    table: dict[str, list[Entry]] = {}
    for e in [*(Entry(w, s, True) for w, s in (builtin or {}).items()), *entries]:
        if e.word.strip() and e.say.strip():
            table.setdefault(e.word.casefold(), []).insert(0, e)  # later entries win
    if not table:
        return text
    words = sorted(table, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.I)

    def replace(match: re.Match) -> str:
        found = match.group(0)
        for e in table[found.casefold()]:
            if not e.match_case or found == e.word:
                return e.say
        return found

    return pattern.sub(replace, text)


def apply_for(text: str, language: str, use_builtin: bool = True) -> str:
    """Your entries, and for Hungarian speech also the built-in list, applied to ``text``."""
    builtin = BUILTIN_HU if use_builtin and language.lower().startswith("hu") else None
    return apply(text, load(), builtin)
