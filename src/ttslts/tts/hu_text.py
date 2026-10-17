"""Hungarian text clean-up before phonemisation.

espeak-ng already reads numbers, dates and percentages in Hungarian, but it spells common
abbreviations letter by letter ("kb." as "ká-bé"). This expands them to the words people mean.
"""

from __future__ import annotations

import re

# Whole-word abbreviations that end in a period (or are written in full as shown).
_ABBREVIATIONS = {
    "pl.": "például",
    "stb.": "és így tovább",
    "kb.": "körülbelül",
    "dr.": "doktor",
    "ill.": "illetve",
    "ún.": "úgynevezett",
    "ker.": "kerület",
    "kft.": "korlátolt felelősségű társaság",
    "zrt.": "zártkörűen működő részvénytársaság",
    "nyrt.": "nyilvánosan működő részvénytársaság",
    "utca": "utca",
    "u.": "utca",
    "ny.": "nyugdíjas",
    "tel.": "telefon",
    "sz.": "szám",
    "old.": "oldal",
    "vö.": "vesd össze",
    "pld.": "például",
    "ált.": "általában",
    "db": "darab",
}
# Units only when they follow a number: "3 kg", "12km".
_UNITS = {
    "kg": "kilogramm",
    "dkg": "dekagramm",
    "g": "gramm",
    "mg": "milligramm",
    "km": "kilométer",
    "m": "méter",
    "cm": "centiméter",
    "mm": "milliméter",
    "l": "liter",
    "dl": "deciliter",
    "cl": "centiliter",
    "ml": "milliliter",
    "db": "darab",
    "perc": "perc",
    "°C": "Celsius-fok",
    "°": "fok",
}
# Abbreviations that usually close a sentence: the period they carry is kept.
_SENTENCE_FINAL = {"stb."}
_ORDINALS = {
    "1": "első",
    "2": "második",
    "3": "harmadik",
    "4": "negyedik",
    "5": "ötödik",
    "6": "hatodik",
    "7": "hetedik",
    "8": "nyolcadik",
    "9": "kilencedik",
    "10": "tizedik",
}

_ABBREV_RE = re.compile(
    r"(?<![\wÀ-ÿ])("
    + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True))
    + r")(?![\wÀ-ÿ])",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"(?<=\d)\s?("
    + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
    + r")(?![\wÀ-ÿ])"
)
_ORDINAL_RE = re.compile(r"(?<![\d,.])(10|[1-9])\.\s+(?=[a-záéíóöőúüű])")


# Hungarian letter names, for abbreviations said letter by letter ("SNS" -> "es-en-es")
_LETTER_NAMES = {
    "A": "a", "Á": "á", "B": "bé", "C": "cé", "D": "dé", "E": "e", "É": "é", "F": "ef",
    "G": "gé", "H": "há", "I": "i", "Í": "í", "J": "jé", "K": "ká", "L": "el", "M": "em",
    "N": "en", "O": "o", "Ó": "ó", "Ö": "ö", "Ő": "ő", "P": "pé", "Q": "kú", "R": "er",
    "S": "es", "T": "té", "U": "u", "Ú": "ú", "Ü": "ü", "Ű": "ű", "V": "vé", "W": "dupla vé",
    "X": "iksz", "Y": "ipszilon", "Z": "zé",
}  # fmt: skip
_CAPS = "A-ZÁÉÍÓÖŐÚÜŰ"
_ACRONYM_RE = re.compile(rf"(?<![\w-])([{_CAPS}]{{2,6}})(?=$|[^\w]|-)")
_HU_VOWELS = set("AÁEÉIÍOÓÖŐUÚÜŰ")
_DIGRAPHS = ("DZS", "CS", "DZ", "GY", "LY", "NY", "SZ", "TY", "ZS")  # one sound each
# short capitalised words that are words, not abbreviations
_CAPS_WORDS = {"AZ", "EZ", "ÉS", "IS", "HA", "DE", "NE", "MA", "MI", "TE", "ŐK", "ÉN", "IGEN"}


def _sounds(word: str) -> list[str]:
    out, i = [], 0
    while i < len(word):
        for d in _DIGRAPHS:
            if word.startswith(d, i):
                out.append(d)
                i += len(d)
                break
        else:
            out.append(word[i])
            i += 1
    return out


def _said_as_word(word: str) -> bool:
    """NATO, MÁV, KRESZ, ELTE are said as words; SNS, BKV, OTP, EU letter by letter."""
    if len(word) <= 2:
        return False
    sounds = _sounds(word)
    run = 0
    for s in sounds:
        run = 0 if s[0] in _HU_VOWELS else run + 1
        if run > 2:
            return False
    ends = sounds[-2:]
    return any(s[0] in _HU_VOWELS for s in sounds) and not all(s[0] not in _HU_VOWELS for s in ends)


def spell_acronyms(text: str) -> str:
    """Abbreviations in capitals said the Hungarian way, for voices that read every word as a
    word (F5): "SNS" -> "es-en-es", "BKV-val" -> "bé-ká-vé-val"; NATO and MÁV stay words.

    Run after the pronunciation dictionary, whose built-in list says FBI, USB etc. with
    English letter names."""

    def spell(match: re.Match) -> str:
        word = match.group(1)
        if word in _CAPS_WORDS or _said_as_word(word):
            return word
        return "-".join(_LETTER_NAMES[ch] for ch in word)

    return _ACRONYM_RE.sub(spell, text)


def normalize_hungarian(text: str) -> str:
    def abbreviation(match: re.Match) -> str:
        word = match.group(1)
        replacement = _ABBREVIATIONS[word.lower()]
        if word.lower() in _SENTENCE_FINAL and re.match(
            r"\s*(?:$|[A-ZÁÉÍÓÖŐÚÜŰ])", match.string[match.end() :]
        ):
            replacement += "."
        # keep a leading capital: "Dr. Kovács" -> "Doktor Kovács"
        return replacement.capitalize() if word[:1].isupper() else replacement

    text = _UNIT_RE.sub(lambda m: " " + _UNITS[m.group(1)], text)
    text = _ORDINAL_RE.sub(lambda m: _ORDINALS[m.group(1)] + " ", text)
    text = _ABBREV_RE.sub(abbreviation, text)
    text = text.replace("&", " és ")
    return re.sub(r"[ \t]{2,}", " ", text).strip()
