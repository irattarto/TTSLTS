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
