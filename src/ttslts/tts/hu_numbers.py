"""Hungarian numbers written out as words, for voices that cannot read digits.

Piper gets its numbers read by espeak-ng; an F5 voice reads the text's characters itself and was
trained on almost no digits, so it reads them badly. ``spell_numbers`` turns them into words the
way they are said: cardinals ("1848" -> "ezernyolcszáznegyvennyolc", "2 alma" -> "két alma"),
ordinals ("21. század" -> "huszonegyedik század"), dates ("1848. március 15." -> "... március
tizenötödike", "2026.10.17."), times ("10:30" -> "tíz óra harminc"), decimals ("3,5" -> "három
egész öt tized"), ranges ("2-3"), percentages, money, signs and number + suffix ("5-ös" ->
"ötös", "2026-ban").
"""

from __future__ import annotations

import re

_ONES = ["", "egy", "kettő", "három", "négy", "öt", "hat", "hét", "nyolc", "kilenc"]
_TENS = ["", "tíz", "húsz", "harminc", "negyven", "ötven", "hatvan", "hetven", "nyolcvan",
         "kilencven"]  # fmt: skip
_TENS_JOIN = ["", "tizen", "huszon", "harminc", "negyven", "ötven", "hatvan", "hetven",
              "nyolcvan", "kilencven"]  # fmt: skip
_GROUPS = [(10**9, "milliárd"), (10**6, "millió"), (1000, "ezer")]
_MONTHS = ("január", "február", "március", "április", "május", "június", "július", "augusztus",
           "szeptember", "október", "november", "december")  # fmt: skip
# the last word of a cardinal -> the ordinal ending
_ORDINAL_END = {
    "egy": "egyedik", "kettő": "kettedik", "három": "harmadik", "négy": "negyedik",
    "öt": "ötödik", "hat": "hatodik", "hét": "hetedik", "nyolc": "nyolcadik",
    "kilenc": "kilencedik", "tíz": "tizedik", "húsz": "huszadik", "harminc": "harmincadik",
    "negyven": "negyvenedik", "ötven": "ötvenedik", "hatvan": "hatvanadik",
    "hetven": "hetvenedik", "nyolcvan": "nyolcvanadik", "kilencven": "kilencvenedik",
    "száz": "századik", "ezer": "ezredik", "millió": "milliomodik", "milliárd": "milliárdodik",
}  # fmt: skip
_CURRENCY = {"Ft": "forint", "HUF": "forint", "€": "euró", "EUR": "euró", "$": "dollár",
             "USD": "dollár", "£": "font"}  # fmt: skip
# after these a "2" stays "kettő" ("2 és 3"); before other words it is "két" ("2 alma")
_NOT_NOUNS = {"és", "vagy", "meg", "is", "sem", "de", "illetve", "mint", "a", "az", "egy"}
_VOWELS = set("aáeéiíoóöőuúüű")
_BACK = set("aáoóuú")
# a suffix with a linking vowel shortens these ("10-et" -> "tizet", "3-as" -> "hármas"); "-ig",
# "-ért" do not ("tízig", "háromért")
_STEM_BEFORE_VOWEL = {"tíz": "tiz", "húsz": "husz", "hét": "het", "három": "hárm"}
_DAY_SUFFIXES = {"án", "én", "áig", "éig", "ától", "étől"}  # "7-én" -> "hetedikén"


def _two(digit: int, short: bool) -> str:
    return "két" if digit == 2 and short else _ONES[digit]


def _below_thousand(n: int, short_two: bool) -> str:
    hundreds, rest = divmod(n, 100)
    tens, ones = divmod(rest, 10)
    out = ""
    if hundreds:
        out += ("" if hundreds == 1 else _two(hundreds, True)) + "száz"
    if tens and ones:
        out += _TENS_JOIN[tens] + _two(ones, short_two)
    elif tens:
        out += _TENS[tens]
    elif ones:
        out += _two(ones, short_two)
    return out


def cardinal(n: int, short_two: bool = False) -> str:
    """``n`` in words. ``short_two``: "két" instead of "kettő" at the end (before a noun)."""
    if n == 0:
        return "nulla"
    if n < 0:
        return "mínusz " + cardinal(-n, short_two)
    parts = []
    rest = n
    for size, name in _GROUPS:
        count, rest = divmod(rest, size)
        if count:
            parts.append(("" if count == 1 and size == 1000 else cardinal(count, True)) + name)
    if rest:
        parts.append(_below_thousand(rest, short_two))
    # the spelling rule: above 2000 the groups are joined with hyphens ("kétezer-egy")
    return ("-" if n > 2000 else "").join(parts)


def ordinal(n: int) -> str:
    if n == 1:
        return "első"
    if n == 2:
        return "második"
    words = cardinal(n)
    for end in sorted(_ORDINAL_END, key=len, reverse=True):
        if words.endswith(end):
            return words[: -len(end)] + _ORDINAL_END[end]
    return words + "adik"


def day(n: int, suffix: str = "") -> str:
    """Day of the month: "15." -> "tizenötödike", "15-én" -> "tizenötödikén", "1." -> "elseje"."""
    if n == 1:
        stem, back = "elsej", False
    else:
        base = ordinal(n)[:-2]  # "tizenötöd"
        stem = base + "ik"
        back = [ch for ch in base if ch in _VOWELS][-1] in _BACK
    short, long_ = ("a", "á") if back else ("e", "é")
    s = suffix.lower()
    if s in ("", "a", "e"):
        end = short
    elif s in ("án", "én", "n"):
        end = long_ + "n"
    elif s in ("áig", "éig", "ig"):
        end = long_ + "ig"
    elif s in ("ától", "étől", "tól", "től"):
        end = long_ + ("tól" if back else "től")
    else:
        end = short + s
    return stem + end


def _decimal(whole: str, frac: str) -> str:
    out = cardinal(int(whole)) + " egész "
    names = {1: "tized", 2: "század", 3: "ezred"}
    if len(frac) in names:
        return out + cardinal(int(frac)) + " " + names[len(frac)]
    return out + " ".join(cardinal(int(d)) for d in frac)


_M = "|".join(_MONTHS)
_DATE = re.compile(rf"\b(\d{{4}})\.\s+({_M})\s+(\d{{1,2}})\.?(?:-(\w+))?", re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"\b(\d{4})\.\s?(\d{1,2})\.\s?(\d{1,2})\.?(?:-(\w+))?")
_YEAR_MONTH = re.compile(rf"\b(\d{{4}})\.(?=\s+(?:{_M})\b)", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b({_M})\s+(\d{{1,2}})\.?(?:-(\w+))?", re.IGNORECASE)
_SUFFIX = r"(?:-([a-záéíóöőúüű]+))?"  # "-kor", "-kal", "-ot": glued on after the words
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b" + _SUFFIX)
_THOUSANDS = re.compile(r"(?<![\d,.])\d{1,3}(?:[  ]\d{3})+(?![\d,])")
_CURRENCY_FIRST = re.compile(r"([€$£])\s?(\d+)")
_CURRENCY_AFTER = re.compile(
    r"(\d)\s?(" + "|".join(re.escape(c) for c in _CURRENCY) + r")(?![\wÀ-ÿ])" + _SUFFIX
)
_PERCENT = re.compile(r"(\d)\s?%" + _SUFFIX)
_DECIMAL = re.compile(r"(?<![\d,])(\d+),(\d+)(?![\d,])")
_RANGE = re.compile(r"(?<![\d,.])(\d+)\s?[-–]\s?(\d+)(?![\d,])")
_ORDINAL = re.compile(r"(?<![\d,.])(\d{1,4})\.(?=\s+[a-záéíóöőúüű])")
_SIGNED = re.compile(r"(?<![\w\d])[-−](\d)")
_SUFFIXED = re.compile(r"(?<![\d,.])(\d+)-([a-záéíóöőúüű]+)")
_NUMBER = re.compile(r"(?<![\d,])(\d+)(?![\d,])")


def spell_numbers(text: str) -> str:
    def date(year, month, d, suffix):
        return f"{cardinal(int(year))} {month} {day(int(d), suffix or '')}"

    def number(m):
        digits = m.group(1)
        if len(digits) > 12 or (len(digits) > 1 and digits.startswith("0")):
            return " ".join(cardinal(int(d)) for d in digits)  # a phone number, a code
        nxt = re.match(r"\s([a-záéíóöőúüű]+)", m.string[m.end() :])
        return cardinal(int(digits), short_two=bool(nxt) and nxt.group(1) not in _NOT_NOUNS)

    def suffixed(m):
        n, suffix = int(m.group(1)), m.group(2)
        if suffix.lower() in _DAY_SUFFIXES and 1 <= n <= 31:
            return day(n, suffix)
        words = cardinal(n)
        if suffix[0] in "aeoö":
            for end, stem in _STEM_BEFORE_VOWEL.items():
                if words.endswith(end):
                    words = words[: -len(end)] + stem
                    break
        return words + suffix

    def numeric_date(m):
        month = int(m.group(2))
        if not 1 <= month <= 12:
            return m.group()
        return date(m.group(1), _MONTHS[month - 1], m.group(3), m.group(4))

    text = _DATE.sub(lambda m: date(*m.groups()), text)
    text = _NUMERIC_DATE.sub(numeric_date, text)
    text = _YEAR_MONTH.sub(lambda m: cardinal(int(m.group(1))), text)
    text = _MONTH_DAY.sub(lambda m: f"{m.group(1)} {day(int(m.group(2)), m.group(3) or '')}", text)
    text = _TIME.sub(
        lambda m: (
            f"{cardinal(int(m.group(1)), True)} óra"
            + (f" {cardinal(int(m.group(2)))}" if int(m.group(2)) else "")
            + (m.group(3) or "")  # "10:30-kor" -> "... harminckor", "10:00-kor" -> "órakor"
        ),
        text,
    )
    text = _THOUSANDS.sub(lambda m: re.sub(r"\D", "", m.group()), text)
    text = _CURRENCY_FIRST.sub(lambda m: f"{m.group(2)} {_CURRENCY[m.group(1)]}", text)
    text = _CURRENCY_AFTER.sub(
        lambda m: f"{m.group(1)} {_CURRENCY[m.group(2)]}{m.group(3) or ''}", text
    )
    text = _PERCENT.sub(lambda m: f"{m.group(1)} százalék{m.group(2) or ''}", text)
    text = _DECIMAL.sub(lambda m: _decimal(m.group(1), m.group(2)), text)
    text = _RANGE.sub(
        lambda m: f"{cardinal(int(m.group(1)), True)}-{m.group(2)}", text
    )  # "2-3 nap" -> "két-három nap" (the second number is spelt below)
    text = _ORDINAL.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _SIGNED.sub(lambda m: "mínusz " + m.group(1), text)
    text = _SUFFIXED.sub(suffixed, text)
    text = _NUMBER.sub(number, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
