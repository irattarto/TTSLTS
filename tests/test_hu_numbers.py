import pytest

from ttslts.tts.hu_numbers import cardinal, day, ordinal, spell_numbers


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "nulla"),
        (2, "kettő"),
        (11, "tizenegy"),
        (20, "húsz"),
        (22, "huszonkettő"),
        (100, "száz"),
        (215, "kétszáztizenöt"),
        (1000, "ezer"),
        (1848, "ezernyolcszáznegyvennyolc"),
        (2000, "kétezer"),
        (2026, "kétezer-huszonhat"),  # hyphens above 2000
        (22000, "huszonkétezer"),
        (1_000_000, "egymillió"),
        (-5, "mínusz öt"),
    ],
)
def test_cardinals(n, words):
    assert cardinal(n) == words


def test_two_before_a_noun_is_ket():
    assert cardinal(2, short_two=True) == "két"
    assert cardinal(32, short_two=True) == "harminckét"


@pytest.mark.parametrize(
    ("n", "words"),
    [(1, "első"), (2, "második"), (3, "harmadik"), (11, "tizenegyedik"), (12, "tizenkettedik"),
     (20, "huszadik"), (21, "huszonegyedik"), (100, "századik"), (1000, "ezredik")],
)  # fmt: skip
def test_ordinals(n, words):
    assert ordinal(n) == words


def test_days_of_the_month():
    assert day(1) == "elseje" and day(1, "én") == "elsején"
    assert day(2) == "másodika" and day(15) == "tizenötödike"
    assert day(15, "én") == "tizenötödikén" and day(20, "áig") == "huszadikáig"


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("1848. március 15-én kitört.", "ezernyolcszáznegyvennyolc március tizenötödikén kitört."),
        ("Ma 2026.10.17. van.", "Ma kétezer-huszonhat október tizenhetedike van."),
        ("10:30-kor és 10:00-kor", "tíz óra harminckor és tíz órakor"),
        ("Kaptam 2 almát és 3 körtét.", "Kaptam két almát és három körtét."),
        ("2 és 3", "kettő és három"),
        ("A 21. században", "A huszonegyedik században"),
        ("1 000 000 ember", "egymillió ember"),
        ("1 499 Ft-ot fizettem", "ezernégyszázkilencvenkilenc forintot fizettem"),
        ("3,5%-kal több", "három egész öt tized százalékkal több"),
        ("-5 fok, 2-3 nap", "mínusz öt fok, két-három nap"),
        ("Hívj: 0630123!", "Hívj: nulla hat három nulla egy kettő három!"),
        ("2001-ben, 5-öt", "kétezer-egyben, ötöt"),
        ("Május 1. ünnep.", "Május elseje ünnep."),
        ("€20", "húsz euró"),
        ("Ez 5.", "Ez öt."),
    ],
)
def test_numbers_in_text(text, said):
    assert spell_numbers(text) == said


def test_text_without_numbers_is_unchanged():
    assert (
        spell_numbers("Szép szőke szirén, a sás susogott.") == "Szép szőke szirén, a sás susogott."
    )
