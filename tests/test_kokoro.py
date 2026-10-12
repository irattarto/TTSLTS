import numpy as np

from ttslts.tts.kokoro_engine import KokoroEngine


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def get_voices(self):
        return ["bf_emma", "af_heart"]

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return np.ones(2400, dtype=np.float32), 24000


def test_synthesize_passes_options():
    k = FakeKokoro()
    audio, sr = KokoroEngine(k).synthesize(" hi ", "af_heart", 1.2, "en-gb")
    assert sr == 24000 and len(audio) == 2400
    assert k.calls == [("hi", "af_heart", 1.2, "en-gb")]


def test_empty_text_skips_model():
    k = FakeKokoro()
    audio, _ = KokoroEngine(k).synthesize("   ", "af_heart")
    assert audio.size == 0 and not k.calls


def test_voices_sorted():
    assert KokoroEngine(FakeKokoro()).voices() == ["af_heart", "bf_emma"]


def test_hungarian_is_offered_and_forwarded():
    from ttslts.tts.kokoro_engine import LANGUAGES

    assert "hu" in LANGUAGES
    k = FakeKokoro()
    KokoroEngine(k).synthesize("Szia!", "af_heart", 1.0, "hu")
    assert k.calls[0][3] == "hu"


def test_hungarian_text_phonemizes_with_kokoro_vocabulary():
    # Every phoneme espeak-ng produces for Hungarian must survive Kokoro's tokenizer,
    # otherwise sounds are silently dropped.
    from kokoro_onnx.tokenizer import Tokenizer

    t = Tokenizer()
    text = "Ez egy magyar nyelvű próbamondat: gyors beszéd, csütörtök, ősz, zöld, hűvös."
    phonemes = t.phonemize(text, "hu")
    assert phonemes
    assert t.known(phonemes) == "".join(c for c in phonemes if c in t.vocab)
    assert all(c == " " or c in t.vocab for c in phonemes)
