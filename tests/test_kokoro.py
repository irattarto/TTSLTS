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
