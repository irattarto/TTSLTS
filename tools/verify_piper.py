"""Checks the Hungarian Piper voices through the app's own download and engine code (CI)."""

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from ttslts.audio.dsp import estimate_f0, resample
from ttslts.tts import piper_download
from ttslts.tts.piper_engine import PiperVoice

TEXT = (
    "Szia! Ez egy magyar nyelvű próbamondat. A hangsúly mindig az első szótagon van, "
    "és a hosszú magánhangzók számítanak: például tűz, hűvös, csütörtök, ősz."
)
WIDE_TEXT = (
    "Árvíztűrő tükörfúrógép. Egy két három négy öt hat hét nyolc kilenc tíz. "
    "Mennyi az idő? Köszönöm szépen, viszontlátásra! Gyönyörű, nagyon szép."
)
NUMBERS = (
    "Dr. Kovács 12 almát vett, kb. 3,5 kg-ot, 1500 forintért. Ez az 1. kerület, pl. a Váci utca. "
    "A hőmérséklet 25 °C, a nyereség 45%."
)
out = Path(tempfile.mkdtemp(prefix="piper-verify-"))
failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS " if ok else "FAIL ") + message, flush=True)
    if not ok:
        failures.append(message)


installed = piper_download.download_voices(dest=out / "voices")
check(
    installed == list(piper_download.VOICES), f"all voices downloaded and size-checked: {installed}"
)

for voice_id in installed:
    model, config = piper_download.voice_paths(voice_id, out / "voices")
    voice = PiperVoice.load(str(model), str(config), use_gpu=False)
    check(
        voice.sample_rate == 22050 and voice.num_speakers == 1, f"{voice_id}: 22.05 kHz, 1 speaker"
    )
    phonemes = [p for s in (TEXT + " " + WIDE_TEXT).split(".") for p in voice.phonemize(s)]
    missing = sorted(voice.missing_phonemes(phonemes))
    check(not missing, f"{voice_id}: no phoneme missing from the voice map {missing}")
    audio, sr = voice.synthesize(TEXT)
    seconds = len(audio) / sr
    rms = float(np.sqrt((audio**2).mean()))
    print(f"  {voice_id}: {seconds:.1f} s, rms {rms:.3f}", flush=True)
    check(np.isfinite(audio).all() and 4 < seconds < 20, f"{voice_id}: sensible length")
    check(rms > 0.02, f"{voice_id}: not silent")
    fast, _ = voice.synthesize(TEXT, speed=1.5)
    check(len(fast) < len(audio) * 0.85, f"{voice_id}: speed 1.5 is shorter")

    def save(name: str, samples, voice_id=voice_id, sr=sr) -> None:
        with wave.open(str(Path("samples") / f"{voice_id}-{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((samples * 32767).astype("<i2").tobytes())

    f0 = estimate_f0(resample(audio, sr, 16000))
    voiced = f0[f0 > 0]
    pitch = float(np.median(voiced)) if voiced.size else 0.0
    print(f"PITCH {voice_id}: median {pitch:.0f} Hz over {voiced.size} voiced frames", flush=True)
    check(60 < pitch < 400, f"{voice_id}: speaking pitch is plausible")
    save("default", audio)
    steady, _ = voice.synthesize(TEXT, noise_scale=0.4, noise_w=0.5)
    lively, _ = voice.synthesize(TEXT, noise_scale=0.9, noise_w=1.1)
    save("steady", steady)
    save("lively", lively)
    numbers, _ = voice.synthesize(NUMBERS)
    check(
        np.isfinite(numbers).all() and len(numbers) / sr > 3,
        f"{voice_id}: numbers and abbreviations",
    )
    save("numbers", numbers)

if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("All checks passed")
