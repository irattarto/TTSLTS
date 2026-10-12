"""One-off probe: Piper's Hungarian voices, their license cards and a real synthesis test."""

import json
import urllib.request
import wave

import numpy as np
from huggingface_hub import hf_hub_download

from ttslts.tts.piper_engine import PiperVoice

REPO = "rhasspy/piper-voices"


def tree(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{path}?recursive=true"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


entries = tree("hu")
for e in entries:
    print(f"  {e['path']}  {e.get('size', '')}")

voices = sorted({e["path"].split("/")[2] for e in entries if e["path"].count("/") >= 2})
print("voices:", voices)

TEXT = (
    "Szia! Ez egy magyar nyelvű próbamondat. A hangsúly mindig az első szótagon van, "
    "és a hosszú magánhangzók számítanak: például tűz, hűvös, csütörtök, ősz."
)
for name in voices:
    base = f"hu/hu_HU/{name}/medium/hu_HU-{name}-medium"
    print(f"\n================ {name} ================")
    for card in (f"hu/hu_HU/{name}/medium/MODEL_CARD", f"hu/hu_HU/{name}/MODEL_CARD"):
        try:
            with open(hf_hub_download(REPO, card), encoding="utf-8") as f:
                print(f"--- {card} ---\n{f.read()[:3000]}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"(no {card}: {type(exc).__name__})")
    model = hf_hub_download(REPO, base + ".onnx")
    config = hf_hub_download(REPO, base + ".onnx.json")
    cfg = json.load(open(config, encoding="utf-8"))
    print("config keys:", sorted(cfg))
    print(
        "audio:",
        cfg["audio"],
        "| espeak:",
        cfg.get("espeak"),
        "| speakers:",
        cfg.get("num_speakers"),
    )
    print("inference:", cfg.get("inference"), "| phoneme_type:", cfg.get("phoneme_type"))
    print("map size:", len(cfg["phoneme_id_map"]), "| piper_version:", cfg.get("piper_version"))
    voice = PiperVoice.load(model, config, use_gpu=False)
    for i in voice.session.get_inputs():
        print("   in ", i.name, i.shape, i.type)
    for o in voice.session.get_outputs():
        print("   out", o.name, o.shape, o.type)
    phonemes = voice.phonemize(TEXT.split(".")[0] + ".")
    print("phonemes:", "".join(phonemes))
    allp = [p for s in TEXT.replace("!", ".").split(".") if s.strip() for p in voice.phonemize(s)]
    print("missing from map:", sorted(voice.missing_phonemes(allp)))
    audio, sr = voice.synthesize(TEXT)
    rms = float(np.sqrt((audio**2).mean()))
    print(
        f"synth: {len(audio) / sr:.2f} s at {sr} Hz, rms {rms:.3f}, peak {np.abs(audio).max():.3f}"
    )
    with wave.open(f"/tmp/{name}.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
