"""F5-TTS reader process: runs in the F5 environment (``f5_runtime``), driven by ``f5_engine``.

Protocol, one JSON object per line. In on stdin:

  {"op": "load", "model": ..., "vocab": ..., "ref_audio": ..., "ref_text": ..., "device": "cuda",
   "whisper_dir": ...}
  {"op": "say", "text": "one sentence", "speed": 1.0, "seed": 3, "out": "x.npy"}

out on stdout: {"type": "loaded", "device": ...}, {"type": "said", "out": ..., "sr": 24000,
"ok": true} or {"type": "error", "message": ...}. Everything else the libraries print goes to
stderr, so stdout stays clean.
"""

from __future__ import annotations

import json
import sys
import time
import traceback

TRIES = 2  # readings per sentence when the start comes out wrong or the melody flat
WHISPER_S = 3.0  # only the start is transcribed: enough to find the first two words


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # library prints must not mix into the protocol

    def send(**msg):
        out.write(json.dumps(msg) + "\n")
        out.flush()

    reader = None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        try:
            if msg.get("op") == "load":
                reader = Reader(msg)
                send(type="loaded", device=reader.device)
            elif msg.get("op") == "say":
                if reader is None:
                    raise RuntimeError("no voice loaded")
                send(type="said", **reader.say(msg))
            elif msg.get("op") == "quit":
                return
        except Exception as exc:  # reported to the app, which shows it
            traceback.print_exc()
            send(type="error", message=f"{type(exc).__name__}: {exc}")


class Reader:
    def __init__(self, msg: dict):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        def load(fp, *a, **kw):  # without torchcodec's FFmpeg DLLs, which are not there
            data, sr = sf.read(str(fp), dtype="float32")
            return torch.from_numpy(data[np.newaxis, :] if data.ndim == 1 else data.T), sr

        torchaudio.load = load
        from f5_tts.api import F5TTS

        from ttslts.tts.f5_text import reference_text

        self.np, self.sf, self.torch, self.torchaudio = np, sf, torch, torchaudio
        self.device = msg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.vocab = set(open(msg["vocab"], encoding="utf-8").read().split("\n"))
        self.model = F5TTS(
            model="F5TTS_v1_Base", ckpt_file=msg["model"], vocab_file=msg["vocab"],
            device=self.device, use_ema=True,
        )  # fmt: skip
        self.ref_audio = msg["ref_audio"]
        ref = open(msg["ref_text"], encoding="utf-8").read()
        self.ref_text = reference_text(ref, self.vocab)
        self.whisper_dir = msg.get("whisper_dir")
        self._asr = None

    def asr(self):
        """Whisper small, on the graphics card when F5 is (0.5 s per start check instead of 5-9
        on the CPU). It uses the CUDA libraries that come with PyTorch."""
        if self._asr is None:
            import os

            from faster_whisper import WhisperModel

            if self.device == "cuda":
                try:
                    os.add_dll_directory(os.path.join(os.path.dirname(self.torch.__file__), "lib"))
                    self._asr = WhisperModel(
                        "small", device="cuda", compute_type="float16",
                        download_root=self.whisper_dir,
                    )  # fmt: skip
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"Whisper on the GPU failed ({exc}); using the CPU", file=sys.stderr)
            if self._asr is None:
                self._asr = WhisperModel(
                    "small", device="cpu", compute_type="int8", download_root=self.whisper_dir
                )
        return self._asr

    def words(self, wav, sr) -> list[tuple[str, float, float]]:
        x = self.torch.from_numpy(wav[: int(WHISPER_S * sr)])
        x16 = self.torchaudio.functional.resample(x, sr, 16000).numpy()
        segments, _ = self.asr().transcribe(
            x16, language="hu", word_timestamps=True, condition_on_previous_text=False
        )
        return [(w.word, w.start, w.end) for s in segments for w in (s.words or [])]

    def say(self, msg: dict) -> dict:
        from f5_tts.infer.utils_infer import infer_process
        from f5_tts.model.utils import seed_everything

        from ttslts.tts.f5_text import (
            FILLER,
            FILLER_MAX_S,
            NO_FILLER_CHARS,
            filler_cut,
            is_lively,
            keep_enough,
            leading_silence,
            melody_range,
            prepare_text,
            starts_with_text,
        )

        np = self.np
        text = prepare_text(msg["text"], self.vocab)
        readings = []
        # a word or two alone (Pronunciations > Listen): no filler. Whisper cannot tell such a
        # word from the filler in front of it, and the cut took most of the word away
        filler = "" if len(text) < NO_FILLER_CHARS else FILLER
        for attempt in range(TRIES):
            t0 = time.perf_counter()
            seed_everything(int(msg.get("seed", 0)) + 1000 * attempt)
            wav, sr, _ = infer_process(
                self.ref_audio, self.ref_text, filler + text, self.model.ema_model,
                self.model.vocoder, self.model.mel_spec_type, show_info=lambda *a: None,
                speed=float(msg.get("speed", 1.0)), nfe_step=int(msg.get("nfe", 32)),
                device=self.device,
            )  # fmt: skip
            wav = np.asarray(wav, dtype=np.float32)
            t1 = time.perf_counter()
            if filler:
                words = self.words(wav, sr)
                print(
                    "heard: " + " ".join(f"{w}@{s:.2f}" for w, s, _ in words[:5]), file=sys.stderr
                )
                cut, ok = filler_cut(words, text)
                # the filler is said within about a second of the speech starting: a later cut
                # means Whisper matched the wrong word (it once took "itt rendezték meg" away)
                latest = leading_silence(wav, sr) + FILLER_MAX_S
                if cut > latest:
                    cut, ok = latest, False
                safe = keep_enough(cut, len(wav) / sr, text)  # never the text itself cut away
                ok = bool(ok and safe == cut)
                cut = safe
            else:
                cut, ok = leading_silence(wav, sr), True
            check = time.perf_counter() - t1
            wav = wav[int(cut * sr) :].copy()
            if filler and ok:  # check the cut by listening to the new start
                after = self.words(wav, sr)
                ok = starts_with_text(after, text)
                if not ok:
                    print("after the cut: " + " ".join(w for w, _, _ in after[:3]), file=sys.stderr)
            fade = min(len(wav), int(0.01 * sr))
            wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            melody = melody_range(wav, sr)
            verdict = "ok" if ok else "WRONG"
            print(
                f"read {len(wav) / sr:.1f} s in {t1 - t0:.1f} s, start check {check:.1f} s, "
                f"cut at {cut:.2f} s, start {verdict}, melody {melody:.1f} st (try {attempt + 1})",
                file=sys.stderr,
            )
            readings.append((ok, melody, wav))
            if ok and is_lively(melody, text):
                break
        ok, melody, wav = max(readings, key=lambda r: (r[0], r[1]))  # a good start, then melody
        np.save(msg["out"], wav)  # float32 samples; the app has numpy but not soundfile
        return {"out": msg["out"], "sr": int(sr), "ok": bool(ok), "melody": float(melody)}


if __name__ == "__main__":
    main()
