# TTSLTS

An offline system tray app for Windows with two features:

- **Text to speech** with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), 54 preset voices in 9 languages.
- **Real-time voice changing** with [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) voice models.
- **Training your own voice** from mp3 or wav recordings, saved as an RVC v2 `.pth` and an `.onnx` file (PyTorch is set up on first use).

Everything runs locally through ONNX Runtime (training uses PyTorch). Nothing is sent over the network except the one-time model downloads.

## Install

Requires Python 3.10–3.13. A GPU is needed for the real-time voice changer (a CPU is about four times too slow); text to speech runs fine on CPU. The Windows installer and portable build use the GPU through DirectML, which works with NVIDIA, AMD and Intel graphics cards.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .

:: NVIDIA GPU (CUDA 12): swap in the CUDA build of ONNX Runtime
pip install -e .[gpu]
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu

:: or any Windows GPU (NVIDIA, AMD, Intel) with DirectML instead of CUDA
pip uninstall -y onnxruntime
pip install onnxruntime-directml
```

Run it with `ttslts` (or `pythonw -m ttslts` to start without a console window).

## Using it

The app lives in the system tray. Click the tray icon to show or hide the window; right-click for **Voice changer** on/off, **Speak clipboard**, **Stop speaking** and **Quit**.

**Settings > Window**

| Option | Default | What it does |
|---|---|---|
| Minimize to tray | on | Minimizing hides the window to the tray instead of the taskbar |
| Close to tray | on | The window's close button hides it to the tray; use **Quit** in the tray menu to exit |
| Start minimized to tray | off | Start with only the tray icon showing |

Settings are saved automatically to `%APPDATA%\TTSLTS\settings.json`.

### Text to speech

1. Settings > Models > **Download Kokoro** (about 340 MB, saved to `%APPDATA%\TTSLTS\models\kokoro`). You can also point the two fields at your own `kokoro-v1.0.onnx` and `voices-v1.0.bin`.
2. On the **Text to Speech** tab, type text, pick a voice, language and speed, then **Speak** or **Save WAV…**.
3. **Play on** picks the output device. Choose a virtual cable to send speech into Discord, games or OBS.
   **Hungarian:** choosing Hungarian as the language switches to the **Piper (Hungarian voices)** engine by default (you can set the engine back to Kokoro by hand). If the voices are not downloaded yet you are offered the download (Settings > Models > *Download Hungarian voices*) (Settings > Models > *Download Hungarian voices*, about 190 MB, from the original [Piper voice collection](https://huggingface.co/rhasspy/piper-voices); you are asked to accept the terms first). The voices Anna, Berta and Imre use CC0 training data but were fine-tuned from Piper's US English *lessac* voice, whose data is licensed for research use, so keep them to research and personal use. Voice variation (default 0.67) and Rhythm variation (default 0.80) make Piper steadier when lowered and livelier when raised, and every setting has hover help. Anna, Berta and Imre are three different speakers, so listen to each with your own text and keep the one you like. Common Hungarian abbreviations (kb., pl., dr., stb.) and units are read as words.

   Kokoro's own **Hungarian (experimental)** language option, by contrast, is not a Kokoro language, so no voice was trained on it. Choosing it makes espeak-ng pronounce the Hungarian text and one of the existing voices read it, so expect a foreign accent. The Android app has the same option.
4. Tick **Speak with my voice model (RVC)** to run Kokoro's output through your RVC voice, giving TTS your custom voice.

### Voice changer

You supply the models (they are never bundled or committed):

| Field | File | Notes |
|---|---|---|
| RVC v2 voice | a downloaded `.pth` voice, or an `.onnx` export | The `.pth` file that voices are shared as works directly: the first use converts it to ONNX on your computer (a few seconds, cached). Only **v2** voices with pitch work (the title or model card says v2; 768-dim). Older `.onnx` exports from RVC WebUI / w-okada also work, in both input layouts and fp16. |
| ContentVec | `vec-768-layer-12.onnx` (or another 768-dim ContentVec/HuBERT ONNX) | RVC v2 models use 768-dim features. |
| RMVPE (optional) | `rmvpe.onnx` from the w-okada voice changer | Better pitch tracking. Without it, a built-in YIN pitch tracker is used. |
| RVC sample rate | 32/40/48 kHz | Only used when the ONNX file doesn't record its own rate (w-okada exports do). |

Then on the **Voice Changer** tab pick your microphone and output and press **Start**. To use the changed voice in other apps, set Output to a virtual cable such as [VB-Audio CABLE](https://vb-audio.com/Cable/) (*CABLE Input*) and select *CABLE Output* as the microphone in the other app.

**Index rate** and **Protect consonants** (Voice Changer tab) make a voice trained in TTSLTS clearer and closer to the real recordings: training saves the features of your recordings next to the voice (`<name>.index.npy`, made on first use for older voices), and each frame of the converted speech is mixed with its closest real frames. Index rate 0.3-0.7 is a good range (0 switches it off, too high can sound rough); Protect keeps s, sh, f and breath sounds clear. Voices from elsewhere have no index, so these two do nothing for them. Joins between chunks are lined up with SOLA, and only the newest part of each chunk is synthesized, which keeps the delay and the load down.

Latency is roughly chunk size + crossfade + processing time. Lower **Chunk size** for less delay, raise it if the voice crackles. **Context** is how much earlier audio the model sees each time (more is smoother but slower). The **Processing** readout should stay well under the chunk size.

### Train your own voice

The **Train** tab makes an RVC v2 voice from recordings of one speaker, in the installed app as well as from source. PyTorch is too big to ship in the installers (the CUDA build is several GB), so the first time you press **Start training** the app asks, then sets up its own private environment in `%APPDATA%\TTSLTS\runtime` (Python from python.org, PyTorch and a few packages; about 3 GB with an NVIDIA GPU, 0.5 GB without; training on a CPU takes many hours). Delete that folder to remove it. Running from source with PyTorch already installed (`pip install -e .[train]`) uses that instead and downloads nothing.

1. Settings > Models > **Download ContentVec + RMVPE** (once, also needed by the voice changer).
2. On the **Train** tab add your recordings (mp3, wav, flac, ogg). Use clean speech of one person without music or noise; about 10 minutes gives a good voice and 2-3 minutes already works.
3. Give the voice a name, pick the sample rate (40 kHz is the usual choice), epochs (100-300 for a few minutes of audio) and batch size (lower it if the GPU runs out of memory), then **Start training**. The first time you are asked to accept the terms and download the pretrained RVC v2 models (about 215 MB) that training starts from.
4. Training cuts the recordings at pauses, extracts ContentVec features and pitch, and fine-tunes the pretrained model. The voice file is refreshed every *Save every* epochs, so you can try it before training ends. **Stop** saves and ends early; starting the same name again offers to continue where it stopped.
5. The result is saved to `%APPDATA%\TTSLTS\voices\<name>\` as `<name>.pth` (the format voices are shared in, works in other RVC tools) and `<name>.onnx`, plus `<name>.index.npy` (the voice index) and `<name>.json` (the voice's natural pitch). **Use this voice** selects it for the voice changer and for *Speak with my voice model*.

Training runs in a separate process (`python -m ttslts.train.run`, from the app's bundled sources in the installed app); its log is `%APPDATA%\TTSLTS\train\<name>\train.log`. Only train on voices you have permission to use.

## Development

```bash
pip install -e .[dev]
ruff check . && ruff format --check .
pytest
```

Layout:

```
src/ttslts/
  app.py            entry point (QApplication + tray)
  controller.py     background work for TTS, voice changer, downloads
  settings.py       JSON settings (tray options default on)
  audio/            DSP helpers (resampling, YIN pitch, crossfade), devices, WAV
  tts/              Kokoro engine and model download
  vc/rvc.py         RVC v2 ONNX inference (ContentVec + pitch + voice model)
  vc/rvc_pth.py     .pth voice checkpoint -> ONNX importer (no PyTorch needed)
  vc/realtime.py    chunked streaming conversion with crossfades
  train/            voice training: prepare.py (cut clips, features), trainer.py (loop), run.py (job process)
  train/rvc_model/  original RVC network code (MIT, see its LICENSE), used only for training
  ui/               main window, tray icon and menu
```

## Licenses and consent

Kokoro is Apache-2.0 and RVC is MIT, so both are usable commercially, but check the license of each voice model you use. The pretrained models used for training carry the RVC author's research-use clause (shown when you download them), and `train/rvc_model/` is RVC's MIT-licensed model code. Only clone voices you have permission to use.

## Android app (text to speech only)

`android/` contains a native Android app that runs Kokoro and the Hungarian Piper voices offline through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). It does text to speech only; the RVC voice changer is desktop-only because real-time voice conversion is too heavy for phones.

- On first launch the app downloads the Kokoro model once (several hundred MB), then works offline.
- The **Engine** choice switches to **Piper (Hungarian voices)**; choosing Hungarian as the language does this by default. Anna, Berta and Imre download on demand (about 65 MB each) after you accept their terms (research and personal use).
- Pick a voice and speed, then **Speak**, or **Save WAV**. Sharing text to TTSLTS from another app also fills the text box.
- Download `TTSLTS-android.apk` from the latest [release](../../releases) and allow installing from unknown sources. Release APKs are debug-signed, so installing a newer build over an older one may require uninstalling first.

Build it yourself (JDK 17, Android SDK, Gradle 8.9+):

```sh
./android/fetch-native.sh          # sherpa-onnx native libraries
cd android && gradle assembleDebug # APK in app/build/outputs/apk/debug/
```

## Releases

Pushing a tag such as `v0.1.0` (or running the Release workflow by hand from the Actions tab with a tag name, which creates the tag) builds these downloads and attaches them to a GitHub pre-release:

| File | For |
|---|---|
| `TTSLTS-Setup.exe` | Windows installer (per-user, no administrator needed) |
| `TTSLTS-windows.zip` | Windows portable folder, run `TTSLTS.exe` |
| `ttslts_<version>_amd64.deb` | Linux installer for Debian, Ubuntu and derivatives (`sudo apt install ./ttslts_*.deb`) |
| `TTSLTS-linux-x86_64.tar.gz` | Linux portable folder, run `TTSLTS` |
| `TTSLTS-android.apk` | Android, text to speech only |

On Linux the tray icon needs a desktop that shows tray icons (GNOME needs an extension).
